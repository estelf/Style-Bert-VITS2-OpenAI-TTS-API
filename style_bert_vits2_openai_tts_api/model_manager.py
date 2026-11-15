"""model_assets/ の走査と TTSModel インスタンスの常駐管理。

コアの `TTSModelHolder` は「現在の 1 モデル」しか保持しない GUI 向けの設計のため、
サーバーでは自前のレジストリを持つ。方針:

- 走査（`refresh()`）は起動時に加え、各リクエストの処理前にも行う（モデル追加に再起動不要）。
  変化検出はディレクトリ一覧と stat のみで行い、変わっていないモデルは config.json の
  再パースをスキップする。差分が無い限り `entries` は差し替えない
- `entries` は読み取り専用スナップショットとして公開する。更新時は新 dict を作って
  `self.entries` を再束縛するだけなので、合成スレッドが古いスナップショットを読んでも安全
- `TTSModel` は生成自体は軽く（config/npy の読み込みのみ）、`load()` で重みをデバイスへ載せる。
  初回リクエスト時に遅延ロードし、`max_resident` を超えた分は LRU で `unload()` する
- 合成（とロード/アンロード）は `synthesis_lock` で直列化する。これにより
  「合成中のモデルが別スレッドのアンロードで消える」ことが起きない。
  処理中に model_assets が変わった場合も、ディスク上の指紋（stat 値）を照合して
  次回合成時に自動的に再ロードする。削除されたモデルの常駐解放もこのタイミング（=
  合成が走っていないとき）にしか行わない
- 日本語 BERT / pyopenjtalk ワーカーはコア側でプロセス全体共有なので、モデル切替でも再ロード不要
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from style_bert_vits2.logging import logger
from style_bert_vits2.models.hyper_parameters import HyperParameters
from style_bert_vits2.tts_model import TTSModel


class ModelNotFound(Exception):
    """model_assets/ に指定モデルが無い（利用可能なモデル名を一覧に持つ）"""

    def __init__(self, name: str, available: list[str]) -> None:
        self.name = name
        self.available = available
        super().__init__(f"Model `{name}` is not found")


class VoiceNotFound(Exception):
    """指定モデルに指定話者が無い（利用可能な話者名を一覧に持つ）"""

    def __init__(self, model: str, voice: str, voices: list[str]) -> None:
        self.model = model
        self.voice = voice
        self.voices = voices
        super().__init__(f"Voice `{voice}` is not found in model `{model}`")


#: ディスク上のファイル 1 つの指紋（(mtime_ns, size)。変化検出の比較のみに使う）
FileStat = tuple[int, int]


@dataclass(frozen=True)
class ModelEntry:
    """model_assets/ 配下の 1 モデル分の登録情報（スキャン結果のスナップショット、不変）"""

    name: str
    dir: Path
    #: チェックポイントファイル一覧（更新日時が新しい順）
    files: list[Path]
    #: 話者名一覧（config.json の spk2id のキー順）
    speakers: list[str]
    spk2id: dict[str, int]
    #: モデルディレクトリ全体の指紋（config / style_vectors / 各チェックポイントの
    #: 名前と stat 値）。refresh() の変化検出と、常駐中のモデルの再ロード判定に使う
    signature: tuple[Any, ...]

    @property
    def default_file(self) -> Path:
        """推論に使う既定チェックポイント。best.safetensors があればそれを優先する
        （コアの TTSModelHolder.default_model_file と同じ規約）"""
        for f in self.files:
            if f.name == "best.safetensors":
                return f
        return self.files[0]

    @property
    def config_path(self) -> Path:
        return self.dir / "config.json"

    @property
    def style_vec_path(self) -> Path:
        return self.dir / "style_vectors.npy"


def _stat_of(path: Path) -> FileStat:
    st = path.stat()
    return (st.st_mtime_ns, st.st_size)


class ModelRegistry:
    """合成モデルの登録・遅延ロード・LRU 常駐管理（スレッドセーフの契約はモジュール docstring 参照）"""

    def __init__(
        self,
        root: Path,
        device: str,
        dtype: str = "float16",
        max_resident: int = 2,
    ) -> None:
        self.root = root
        self.device = device
        self.dtype = dtype
        self.max_resident = max_resident

        #: model 名 -> 登録情報。更新時は新 dict を再束縛する（既存 dict は変更しない）
        self.entries: dict[str, ModelEntry] = {}

        self._loaded: dict[str, TTSModel] = {}
        #: 常駐モデルのロード時指紋。entries の signature と違えば再ロードする
        self._loaded_sig: dict[str, tuple[Any, ...]] = {}
        self._order: list[str] = []  # 使用順（先頭が最も古い＝退避候補）
        #: 登録から外れたが合成中かもしれないので unload を保留したインスタンス
        #: （次の合成 = synthesis_lock 取得時に解放する）
        self._pending_unload: dict[str, TTSModel] = {}
        #: _loaded/_order/_loaded_sig/_pending_unload の短锁（健康チェック等の読み取り用。
        #: 重い処理は synthesis_lock の下で行う）
        self._state_lock = threading.Lock()

        #: refresh() 同士を直列化するロック（リクエストごとに呼ばれる）
        self._scan_lock = threading.Lock()
        #: 前回スキャンのディレクトリ指紋。これと一致すれば走査結果は不変
        self._signatures: dict[str, tuple[Any, ...]] = {}
        #: config.json 指紋 -> 解析結果（spk2id 等）。変化した config だけ再パースする
        self._config_cache: dict[tuple[Any, ...], tuple[list[str], dict[str, int]]] = {}
        #: 前回スキャン時の「スキップ理由」（モデル名 -> 理由）。変化があったときだけ warns
        self._skip_reasons: dict[str, str] = {}
        self._root_missing_warned = False

        #: 合成（とモデルのロード/アンロード）を直列化するロック。ルートハンドラが保持する
        self.synthesis_lock = threading.Lock()

    # --- 走査 ---

    def refresh(self) -> bool:
        """root を走査し、model_assets に変化があれば登録を更新する。

        起動時と、各リクエストの処理前に呼ぶ（モデル追加・更新のホットリロード）。
        変化検出はディレクトリ一覧と stat のみ。差分が無ければ config.json を読み直さず
        False を返すので、リクエストごとに呼んでも実務上コストは無視できる。

        Returns:
            登録内容（モデルの追加・削除・更新）が変わったかどうか
        """
        with self._scan_lock:
            scanned, skip_reasons = self._scan_root()
            self._log_skip_changes(skip_reasons)

            if scanned == self._signatures:
                return False

            old_signatures = self._signatures
            entries: dict[str, ModelEntry] = {}
            for name, fingerprint in scanned.items():
                if old_signatures.get(name) == fingerprint:
                    # 無変化のモデルは前回スナップショットをそのまま再利用する
                    entries[name] = self.entries[name]
                else:
                    entries[name] = self._build_entry(name, fingerprint)

            self.entries = entries
            self._signatures = scanned
            # 存在しなくなった config 用のキャッシュを捨てて肥大化するのを防ぐ
            alive = {(str(self.root / n / "config.json"), *fp[0]) for n, fp in scanned.items()}
            self._config_cache = {k: v for k, v in self._config_cache.items() if k in alive}
            self._log_changes(old_signatures, scanned)
            self._forget_removed(set(old_signatures) - set(scanned))
            return True

    def _scan_root(self) -> tuple[dict[str, tuple[Any, ...]], dict[str, str]]:
        """root を歩いて valid なモデルディレクトリの指紋を作る。

        Returns:
            ({モデル名: 指紋}, {モデル名: スキップ理由})。root 自体が無い場合は空。
        """
        scanned: dict[str, tuple[Any, ...]] = {}
        skip_reasons: dict[str, str] = {}

        if not self.root.exists():
            if not self._root_missing_warned:
                logger.warning(f"Model root {self.root} does not exist. No models registered.")
                self._root_missing_warned = True
            return scanned, skip_reasons
        self._root_missing_warned = False

        for model_dir in sorted(d for d in self.root.iterdir() if d.is_dir()):
            if model_dir.name.startswith("."):
                continue
            fingerprint, reason = self._fingerprint(model_dir)
            if fingerprint is None:
                skip_reasons[model_dir.name] = reason
            else:
                scanned[model_dir.name] = fingerprint
        return scanned, skip_reasons

    def _fingerprint(self, model_dir: Path) -> tuple[tuple[Any, ...] | None, str]:
        """モデルディレクトリ 1 つを stat し、推論に必要なファイルが揃っていれば指紋を返す。

        揃っていなければ (None, スキップ理由)。config.json の中身はパースしない
        （解析は指紋が変わったモデルだけ）。
        """
        config_path = model_dir / "config.json"
        if not config_path.exists():
            return None, "config.json not found"
        style_vec_path = model_dir / "style_vectors.npy"
        if not style_vec_path.exists():
            return None, "style_vectors.npy not found"
        checkpoints = sorted(
            f
            for f in model_dir.iterdir()
            if f.suffix in (".pth", ".pt", ".safetensors")
            and not f.name.startswith(".")
        )
        if not checkpoints:
            return None, "no model file (.safetensors/.pth/.pt)"
        fingerprint = (
            _stat_of(config_path),
            _stat_of(style_vec_path),
            tuple((f.name, *_stat_of(f)) for f in checkpoints),
        )
        return fingerprint, ""

    def _build_entry(self, name: str, fingerprint: tuple[Any, ...]) -> ModelEntry:
        """指紋の変わったモデルの登録情報を構築する。config.json は指紋込みでキャッシュし、
        変化が無い限り解析を繰り返さない"""
        config_stat, _style_stat, checkpoints = fingerprint
        model_dir = self.root / name
        config_path = model_dir / "config.json"

        cache_key: tuple[Any, ...] = (str(config_path), *config_stat)
        cached = self._config_cache.get(cache_key)
        if cached is None:
            hps = HyperParameters.load_from_json(config_path)
            cached = (list(hps.data.spk2id.keys()), dict(hps.data.spk2id))
            self._config_cache[cache_key] = cached
        speakers, spk2id = cached

        # 走査時のファイル一覧は指紋に焼き込んでいるので、Path は作り直す
        files = sorted(
            (model_dir / c[0] for c in checkpoints),
            key=lambda f: f.stat().st_mtime,
            reverse=True,
        )

        return ModelEntry(
            name=name,
            dir=model_dir,
            files=files,
            speakers=speakers,
            spk2id=spk2id,
            signature=fingerprint,
        )

    def _log_skip_changes(self, skip_reasons: dict[str, str]) -> None:
        """スキップ理由が新規/変化した場合のみ warning を出す（無効ディレクトリを
        リクエストごとに繰り返し警告しない）"""
        for name, reason in skip_reasons.items():
            if self._skip_reasons.get(name) != reason:
                logger.warning(f"Skip {self.root / name}: {reason}")
        self._skip_reasons = skip_reasons

    def _log_changes(
        self,
        old: dict[str, tuple[Any, ...]],
        new: dict[str, tuple[Any, ...]],
    ) -> None:
        for name in sorted(set(new) - set(old)):
            entry = self.entries[name]
            logger.info(
                f"Registered model `{name}` (speakers: {entry.speakers})"
            )
        for name in sorted(set(old) - set(new)):
            logger.info(f"Unregistered model `{name}` (directory removed or invalid)")
        for name in sorted(set(old) & set(new)):
            if old[name] != new[name]:
                logger.info(f"Model `{name}` changed on disk. Will reload on next synthesis.")
        if old or new:
            logger.info(f"Model list now: {list(new.keys())}")

    def _forget_removed(self, removed: set[str]) -> None:
        """登録から外れたモデルを参照から外す。常駐中のインスタンスは合成中かもしれないため
        ここでは unload せず、次の合成（synthesis_lock 取得済み）のときに解放する"""
        if not removed:
            return
        with self._state_lock:
            for name in removed:
                self._loaded_sig.pop(name, None)
                if name in self._order:
                    self._order.remove(name)
                model = self._loaded.pop(name, None)
                if model is not None:
                    self._pending_unload[name] = model

    # --- 検証 ---

    def get_entry(self, model_name: str) -> ModelEntry:
        """登録済みモデルの取得（無ければ ModelNotFound）"""
        entry = self.entries.get(model_name)
        if entry is None:
            raise ModelNotFound(model_name, self.model_names)
        return entry

    def speaker_id(self, entry: ModelEntry, voice: str | None) -> int:
        """voice（話者名）をコアの speaker_id へ変換（無ければ VoiceNotFound）。

        voice が None（リクエストで省略・空文字）の場合はモデルの先頭話者を選択する。
        本サーバーは 1 モデル = 1 話者前提なので、実質的に選択の余地は無い。
        """
        if voice is None:
            if not entry.spk2id:
                raise VoiceNotFound(entry.name, "(unspecified)", entry.speakers)
            return next(iter(entry.spk2id.values()))
        if voice not in entry.spk2id:
            raise VoiceNotFound(entry.name, voice, entry.speakers)
        return entry.spk2id[voice]

    # --- 常駐管理（synthesis_lock の下でのみ変更する） ---

    @property
    def model_names(self) -> list[str]:
        return list(self.entries.keys())

    def loaded_names(self) -> list[str]:
        with self._state_lock:
            return list(self._loaded.keys())

    def ensure_loaded(self, model_name: str) -> TTSModel:
        """モデルをデバイスにロードして返す。既にロード済みならそれを返す。
        ディスク上のファイルが変わっている場合はアンロードしてから再ロードする。

        **呼び出し側は `synthesis_lock` を保持していること**（ロード中の退避が競合するため。
        またこの前提の下でのみ、削除済みモデルの保留中 unload をまとめて行える）。
        """
        entry = self.get_entry(model_name)
        self._drain_pending_unloads()

        model = self._loaded.get(model_name)
        if model is not None:
            if self._loaded_sig.get(model_name) == entry.signature:
                with self._state_lock:
                    self._move_to_end(model_name)
                return model
            # 合成していない状態でしか呼ばれないので、即 unload して問題ない
            logger.info(f"Model `{model_name}` changed on disk. Reloading ...")
            with self._state_lock:
                self._loaded.pop(model_name, None)
                self._loaded_sig.pop(model_name, None)
                if model_name in self._order:
                    self._order.remove(model_name)
            model.unload()

        logger.info(f"Loading model `{model_name}` ({entry.default_file.name}) on {self.device} ...")
        model = TTSModel(
            model_path=entry.default_file,
            config_path=entry.config_path,
            style_vec_path=entry.style_vec_path,
            device=self.device,
            dtype=self.dtype,
        )
        model.load()

        with self._state_lock:
            self._loaded[model_name] = model
            self._loaded_sig[model_name] = entry.signature
            self._move_to_end(model_name)
            self._evict_locked()
        return model

    def unload_all(self) -> None:
        """常駐中の全モデルを解放する（シャットダウン用）"""
        with self.synthesis_lock:
            for name, model in list(self._loaded.items()):
                logger.info(f"Unloading model `{name}` ...")
                model.unload()
            with self._state_lock:
                self._loaded.clear()
                self._loaded_sig.clear()
                self._order.clear()
                pending = self._pending_unload
                self._pending_unload = {}
            for name, model in pending.items():
                logger.info(f"Unloading removed model `{name}` ...")
                model.unload()

    # --- 内部 ---

    def _drain_pending_unloads(self) -> None:
        """登録削除時に保留したモデルを実際に解放する。
        synthesis_lock を保持した状態（= どの合成も走っていない）でのみ呼ぶこと。"""
        with self._state_lock:
            if not self._pending_unload:
                return
            pending = self._pending_unload
            self._pending_unload = {}
        for name, model in pending.items():
            logger.info(f"Unloading removed model `{name}` ...")
            model.unload()

    def _move_to_end(self, name: str) -> None:
        if name in self._order:
            self._order.remove(name)
        self._order.append(name)

    def _evict_locked(self) -> None:
        """常駐上限を超える分を最も古いものからアンロード（_state_lock 保持中に行う）"""
        while len(self._loaded) > self.max_resident:
            victim_name = self._order.pop(0)
            victim = self._loaded.pop(victim_name)
            self._loaded_sig.pop(victim_name, None)
            logger.info(f"Model `{victim_name}` exceeds max_resident_models. Unloading ...")
            victim.unload()
