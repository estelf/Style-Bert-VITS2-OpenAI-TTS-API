"""model_assets/ の走査と TTSModel インスタンスの常駐管理。

コアの `TTSModelHolder` は「現在の 1 モデル」しか保持しない GUI 向けの設計のため、
サーバーでは自前のレジストリを持つ。方針:

- 走査は起動時 1 回（`refresh()`）。以降 `entries` は不変として読み取り専用で公開する
- `TTSModel` は生成自体は軽く（config/npy の読み込みのみ）、`load()` で重みをデバイスへ載せる。
  初回リクエスト時に遅延ロードし、`max_resident` を超えた分は LRU で `unload()` する
- 合成（とロード/アンロード）は `synthesis_lock` で直列化する。これにより
  「合成中のモデルが別スレッドのアンロードで消える」ことが起きない
- 日本語 BERT / pyopenjtalk ワーカーはコア側でプロセス全体共有なので、モデル切替でも再ロード不要
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path

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


@dataclass(frozen=True)
class ModelEntry:
    """model_assets/ 配下の 1 モデル分の登録情報（起動時スキャン結果、不変）"""

    name: str
    dir: Path
    #: チェックポイントファイル一覧（更新日時が新しい順）
    files: list[Path]
    #: 話者名一覧（config.json の spk2id のキー順）
    speakers: list[str]
    spk2id: dict[str, int]

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

        #: model 名 -> 登録情報。refresh() でのみ更新され、その後は読み取り専用
        self.entries: dict[str, ModelEntry] = {}

        self._loaded: dict[str, TTSModel] = {}
        self._order: list[str] = []  # 使用順（先頭が最も古い＝退避候補）
        #: _loaded/_order の短锁（健康チェック等の読み取り用。重い処理は synthesis_lock の下で行う）
        self._state_lock = threading.Lock()

        #: 合成（とモデルのロード/アンロード）を直列化するロック。ルートハンドラが保持する
        self.synthesis_lock = threading.Lock()

    # --- 走査 ---

    def refresh(self) -> None:
        """root を走査して登録情報を構築する（起動時のみ呼ぶ）"""
        entries: dict[str, ModelEntry] = {}

        if not self.root.exists():
            logger.warning(f"Model root {self.root} does not exist. No models registered.")
            self.entries = {}
            return

        for model_dir in sorted(d for d in self.root.iterdir() if d.is_dir()):
            if model_dir.name.startswith("."):
                continue
            config_path = model_dir / "config.json"
            if not config_path.exists():
                logger.warning(f"Skip {model_dir}: config.json not found")
                continue
            style_vec_path = model_dir / "style_vectors.npy"
            if not style_vec_path.exists():
                logger.warning(f"Skip {model_dir}: style_vectors.npy not found")
                continue
            files = sorted(
                (
                    f
                    for f in model_dir.iterdir()
                    if f.suffix in (".pth", ".pt", ".safetensors")
                    and not f.name.startswith(".")
                ),
                key=lambda f: f.stat().st_mtime,
                reverse=True,
            )
            if not files:
                logger.warning(f"Skip {model_dir}: no model file (.safetensors/.pth/.pt)")
                continue
            try:
                hps = HyperParameters.load_from_json(config_path)
            except Exception as e:
                logger.warning(f"Skip {model_dir}: failed to load config.json ({e})")
                continue
            entries[model_dir.name] = ModelEntry(
                name=model_dir.name,
                dir=model_dir,
                files=files,
                speakers=list(hps.data.spk2id.keys()),
                spk2id=dict(hps.data.spk2id),
            )

        self.entries = entries
        logger.info(
            f"Registered {len(entries)} model(s) under {self.root}: {list(entries.keys())}"
        )

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

        **呼び出し側は `synthesis_lock` を保持していること**（ロード中の退避が競合するため）。
        """
        entry = self.get_entry(model_name)

        model = self._loaded.get(model_name)
        if model is not None:
            with self._state_lock:
                self._move_to_end(model_name)
            return model

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
                self._order.clear()

    # --- 内部 ---

    def _move_to_end(self, name: str) -> None:
        if name in self._order:
            self._order.remove(name)
        self._order.append(name)

    def _evict_locked(self) -> None:
        """常駐上限を超える分を最も古いものからアンロード（_state_lock 保持中に行う）"""
        while len(self._loaded) > self.max_resident:
            victim_name = self._order.pop(0)
            victim = self._loaded.pop(victim_name)
            logger.info(f"Model `{victim_name}` exceeds max_resident_models. Unloading ...")
            victim.unload()
