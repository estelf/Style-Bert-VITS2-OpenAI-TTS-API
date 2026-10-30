"""サーバー固有設定。

パス系のアセット（モデル資産・事前学習モデル・ユーザー辞書）は単一ソースを明確にするため
本モジュールでは再定義せず、同梱コアの `style_bert_vits2.constants`（環境変数
`SBV2_ASSETS_ROOT` / `SBV2_PRETRAINED_ROOT` / `SBV2_SKIP_PRETRAINED_DOWNLOAD` 対応）をそのまま正とする。
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

from style_bert_vits2.constants import ASSETS_ROOT, DEFAULT_BERT_MODEL_PATH

from .breath import DEFAULT_BREATH_SYMBOLS, DEFAULT_SENTENCE_SYMBOLS


class Settings(BaseSettings):
    """環境変数（または .env）から読むサーバー設定。大文字小文字は区別しない"""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    host: str = "127.0.0.1"
    port: int = 8000

    #: 推論デバイス。"cuda" でも CUDA が無い場合は起動時に "cpu" へフォールバックする
    device: str = "cuda"

    #: 合成モデルの重みの精度（"float16" / "float32"。コアの TTSModel と同じ契約）
    dtype: Literal["float16", "float32"] = "float16"

    #: GPU/メモリに常駐させる合成モデルの上限。超えた分は LRU でアンロード
    max_resident_models: int = 2

    #: カンマ区切りで起動時に先読みロードするモデル名（空なら初回リクエスト時に遅延ロード）
    preload_models: str = ""

    #: True にすると起動時に 1 文字だけ実際に合成して CUDA カーネル等を暖気する。
    #: BERT/pyopenjtalk の初期化自体は常に起動時に行われる（合成は不要）
    warmup_synth: bool = False

    #: 記号・空白位置に挿入する「一呼吸」の無音長（秒）。リクエストの
    #: `breath_interval` で上書き可（0 で無効化）
    breath_interval: float = 0.4

    #: 「。」等の文末シンボルの直後に挿入する無音長（秒）。一呼吸とは別枠。
    #: リクエストの `sentence_interval` で上書き可（0 で無効化）
    sentence_interval: float = 0.8

    #: 非文末級（一呼吸）の対象シンボル一覧。空白類（半角・全角・改行等）は
    #: ここに書かなくても常に対象。対象から外すものは空にする等の設計ではない
    #: （既定は breath.DEFAULT_BREATH_SYMBOLS 参照）
    breath_symbols: str = DEFAULT_BREATH_SYMBOLS

    #: 文末級の対象シンボル一覧（例: `SENTENCE_SYMBOLS=。？` なら「。」と「？」の
    #: 直後に sentence_interval 秒の無音）。同じシンボルを両方に指定したら文末級優先
    sentence_symbols: str = DEFAULT_SENTENCE_SYMBOLS

    #: 生成音声の末尾に付ける無音の秒数（連続生成時の詰まり防止）。
    #: リクエストの `tail_silence` で上書き可（0 で無効化）
    tail_silence: float = 0.3

    @property
    def preload_list(self) -> list[str]:
        return [name.strip() for name in self.preload_models.split(",") if name.strip()]

    @property
    def assets_root(self):  # noqa: ANN201 - Path を返す（型はコア側定数の型に合わせる）
        """合成モデル資産の走査先（コアの ASSETS_ROOT そのまま）"""
        return ASSETS_ROOT

    @property
    def bert_model_path(self):  # noqa: ANN201 - Path を返す
        """日本語 BERT（DeBERTa）のパス（コアの DEFAULT_BERT_MODEL_PATH そのまま）"""
        return DEFAULT_BERT_MODEL_PATH


@lru_cache
def get_settings() -> Settings:
    return Settings()
