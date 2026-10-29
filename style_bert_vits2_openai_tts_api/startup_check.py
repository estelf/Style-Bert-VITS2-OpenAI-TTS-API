"""起動時の必須アセット検証。

- 日本語 BERT（DeBERTa）: 未取得ならコアの `ensure_pretrained` が Hugging Face から取得する。
  サーバーは **HTTP を bind する前（＝初回リクエストの前に）** 明示的にここを通すことで、
  合成リクエストの最中に 1.2GB のダウンロードが走り出す状態を防ぐ
  （未取得が続く・オフライン・`SBV2_SKIP_PRETRAINED_DOWNLOAD=1` ならコアが `PretrainedModelMissing`
  で明確に止まり、そのメッセージをそのまま起動失敗として返す）
- `model_assets/`: 有効な合成モデルが 1 つ以上あること（走査自体は ModelRegistry が担う）。
  ここはユーザーの学習成果物でありダウンロード対象ではないため、欠けていれば配置手順を表示して止まる
"""

from __future__ import annotations

from pathlib import Path

from style_bert_vits2.logging import logger
from style_bert_vits2.pretrained import PretrainedModelMissing, ensure_pretrained

from .settings import Settings

ASSETS_SETUP_HELP = (
    "model_assets/ に有効な合成モデルが見つかりません。学習済みモデルを "
    "<assets_root>/<モデル名>/ 直下に次の構成で配置してください（例: model_assets/my-model/ ...）。 "
    "配置先は環境変数 SBV2_ASSETS_ROOT で変更できます。\n"
    "  <モデル名>/\n"
    "    config.json\n"
    "    <任意の名前>.safetensors（best.safetensors があれば優先）\n"
    "    style_vectors.npy"
)


class StartupError(RuntimeError):
    """起動に必要なアセットが揃っておらずサーバーを開始できない"""


def ensure_bert_model(settings: Settings) -> None:
    """BERT が揃っていれば即 return、無ければ取得する（詳細はモジュール docstring 参照）"""
    try:
        ensure_pretrained(("bert",))
    except PretrainedModelMissing as e:
        raise StartupError(str(e)) from e
    logger.info(f"JP BERT is ready at {settings.bert_model_path}")


def validate_assets_root(settings: Settings) -> Path:
    """モデル資産ルートが存在し、後続の走査を受けられることを確認する（中身の有無は Registry が見る）"""
    root = settings.assets_root
    if not root.exists():
        raise StartupError(
            f"モデル資産ルート {root} が存在しません。\n{ASSETS_SETUP_HELP.format(assets_root=root)}"
        )
    return root
