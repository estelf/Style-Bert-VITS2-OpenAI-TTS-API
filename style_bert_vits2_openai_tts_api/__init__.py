"""Style-Bert-VITS2 ベースの OpenAI TTS API 互換音声合成サーバー。

起動: `style-bert-vits2-openai-tts-api [--host HOST] [--port PORT]`
（`uv run uvicorn style_bert_vits2_openai_tts_api.app:app --host 0.0.0.0` でも可）
設定は環境変数（HOST / PORT / DEVICE / DTYPE / MAX_RESIDENT_MODELS / PRELOAD_MODELS / WARMUP_SYNTH、
アセット系は SBV2_ASSETS_ROOT / SBV2_PRETRAINED_ROOT / SBV2_SKIP_PRETRAINED_DOWNLOAD）。
バインド先は CLI 引数 > 環境変数 > 既定 の順で解決する。
"""

from __future__ import annotations

import argparse


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    from .settings import get_settings

    settings = get_settings()
    parser = argparse.ArgumentParser(
        prog="style-bert-vits2-openai-tts-api",
        description="Style-Bert-VITS2 ベースの OpenAI TTS API 互換音声合成サーバー",
    )
    parser.add_argument(
        "--host",
        default=settings.host,
        help=f"バインド先ホスト（既定: 環境変数 HOST、未設定なら {settings.host}。"
        "全インターフェースなら 0.0.0.0）",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=settings.port,
        help=f"バインド先ポート（既定: 環境変数 PORT、未設定なら {settings.port}）",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    import uvicorn

    args = _parse_args(argv)
    # 単一プロセス想定（合成はロックで直列化されるため workers で throughput は上がらない）
    uvicorn.run(
        "style_bert_vits2_openai_tts_api.app:app",
        host=args.host,
        port=args.port,
    )
