"""Style-Bert-VITS2 ベースの OpenAI TTS API 互換音声合成サーバー。

起動: `style-bert-vits2-openai-tts-api`（または `uv run uvicorn style_bert_vits2_openai_tts_api.app:app`）
設定は環境変数（HOST / PORT / DEVICE / DTYPE / MAX_RESIDENT_MODELS / PRELOAD_MODELS / WARMUP_SYNTH、
アセット系は SBV2_ASSETS_ROOT / SBV2_PRETRAINED_ROOT / SBV2_SKIP_PRETRAINED_DOWNLOAD）。
"""


def main() -> None:
    import uvicorn

    from .settings import get_settings

    settings = get_settings()
    # 単一プロセス想定（合成はロックで直列化されるため workers で throughput は上がらない）
    uvicorn.run(
        "style_bert_vits2_openai_tts_api.app:app",
        host=settings.host,
        port=settings.port,
    )
