"""FastAPI アプリケーション（OpenAI TTS API 互換サーバー本体）。

起動時（lifespan、HTTP bind の前）:
  1. BERT 取得検証（style_bert_vits2.pretrained。未取得時のみダウンロード。総量ログ付き）
  2. pyopenjtalk ワーカー起動
  3. BERT モデル / トークナイザー明示ロード（コア公式の推奨手順）
  4. 1 文字の G2P ウォームアップ（mecab / onnx 読み込み。合成は行わない）
  5. PRELOAD_MODELS 指定モデルの先読みロード（任意）と WARMUP_SYNTH（任意）
モデルの遅延ロード・LRU 退避と合成は synthesis_lock で直列化される。
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from pydantic import ValidationError
from starlette.concurrency import run_in_threadpool
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.responses import JSONResponse, Response

from style_bert_vits2.logging import logger
from style_bert_vits2.nlp import clean_text
from style_bert_vits2.nlp import bert_models
from style_bert_vits2.nlp.japanese import pyopenjtalk_worker as pyopenjtalk

from .audio import MEDIA_TYPES, encode_flac
from .model_manager import (
    ModelNotFound,
    ModelRegistry,
    VoiceNotFound,
)
from .schemas import SPEED_MAX, SPEED_MIN, SpeechRequest, openai_error
from .settings import Settings, get_settings
from .startup_check import StartupError, ensure_bert_model, validate_assets_root

_NOT_FOUND_TYPES = {404: "not_found_error"}


def _start_pyopenjtalk_worker() -> None:
    """pyopenjtalk ワーカーを起動する。

    注意: コアの `initialize_worker()` は SIGTERM ハンドラを「ワーカーの後始末だけ」に
    上書きしてしまい、uvicorn がインストール済みのグレースフルシャットダウンハンドラが
    握りつぶされる（SIGTERM で uvicorn の shutdown 処理が走らず、プロセスが落ちない）。
    ワーカーの後始末は lifespan の shutdown（terminate_worker）で必ず行うため、
    起動後に SIGTERM ハンドラを復元する。
    """
    import signal

    previous_sigterm = signal.getsignal(signal.SIGTERM)
    pyopenjtalk.initialize_worker()
    try:
        signal.signal(signal.SIGTERM, previous_sigterm)
    except ValueError:
        # メインスレッド以外（TestClient の一部実行環境など）では設定できないためそのまま
        pass


def _resolve_device(requested: str) -> str:
    """CUDA 非搭載環境で "cuda" を "cpu" にフォールバックする（torch はここで初めて import する）"""
    import torch

    if requested == "cuda" and not torch.cuda.is_available():
        logger.warning('Requested device "cuda" but CUDA is unavailable. Falling back to "cpu".')
        return "cpu"
    return requested


def _warmup_g2p() -> None:
    """pyopenjtalk ワーカー内部的な mecab / onnx モデル読み込みを起動時に済ませる（合成はしない）"""
    start = time.time()
    clean_text("あ")
    logger.info(f"G2P warmup finished ({time.time() - start:.2f}s)")


def _warmup_synth(registry: ModelRegistry) -> None:
    """1 文字を 1 回だけ実合成し、CUDA 遅延カーネル初期化等の初回オーバーヘッドを吸収する。
    既定では走らせない（WARMUP_SYNTH=1 のときのみ）。"""
    name = registry.model_names[0]
    entry = registry.get_entry(name)
    with registry.synthesis_lock:
        model = registry.ensure_loaded(name)
        start = time.time()
        model.infer("あ", speaker_id=next(iter(entry.spk2id.values())), line_split=False)
    logger.info(f"Synthesis warmup on model `{name}` finished ({time.time() - start:.2f}s)")


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    registry = ModelRegistry(
        root=settings.assets_root,
        device=settings.device,
        dtype=settings.dtype,
        max_resident=settings.max_resident_models,
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        try:
            # 1. アセット検証（BERT はここでのみ DL される。リクエスト中に DL を走らせない）
            ensure_bert_model(settings)
            validate_assets_root(settings)

            # 2. pyopenjtalk ワーカー起動（TCP ソケットサーバー。SIGTERM ハンドラ保護のためラッパー経由）
            _start_pyopenjtalk_worker()

            # 3. BERT / トークナイザーを明示ロード（パスを明示するのでコア側の暗黙取得は走らない）
            registry.device = _resolve_device(settings.device)
            bert_models.load_model(str(settings.bert_model_path), device_map=registry.device)
            bert_models.load_tokenizer(str(settings.bert_model_path))

            # 4. G2P ウォームアップ（合成なし）
            await run_in_threadpool(_warmup_g2p)

            # 5. モデル登録と任意の先読みロード
            registry.refresh()
            if not registry.entries:
                raise StartupError(
                    f"No valid model found under {registry.root}.\n"
                    "学習済みモデルを <assets_root>/<モデル名>/ に配置してください "
                    "（config.json / *.safetensors / style_vectors.npy。詳細は README）"
                )
            for name in settings.preload_list:
                logger.info(f"Preloading model `{name}` ...")
                with registry.synthesis_lock:
                    registry.ensure_loaded(name)
            if settings.warmup_synth:
                await run_in_threadpool(_warmup_synth, registry)

            logger.info(
                f"Server ready. device={registry.device} dtype={registry.dtype} "
                f"models={registry.model_names}"
            )
        except Exception:
            # 起動途中で失敗した場合、既に起動したワーカー等を残さない
            try:
                pyopenjtalk.terminate_worker()
            except Exception:
                pass
            raise

        yield

        registry.unload_all()
        try:
            pyopenjtalk.terminate_worker()
        except Exception as e:
            logger.warning(f"Failed to terminate pyopenjtalk worker: {e}")

    app = FastAPI(
        title="Style-Bert-VITS2 OpenAI TTS API",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.state.settings = settings
    app.state.registry = registry

    # --- エラーハンドラ（全エラーを OpenAI 形式 JSON に揃える） ---

    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(
        request: Request, exc: StarletteHTTPException
    ) -> JSONResponse:
        err_type = _NOT_FOUND_TYPES.get(exc.status_code, "invalid_request_error")
        return openai_error(
            exc.status_code, str(exc.detail), err_type=err_type, code=err_type
        )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        first = exc.errors()[0] if exc.errors() else {}
        param = ".".join(str(loc) for loc in first.get("loc", ())[1:]) or None
        return openai_error(
            400, str(first.get("msg", "Invalid request")), param=param
        )

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        logger.exception(f"Unhandled error on {request.method} {request.url.path}: {exc}")
        return openai_error(
            500, "Internal server error", err_type="server_error", code="internal_error"
        )

    # --- ルート ---

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "device": registry.device,
            "dtype": registry.dtype,
            "models": {
                "total": len(registry.entries),
                "resident": registry.loaded_names(),
            },
            "bert_loaded": bert_models.is_model_loaded(),
        }

    @app.get("/v1/models")
    def list_models() -> dict[str, Any]:
        data = [
            {
                "id": name,
                "object": "model",
                "created": int(entry.dir.stat().st_mtime),
                "owned_by": "style-bert-vits2",
            }
            for name, entry in registry.entries.items()
        ]
        return {"object": "list", "data": data}

    @app.get("/v1/audio/voices")
    def list_voices() -> dict[str, Any]:
        data = [
            {"id": voice, "object": "voice", "model": name}
            for name, entry in registry.entries.items()
            for voice in entry.speakers
        ]
        return {"object": "list", "data": data}

    @app.post("/v1/audio/speech")
    async def create_speech(request: Request) -> Response:
        try:
            payload = await request.json()
            if not isinstance(payload, dict):
                raise ValueError("body must be a JSON object")
        except Exception:
            return openai_error(
                400, "Request body must be a JSON object", param=None
            )

        try:
            req = SpeechRequest.model_validate(payload)
        except ValidationError as e:
            first = e.errors()[0] if e.errors() else {}
            param = ".".join(str(loc) for loc in first.get("loc", ())) or None
            return openai_error(400, str(first.get("msg", "Invalid request")), param=param)

        if req.response_format not in MEDIA_TYPES:
            return openai_error(
                400,
                f"Unsupported response_format `{req.response_format}`. "
                f"This server only supports: {', '.join(MEDIA_TYPES)}.",
                param="response_format",
            )
        if not SPEED_MIN <= req.speed <= SPEED_MAX:
            return openai_error(
                400,
                f"speed must be between {SPEED_MIN} and {SPEED_MAX}",
                param="speed",
            )

        try:
            entry = registry.get_entry(req.model)
            speaker_id = registry.speaker_id(entry, req.voice)
        except (ModelNotFound, VoiceNotFound) as e:
            available = e.available if isinstance(e, ModelNotFound) else e.voices
            return openai_error(
                404, f"{e}. Available: {', '.join(available)}", err_type="not_found_error"
            )

        # OpenAI の speed は「話速」（大きいほど早い）。コアの length は「長さ」なので逆数で渡す
        infer_kwargs: dict[str, Any] = {
            "text": req.input,
            "speaker_id": speaker_id,
            "length": 1.0 / req.speed,
            # 改行分割はサーバー側で既定オフ（OpenAI の 1 リクエスト＝1 音声に合わせる）
            "line_split": req.line_split if req.line_split is not None else False,
        }
        for key in ("sdp_ratio", "noise", "noise_w", "pitch_scale", "intonation_scale", "split_interval"):
            value = getattr(req, key)
            if value is not None:
                infer_kwargs[key] = value

        def synthesize() -> tuple[int, Any]:
            with registry.synthesis_lock:
                model = registry.ensure_loaded(req.model)
                return model.infer(**infer_kwargs)

        try:
            sample_rate, audio = await run_in_threadpool(synthesize)
            flac_bytes = await run_in_threadpool(encode_flac, sample_rate, audio)
        except Exception as e:
            logger.exception(f"Synthesis failed (model={req.model}, voice={req.voice}): {e}")
            return openai_error(
                500,
                "Voice synthesis failed. Check server logs for details.",
                err_type="server_error",
                code="synthesis_failed",
            )

        return Response(
            content=flac_bytes,
            media_type=MEDIA_TYPES[req.response_format],
            headers={"Cache-Control": "no-store"},
        )

    return app


#: `uvicorn style_bert_vits2_openai_tts_api.app:app` 用（環境変数は lifespan / Settings 側で読む）
app = create_app()
