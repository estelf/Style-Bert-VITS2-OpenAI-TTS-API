"""`POST /v1/audio/speech` のリクエストスキーマと OpenAI 形式エラーヘルパー。

OpenAI の TTS API 互換を意識した最小構成。本仕様に無いキー（`stream` など）は
Pydantic の extra="ignore" で黙って無視し、既存 OpenAI クライアントとの互換を保つ。
スタイル系（style / style_weight / instructions）は本サーバーでは非対応（Neutral 固定）。
"""

from __future__ import annotations

from typing import Any, Literal

from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator

#: OpenAI 仕様と同じ受付範囲
SPEED_MIN = 0.25
SPEED_MAX = 4.0


class SpeechRequest(BaseModel):
    """/v1/audio/speech のリクエストボディ。

    標準フィールドに加え、コア TTSModel.infer の数値系パラメータを任意の拡張フィールドとして受け付ける。
    未指定（None）のコアパラメータはコア側の既定値にフォールバックする。
    """

    model_config = ConfigDict(extra="ignore")

    # --- OpenAI TTS 標準フィールド ---
    model: str = Field(..., description="model_assets/ のモデル名")
    input: str = Field(..., description="合成テキスト")
    voice: str = Field(..., description="話者名")
    response_format: Literal["flac"] = Field(
        default="flac", description="本サーバーは flac のみ対応"
    )
    speed: float = Field(
        default=1.0, ge=SPEED_MIN, le=SPEED_MAX, description="話速（大きいほど早い）"
    )

    # --- 拡張フィールド（任意。None でコア既定値） ---
    sdp_ratio: float | None = None
    noise: float | None = None
    noise_w: float | None = None
    pitch_scale: float | None = None
    intonation_scale: float | None = None
    line_split: bool | None = None
    split_interval: float | None = None

    @field_validator("input")
    @classmethod
    def _input_not_empty(cls, v: str) -> str:
        if v.strip() == "":
            raise ValueError("input must not be empty")
        return v

    @field_validator("model", "voice")
    @classmethod
    def _not_blank(cls, v: str) -> str:
        if v.strip() == "":
            raise ValueError("must not be empty")
        return v


def openai_error(
    status_code: int,
    message: str,
    *,
    err_type: str = "invalid_request_error",
    param: str | None = None,
    code: str | None = None,
) -> JSONResponse:
    """OpenAI 形式のエラー JSON を返す。

    形式: {"error": {"message": ..., "type": ..., "param": ..., "code": ...}}
    """
    body: dict[str, Any] = {
        "error": {
            "message": message,
            "type": err_type,
            "param": param,
            "code": code,
        }
    }
    return JSONResponse(status_code=status_code, content=body)
