"""合成音声（int16 PCM）から API レスポンス用の音声バイナリへのエンコード。

`TTSModel.infer()` は (サンプリングレート, 16bit PCM NDArray) を返す。
本サーバーの response_format は flac のみ対応のため、soundfile（libsndfile）で
メモリ上に FLAC エンコードする（ffmpeg 等の外部プロセスは使わない）。
"""

from __future__ import annotations

import io
from typing import Any

import soundfile as sf
from numpy.typing import NDArray

#: 対応する response_format と HTTP レスポンスの Content-Type
MEDIA_TYPES: dict[str, str] = {"flac": "audio/flac"}


def encode_flac(sample_rate: int, audio: NDArray[Any]) -> bytes:
    """16bit PCM を FLAC バイト列へ変換する（可逆・サンプリングレートはそのまま）"""
    buf = io.BytesIO()
    sf.write(buf, audio, sample_rate, format="FLAC", subtype="PCM_16")
    return buf.getvalue()
