"""実モデル・GPU・実 BERT を使う統合テスト。

既定ではスキップ。実行には実モデル資産のルート明示と BERT 取得が必要:

    SBV2_REAL_TEST=1 SBV2_ASSETS_ROOT=/path/to/model_assets uv run pytest -m gpu

（BERT は既定では本リポジトリ ./pretrained/ に無くれば自動ダウンロードされる）
"""

from __future__ import annotations

import io
from collections.abc import Iterator

import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from style_bert_vits2_openai_tts_api.app import create_app

from .conftest import requires_real_model

pytestmark = [pytest.mark.gpu, requires_real_model]


@pytest.fixture(scope="module")
def real_client() -> Iterator[TestClient]:
    """lifespan 込みの完全初期化（BERT 取得検証→ロード→G2P→モデル走査）"""
    app = create_app()
    with TestClient(app) as client:
        yield client


def _pick_model_voice(client: TestClient) -> tuple[str, str]:
    models = client.get("/v1/models").json()["data"]
    assert models, "SBV2_ASSETS_ROOT 配下に有効なモデルが 1 つ以上必要"
    model = models[0]["id"]
    voices = client.get("/v1/audio/voices").json()["data"]
    voice = next(v["id"] for v in voices if v["model"] == model)
    return model, voice


def _speech(client: TestClient, model: str, voice: str, **over) -> tuple[bytes, float]:
    r = client.post(
        "/v1/audio/speech",
        json={"model": model, "input": "こんにちは、はじめまして。", "voice": voice, **over},
    )
    assert r.status_code == 200, r.text
    assert r.headers["content-type"] == "audio/flac"
    audio, sr = sf.read(io.BytesIO(r.content))
    return r.content, len(audio) / sr


def test_synth_returns_flac(real_client: TestClient) -> None:
    model, voice = _pick_model_voice(real_client)
    flac, duration = _speech(real_client, model, voice)
    assert flac[:4] == b"fLaC"
    assert duration > 0.3


def test_synth_without_voice_uses_model_speaker(real_client: TestClient) -> None:
    """voice 省略・空文字でもモデルの単一話者で合成できる（1 モデル = 1 話者）"""
    model, _ = _pick_model_voice(real_client)
    for voice in (None, ""):
        body = {"model": model, "input": "ぼいすをぬいてもよめます。"}
        if voice is not None:
            body["voice"] = voice
        r = real_client.post("/v1/audio/speech", json=body)
        assert r.status_code == 200, r.text
        assert r.headers["content-type"] == "audio/flac"
        audio, sr = sf.read(io.BytesIO(r.content))
        assert len(audio) / sr > 0.3


def test_speed_scales_duration(real_client: TestClient) -> None:
    """speed=2.0（length=0.5）は speed=1.0 より明らかに短い"""
    model, voice = _pick_model_voice(real_client)
    _, dur_1x = _speech(real_client, model, voice, speed=1.0)
    _, dur_2x = _speech(real_client, model, voice, speed=2.0)
    assert dur_2x < dur_1x * 0.85


def test_second_request_is_ready(real_client: TestClient) -> None:
    """同一モデルの 2 回目合成も同じく 200/flac（遅延ロード後の常駐状態が壊れていないこと）"""
    model, voice = _pick_model_voice(real_client)
    _speech(real_client, model, voice)
    _speech(real_client, model, voice)
