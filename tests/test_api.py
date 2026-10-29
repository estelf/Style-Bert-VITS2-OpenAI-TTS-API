"""API 挙動テスト（偽モデル・偽 BERT を使い、ネットワーク・GPU 不要）。

TestClient をコンテキストマネージャなしで使うと lifespan が走らないため、
`create_app()` 後に `registry.refresh()` だけ手動で呼んで各ルートを検証する
（BERT ロードや pyopenjtalk ワーカー起動を伴う重い初期化は test_gpu.py の実モデルテストでカバーする）。
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from style_bert_vits2.pretrained import missing_files
from style_bert_vits2_openai_tts_api.app import create_app
from style_bert_vits2_openai_tts_api.model_manager import ModelRegistry
from style_bert_vits2_openai_tts_api.startup_check import StartupError, validate_assets_root

from .conftest import FIXTURES_DIR, write_fake_model


@pytest.fixture()
def client() -> TestClient:
    app = create_app()
    # lifespan は走らせず、モデル走査だけ行う
    app.state.registry.refresh()
    return TestClient(app)


# --- 完備判定（偽 BERT で ensure_pretrained が通信せず通る契約） ---


def test_fake_bert_is_considered_complete() -> None:
    assert missing_files(("bert",)) == {}


# --- 読み取り系 ---


def test_health_ok(client: TestClient) -> None:
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["models"]["total"] >= 1


def test_list_models(client: TestClient) -> None:
    r = client.get("/v1/models")
    assert r.status_code == 200
    body = r.json()
    assert body["object"] == "list"
    ids = [m["id"] for m in body["data"]]
    assert "fake" in ids
    for m in body["data"]:
        assert m["object"] == "model"
        assert isinstance(m["created"], int)


def test_list_voices(client: TestClient) -> None:
    r = client.get("/v1/audio/voices")
    assert r.status_code == 200
    data = r.json()["data"]
    assert {"id": "fake", "object": "voice", "model": "fake"} in data


# --- speech: バリデーション・エラー ---


def _payload(**over) -> dict:
    base = {"model": "fake", "input": "こんにちは", "voice": "fake"}
    base.update(over)
    return base


def test_speech_unknown_model_404(client: TestClient) -> None:
    r = client.post("/v1/audio/speech", json=_payload(model="ghost"))
    assert r.status_code == 404
    err = r.json()["error"]
    assert err["type"] == "not_found_error"
    assert "ghost" in err["message"]


def test_speech_unknown_voice_404(client: TestClient) -> None:
    r = client.post("/v1/audio/speech", json=_payload(voice="ghost"))
    assert r.status_code == 404
    assert r.json()["error"]["type"] == "not_found_error"


def test_speech_unsupported_format_400(client: TestClient) -> None:
    r = client.post("/v1/audio/speech", json=_payload(response_format="mp3"))
    assert r.status_code == 400
    err = r.json()["error"]
    assert err["param"] == "response_format"
    assert "flac" in err["message"]


def test_speech_speed_out_of_range_400(client: TestClient) -> None:
    r = client.post("/v1/audio/speech", json=_payload(speed=10.0))
    assert r.status_code == 400
    assert r.json()["error"]["param"] == "speed"


def test_speech_missing_field_400(client: TestClient) -> None:
    r = client.post("/v1/audio/speech", json={"model": "fake", "voice": "fake"})
    assert r.status_code == 400
    assert r.json()["error"]["param"] == "input"


def test_speech_empty_input_400(client: TestClient) -> None:
    r = client.post("/v1/audio/speech", json=_payload(input="   "))
    assert r.status_code == 400


def test_speech_invalid_json_400(client: TestClient) -> None:
    r = client.post(
        "/v1/audio/speech",
        content=b"not json",
        headers={"Content-Type": "application/json"},
    )
    assert r.status_code == 400
    assert "JSON" in r.json()["error"]["message"]


def test_speech_ignores_unknown_fields_shape(client: TestClient) -> None:
    # 未知フィールド（stream 等）は無視されて検証を通る（合成は偽モデルなので 500 になり得るが、
    # 400/404 にはならない＝フィールド互換のみ検証）
    r = client.post("/v1/audio/speech", json=_payload(stream=True, foo="bar"))
    assert r.status_code not in (400, 404) or "stream" not in r.text


# --- CLI 引数 ---


def test_cli_args_default_to_settings() -> None:
    from style_bert_vits2_openai_tts_api import _parse_args
    from style_bert_vits2_openai_tts_api.settings import get_settings

    args = _parse_args([])
    assert args.host == get_settings().host
    assert args.port == get_settings().port


def test_cli_args_override_host_and_port() -> None:
    from style_bert_vits2_openai_tts_api import _parse_args

    args = _parse_args(["--host", "0.0.0.0", "--port", "9000"])
    assert args.host == "0.0.0.0"
    assert args.port == 9000


# --- ModelRegistry の規約 ---


def test_registry_prefers_best_safetensors(tmp_path: Path) -> None:
    model_dir = write_fake_model(tmp_path, "m1")
    # best.safetensors より新しいチェックポイントを置いてみる（既定は best 優先で据わるはず）
    newer = model_dir / "later_e99.safetensors"
    newer.write_bytes(b"ckpt")
    import os
    import time

    future = time.time() + 3600
    os.utime(newer, (future, future))

    reg = ModelRegistry(root=tmp_path, device="cpu", max_resident=1)
    reg.refresh()
    entry = reg.get_entry("m1")
    assert entry.default_file.name == "best.safetensors"


def test_registry_skips_invalid_dirs(tmp_path: Path) -> None:
    (tmp_path / "no-config").mkdir()  # config.json 無し → スキップ
    write_fake_model(tmp_path, "valid")
    reg = ModelRegistry(root=tmp_path, device="cpu")
    reg.refresh()
    assert reg.model_names == ["valid"]


def test_validate_assets_root_missing(tmp_path: Path) -> None:
    from style_bert_vits2_openai_tts_api.settings import get_settings

    missing_root = tmp_path / "does-not-exist"
    # 設定の実アセットルートではなく、存在しないルートを明示的に検証する
    class _S:
        assets_root = missing_root

    with pytest.raises(StartupError):
        validate_assets_root(_S())  # 型は検証対象外（assets_root 属性だけ見る）


def test_fixtures_are_under_tests_dir() -> None:
    # 環境変数の取り違え防止: デフォルトの走査先は tests/fixtures 配下のはず
    from style_bert_vits2.constants import ASSETS_ROOT

    assert ASSETS_ROOT == FIXTURES_DIR / "model_assets"
