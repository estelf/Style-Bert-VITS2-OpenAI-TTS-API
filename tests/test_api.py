"""API 挙動テスト（偽モデル・偽 BERT を使い、ネットワーク・GPU 不要）。

TestClient をコンテキストマネージャなしで使うと lifespan が走らないため、
`create_app()` 後に `registry.refresh()` だけ手動で呼んで各ルートを検証する
（BERT ロードや pyopenjtalk ワーカー起動を伴う重い初期化は test_gpu.py の実モデルテストでカバーする）。
"""

from __future__ import annotations

import shutil
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


# --- speech: voice 省略・空文字（1 モデル = 1 話者なので自動選択） ---


def test_speech_voice_omitted_ok(client: TestClient) -> None:
    # voice 省略は検証エラー（400）にも話者NotFound（404）にもならない（偽モデル合成のため 500 になり得る）
    payload = _payload()
    del payload["voice"]
    r = client.post("/v1/audio/speech", json=payload)
    assert r.status_code not in (400, 404)


@pytest.mark.parametrize("blank", ["", "   ", "  \t\n"])
def test_speech_voice_blank_treated_as_omitted(client: TestClient, blank: str) -> None:
    r = client.post("/v1/audio/speech", json=_payload(voice=blank))
    assert r.status_code not in (400, 404)


def test_speech_request_voice_normalization() -> None:
    from style_bert_vits2_openai_tts_api.schemas import SpeechRequest

    base = {"model": "m", "input": "あ"}
    assert SpeechRequest.model_validate(base).voice is None  # 省略
    assert SpeechRequest.model_validate({**base, "voice": ""}).voice is None
    assert SpeechRequest.model_validate({**base, "voice": "  "}).voice is None
    assert SpeechRequest.model_validate({**base, "voice": " ok "}).voice == "ok"


def test_speaker_id_auto_selects_model_speaker(tmp_path: Path) -> None:
    write_fake_model(tmp_path, "m1", voice="solo")
    reg = ModelRegistry(root=tmp_path, device="cpu")
    reg.refresh()
    entry = reg.get_entry("m1")
    assert reg.speaker_id(entry, None) == entry.spk2id["solo"]


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


def test_speech_request_breath_interval() -> None:
    from pydantic import ValidationError

    from style_bert_vits2_openai_tts_api.schemas import SpeechRequest

    base = {"model": "m", "input": "あ"}
    assert SpeechRequest.model_validate(base).breath_interval is None
    assert SpeechRequest.model_validate({**base, "breath_interval": 0.0}).breath_interval == 0.0
    assert SpeechRequest.model_validate({**base, "breath_interval": 0.3}).breath_interval == 0.3
    with pytest.raises(ValidationError):
        SpeechRequest.model_validate({**base, "breath_interval": -0.1})
    with pytest.raises(ValidationError):
        SpeechRequest.model_validate({**base, "breath_interval": 10.5})


def test_speech_breath_interval_out_of_range_400(client: TestClient) -> None:
    r = client.post("/v1/audio/speech", json=_payload(breath_interval=10.5))
    assert r.status_code == 400
    assert r.json()["error"]["param"] == "breath_interval"


def test_speech_breath_interval_ok(client: TestClient) -> None:
    # 偽モデル合成のため 500 になり得るが、breath_interval による検証エラー・NotFound にはならない
    r = client.post(
        "/v1/audio/speech",
        json=_payload(input="これは テスト です", breath_interval=0.3),
    )
    assert r.status_code not in (400, 404)


def test_speech_request_sentence_interval() -> None:
    from pydantic import ValidationError

    from style_bert_vits2_openai_tts_api.schemas import SpeechRequest

    base = {"model": "m", "input": "あ"}
    assert SpeechRequest.model_validate(base).sentence_interval is None
    assert (
        SpeechRequest.model_validate({**base, "sentence_interval": 0.0}).sentence_interval == 0.0
    )
    assert (
        SpeechRequest.model_validate({**base, "sentence_interval": 0.8}).sentence_interval == 0.8
    )
    with pytest.raises(ValidationError):
        SpeechRequest.model_validate({**base, "sentence_interval": -0.1})
    with pytest.raises(ValidationError):
        SpeechRequest.model_validate({**base, "sentence_interval": 10.5})


def test_speech_sentence_interval_out_of_range_400(client: TestClient) -> None:
    r = client.post("/v1/audio/speech", json=_payload(sentence_interval=10.5))
    assert r.status_code == 400
    assert r.json()["error"]["param"] == "sentence_interval"


def test_speech_sentence_interval_ok(client: TestClient) -> None:
    r = client.post(
        "/v1/audio/speech",
        json=_payload(input="天気です。外出します", sentence_interval=0.6),
    )
    assert r.status_code not in (400, 404)


def test_speech_request_tail_silence() -> None:
    from pydantic import ValidationError

    from style_bert_vits2_openai_tts_api.schemas import SpeechRequest

    base = {"model": "m", "input": "あ"}
    assert SpeechRequest.model_validate(base).tail_silence is None
    assert SpeechRequest.model_validate({**base, "tail_silence": 0.0}).tail_silence == 0.0
    assert SpeechRequest.model_validate({**base, "tail_silence": 0.5}).tail_silence == 0.5
    with pytest.raises(ValidationError):
        SpeechRequest.model_validate({**base, "tail_silence": -0.1})
    with pytest.raises(ValidationError):
        SpeechRequest.model_validate({**base, "tail_silence": 10.5})


def test_speech_tail_silence_out_of_range_400(client: TestClient) -> None:
    r = client.post("/v1/audio/speech", json=_payload(tail_silence=10.5))
    assert r.status_code == 400
    assert r.json()["error"]["param"] == "tail_silence"


def test_speech_tail_silence_ok(client: TestClient) -> None:
    r = client.post("/v1/audio/speech", json=_payload(tail_silence=0.5))
    assert r.status_code not in (400, 404)


# --- 記号グループ設定（BREATH_SYMBOLS / SENTENCE_SYMBOLS） ---


def test_symbol_settings_defaults() -> None:
    from style_bert_vits2_openai_tts_api.breath import (
        DEFAULT_BREATH_SYMBOLS,
        DEFAULT_SENTENCE_SYMBOLS,
    )
    from style_bert_vits2_openai_tts_api.settings import Settings

    # 実 .env を読まず、環境変数のみで解決させる
    s = Settings(_env_file=None)
    assert s.breath_symbols == DEFAULT_BREATH_SYMBOLS
    assert s.sentence_symbols == DEFAULT_SENTENCE_SYMBOLS


def test_symbol_settings_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    from style_bert_vits2_openai_tts_api.settings import Settings

    monkeypatch.setenv("SENTENCE_SYMBOLS", "。？")
    monkeypatch.setenv("BREATH_SYMBOLS", "・※")
    s = Settings(_env_file=None)
    assert s.sentence_symbols == "。？"
    assert s.breath_symbols == "・※"


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


# --- model_assets ホットリロード（話者・モデル追加に再起動不要） ---


def test_registry_refresh_no_change_returns_false(tmp_path: Path) -> None:
    write_fake_model(tmp_path, "m1")
    reg = ModelRegistry(root=tmp_path, device="cpu")
    assert reg.refresh() is True
    assert reg.refresh() is False  # 差分無し（config.json も読み直さない）


def test_registry_detects_added_model(tmp_path: Path) -> None:
    reg = ModelRegistry(root=tmp_path, device="cpu")
    reg.refresh()
    assert reg.model_names == []
    write_fake_model(tmp_path, "m2")
    assert reg.refresh() is True
    assert reg.model_names == ["m2"]
    assert reg.get_entry("m2").speakers == ["m2"]


def test_registry_detects_removed_model(tmp_path: Path) -> None:
    write_fake_model(tmp_path, "m1")
    reg = ModelRegistry(root=tmp_path, device="cpu")
    reg.refresh()
    shutil.rmtree(tmp_path / "m1")
    assert reg.refresh() is True
    assert reg.model_names == []


def test_registry_detects_changed_checkpoint(tmp_path: Path) -> None:
    model_dir = write_fake_model(tmp_path, "m1")
    reg = ModelRegistry(root=tmp_path, device="cpu")
    reg.refresh()
    sig = reg.get_entry("m1").signature
    (model_dir / "best.safetensors").write_bytes(b"retrained-checkpoint-bytes")
    assert reg.refresh() is True
    assert reg.get_entry("m1").signature != sig


class _FakeTTSModel:
    """TTSModel の検証用代替。load/unload を記録する（instance を生成順に captured に積む）"""

    instances: list["_FakeTTSModel"] = []

    def __init__(self, **kwargs: object) -> None:
        self.loaded = False
        self.unloaded = False
        _FakeTTSModel.instances.append(self)

    def load(self) -> None:
        self.loaded = True

    def unload(self) -> None:
        self.unloaded = True


@pytest.fixture()
def fake_tts_model(monkeypatch: pytest.MonkeyPatch) -> list[_FakeTTSModel]:
    import style_bert_vits2_openai_tts_api.model_manager as mm

    _FakeTTSModel.instances = []
    monkeypatch.setattr(mm, "TTSModel", _FakeTTSModel)
    return _FakeTTSModel.instances


def test_removed_model_defers_unload_until_next_synthesis(
    tmp_path: Path, fake_tts_model: list[_FakeTTSModel]
) -> None:
    write_fake_model(tmp_path, "m1")
    write_fake_model(tmp_path, "m2")
    reg = ModelRegistry(root=tmp_path, device="cpu", max_resident=5)
    reg.refresh()
    with reg.synthesis_lock:
        reg.ensure_loaded("m1")
    m1 = fake_tts_model[0]

    shutil.rmtree(tmp_path / "m1")
    reg.refresh()  # m1 は登録から外れるが、合成中かもしれないので即 unload はしない
    assert reg.model_names == ["m2"]
    assert m1.unloaded is False

    with reg.synthesis_lock:  # 次の合成（synthesis_lock 取得時）にまとめて解放される
        reg.ensure_loaded("m2")
    assert m1.unloaded is True
    assert reg.loaded_names() == ["m2"]


def test_changed_model_reloads_on_next_synthesis(
    tmp_path: Path, fake_tts_model: list[_FakeTTSModel]
) -> None:
    model_dir = write_fake_model(tmp_path, "m1")
    reg = ModelRegistry(root=tmp_path, device="cpu")
    reg.refresh()
    with reg.synthesis_lock:
        reg.ensure_loaded("m1")
    assert len(fake_tts_model) == 1

    # 無変化なら再ロードされない
    with reg.synthesis_lock:
        assert reg.ensure_loaded("m1") is fake_tts_model[0]
    assert len(fake_tts_model) == 1

    # 会話中にチェックポイントが更新された → 次回合成時に旧インスタンスを下ろして再ロード
    (model_dir / "best.safetensors").write_bytes(b"retrained-checkpoint-bytes")
    reg.refresh()
    with reg.synthesis_lock:
        reloaded = reg.ensure_loaded("m1")
    assert len(fake_tts_model) == 2
    assert fake_tts_model[0].unloaded is True
    assert reloaded is fake_tts_model[1]


def test_api_sees_added_model_without_restart(tmp_path: Path) -> None:
    """model_assets（走査先）へ後から追加したモデル・話者が、再起動無しで
    各 API と合成に反映されること"""
    app = create_app()
    app.state.registry.root = tmp_path  # 空の一時ディレクトリへ向けた状態で起動
    app.state.registry.refresh()
    client = TestClient(app)

    ids = [m["id"] for m in client.get("/v1/models").json()["data"]]
    assert "late" not in ids

    write_fake_model(tmp_path, "late", voice="late")

    ids = [m["id"] for m in client.get("/v1/models").json()["data"]]
    assert "late" in ids
    voices = client.get("/v1/audio/voices").json()["data"]
    assert {"id": "late", "object": "voice", "model": "late"} in voices
    # 合成リクエストからも見える（404 ではない。偽モデルなので 500 になり得る）
    r = client.post("/v1/audio/speech", json={"model": "late", "input": "あ"})
    assert r.status_code != 404


def test_api_speech_gone_model_404_without_restart(tmp_path: Path) -> None:
    app = create_app()
    write_fake_model(tmp_path, "gone")
    app.state.registry.root = tmp_path
    app.state.registry.refresh()
    client = TestClient(app)

    r = client.post("/v1/audio/speech", json={"model": "gone", "input": "あ"})
    assert r.status_code != 404

    shutil.rmtree(tmp_path / "gone")
    r = client.post("/v1/audio/speech", json={"model": "gone", "input": "あ"})
    assert r.status_code == 404
    assert r.json()["error"]["type"] == "not_found_error"


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
