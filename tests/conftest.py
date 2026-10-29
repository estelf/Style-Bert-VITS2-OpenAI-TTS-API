"""テスト共通設定。

**style_bert_vits2 の import より前に** 環境変数をセットし、テストの走査先を
git 管理のフィクスチャ（tests/fixtures/）へ向けることで、ネットワーク・実モデル無しで
API 挙動のテストが完結するようにする（フィクスチャの BERT ディレクトリは
`style_bert_vits2.pretrained` の完備判定がファイル存在チェックだけなのを利用した偽物で、
ensure_pretrained は通信せずに通る）。

実モデル・GPU を使う統合テスト（`-m gpu`）は、環境変数を明示的に渡したときだけ有効になる:

    SBV2_REAL_TEST=1 SBV2_ASSETS_ROOT=/path/to/model_assets uv run pytest -m gpu
"""

from __future__ import annotations

import json
import os
from pathlib import Path

TESTS_DIR = Path(__file__).parent
FIXTURES_DIR = TESTS_DIR / "fixtures"

# import style_bert_vits2.constants（環境変数を import 時にバインドする）より先にセットする。
# setdefault なので、GPU テスト実行時に外から渡した SBV2_* 環境変数は上書きされない
if os.environ.get("SBV2_REAL_TEST") == "1":
    # 実モデル統合テスト: 偽フィクスチャは使わない。BERT は本リポジトリ ./pretrained/ の
    # 実物が既定（無ければ起動時に自動ダウンロードされる流れ自体を検証する）
    if not os.environ.get("SBV2_ASSETS_ROOT"):
        raise RuntimeError(
            "SBV2_REAL_TEST=1 の実行には SBV2_ASSETS_ROOT に実モデル資産ルートの指定が必要"
        )
else:
    os.environ.setdefault("SBV2_ASSETS_ROOT", str(FIXTURES_DIR / "model_assets"))
    os.environ.setdefault("SBV2_PRETRAINED_ROOT", str(FIXTURES_DIR / "pretrained"))

_FAKE_MODEL = "fake"
_FAKE_VOICE = "fake"
_BERT_DIR_NAME = "deberta-v2-large-japanese-char-wwm"
# style_bert_vits2.pretrained の "bert" manifest と同一ファイル名（内容は中身を見ない偽物）
_BERT_FILES = (
    "config.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.txt",
    "model.safetensors",
)


def _build_fixtures() -> None:
    """偽モデル資産と偽 BERT を作る（既にあるものは触らない。ネットワーク不要）"""
    model_dir = FIXTURES_DIR / "model_assets" / _FAKE_MODEL
    model_dir.mkdir(parents=True, exist_ok=True)
    config_path = model_dir / "config.json"
    if not config_path.exists():
        config_path.write_text(
            json.dumps(
                {
                    "model_name": _FAKE_MODEL,
                    "version": "2.7.0-JP-Extra",
                    "data": {"spk2id": {_FAKE_VOICE: 0}, "num_styles": 1},
                }
            ),
            encoding="utf-8",
        )
    style_vec = model_dir / "style_vectors.npy"
    if not style_vec.exists():
        import numpy as np

        style_vec.write_bytes(np.zeros((1, 256), dtype=np.float32).tobytes())
    # TTSModel のロードまで行うテストはないので、チェックポイントファイルは存在だけあればよい
    for name in ("best.safetensors", "fake_e1_s10.safetensors"):
        f = model_dir / name
        if not f.exists():
            f.write_bytes(b"fake-checkpoint")

    bert_dir = FIXTURES_DIR / "pretrained" / "bert" / _BERT_DIR_NAME
    bert_dir.mkdir(parents=True, exist_ok=True)
    for name in _BERT_FILES:
        f = bert_dir / name
        if not f.exists():
            f.write_text('{"fake": true}\n', encoding="utf-8")


_build_fixtures()

import pytest  # noqa: E402

requires_real_model = pytest.mark.skipif(
    os.environ.get("SBV2_REAL_TEST") != "1",
    reason="実モデル統合テストは SBV2_REAL_TEST=1 SBV2_ASSETS_ROOT=... を渡したときだけ実行される",
)


def write_fake_model(root: Path, name: str, voice: str | None = None) -> Path:
    """テスト内で追加の偽モデルを作るヘルパー（config.json / style_vectors.npy / best.safetensors）"""
    import numpy as np

    voice = voice or name
    model_dir = root / name
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "config.json").write_text(
        json.dumps(
            {
                "model_name": name,
                "version": "2.7.0-JP-Extra",
                "data": {"spk2id": {voice: 0}, "num_styles": 1},
            }
        ),
        encoding="utf-8",
    )
    (model_dir / "style_vectors.npy").write_bytes(
        np.zeros((1, 256), dtype=np.float32).tobytes()
    )
    (model_dir / "best.safetensors").write_bytes(b"fake-checkpoint")
    return model_dir
