# Style-Bert-VITS2 OpenAI TTS API

[Style-Bert-VITS2](https://github.com/estelf/Style-Bert-VITS2) のコアライブラリを本リポジトリに同梱（vendor）した、
**OpenAI TTS API 互換**の日本語音声合成サーバーです。OpenAI SDK や既存の互換クライアントから
`base_url` の差し替えだけで呼び出せます。

## 特徴

- **自己完結**: コア（`style_bert_vits2/`）・サーバー（`style_bert_vits2_openai_tts_api/`）を 1 プロジェクトに同梱。
  他リポジトリへのパス依存は一切なし（`VENDORED.md` 参照）
- **事前学習モデルは起動時に自動取得**: 日本語 BERT（DeBERTa）は未配置だと初回起動時に
  Hugging Face から自動ダウンロードされます（コアの `style_bert_vits2.pretrained` によるもの。
  ダウンロードは HTTP bind 前に完結し、リクエスト中に 1.2GB が走り出すことはありません）
- **認証なし / 一括レスポンス（flac のみ）** — ローカル〜LAN 利用向けの構成
- **スタイル指定なし**: 同梱環境のモデルは単話者・Neutral のみのため、`style` 等は持たせず
  内部で Neutral 固定。話者（voice）とOpenAI標準パラメータに集中した薄い API です

## セットアップ

```bash
uv sync          # 依存インストール（torch のため初回は容量が多め）
```

### 1. 学習済みモデルの配置（`model_assets/`）

`model_assets/` ディレクトリ自体はリポジトリに同梱されていますが、中身は git 管理外です。
学習済みモデルを次の構成で配置してください
（他の環境で学習したモデルは `cp -r` で持ってきてください。`cp` ではなく `ln -s` でも動作します）:

```
model_assets/
└── my-model/
    ├── config.json
    ├── best.safetensors        ← あれば優先。無ければ更新日時最新の .safetensors
    └── style_vectors.npy
```

- 走査先の変更: 環境変数 `SBV2_ASSETS_ROOT`
- モデルが 1 つも無い場合は**具体的な配置手順を表示して起動に失敗**します（空サーバーで起動しません）

### 2. 事前学習済み BERT（自動）

`pretrained/`（git 管理外）に未配置なら、起動時に `ku-nlp/deberta-v2-large-japanese-char-wwm`
（約 1.24GB）を自動ダウンロードします。手動で揃える／確認する場合:

```bash
uv run -m style_bert_vits2.pretrained --only bert --dry-run   # 取得予定の確認
uv run -m style_bert_vits2.pretrained --only bert             # 手動取得
uv run -m style_bert_vits2.pretrained --check                 # 完備チェック
```

- 置き場の変更: 環境変数 `SBV2_PRETRAINED_ROOT`
- オフライン／ミラー環境: `SBV2_SKIP_PRETRAINED_DOWNLOAD=1`（未取得だと明確なエラーで停止。黙って通信しません）

## 起動

```bash
uv run style-bert-vits2-openai-tts-api
# → http://127.0.0.1:8000  (OpenAI 互換: POST /v1/audio/speech)

# 全インターフェースで待ち受け（LAN 公開など）
uv run style-bert-vits2-openai-tts-api --host 0.0.0.0 --port 8000
```

- バインド先は **CLI 引数 `--host` / `--port` > 環境変数 `HOST` / `PORT` > 既定（127.0.0.1 / 8000）** の順で解決します
- 認証なしで動くため、`--host 0.0.0.0` は **信頼できるネットワーク内での利用を前提**にしてください。到達できる相手は誰でも合成できます
- 初回起動は BERT の取得＋ロードで数分かかります。2 回目以降は BERT / G2P / 指定モデルの
ロードまで**起動時に完了**します（合成のダミー実行は不要。CUDA 初回カーネル初期化の
数秒まで吸収したいときだけ `WARMUP_SYNTH=1` をオプションで有効化）。

### 設定（環境変数 / .env）

| 変数 | 既定 | 説明 |
| --- | --- | --- |
| `HOST` / `PORT` | `127.0.0.1` / `8000` | bind 先（`--host` / `--port` があればそちらが優先） |
| `DEVICE` | `cuda` | 推論デバイス（CUDA 無ければ cpu へ自動フォールバック） |
| `DTYPE` | `float16` | 合成モデルの精度（`float16` / `float32`） |
| `MAX_RESIDENT_MODELS` | `2` | 常駐モデル上限（超過は LRU でアンロード） |
| `PRELOAD_MODELS` | （空） | 起動時に先読みロードするモデル名（カンマ区切り） |
| `WARMUP_SYNTH` | `0` | 起動時に 1 文字だけ実合成して暖気 |
| `SBV2_ASSETS_ROOT` | `./model_assets` | モデル資産ルート（コア共通） |
| `SBV2_PRETRAINED_ROOT` | `./pretrained` | 事前学習モデル置き場（コア共通） |
| `SBV2_SKIP_PRETRAINED_DOWNLOAD` | （オフ） | 自動ダウンロード抑止（コア共通） |

## API

### `POST /v1/audio/speech`

リクエスト（JSON）:

```json
{
  "model": "my-model",
  "input": "こんにちは、はじめまして。",
  "voice": "my-model",
  "response_format": "flac",
  "speed": 1.0
}
```

- `model`: `model_assets/` のモデル名
- `voice`: 話者名（config.json の spk2id。現状のモデルは単話者でモデル名と同名）
- `response_format`: **`flac` のみ**（省略時既定）。mp3 等は 400 を返します
- `speed`: 0.25〜4.0（OpenAI と同じく大きいほど早い。内部では length=1/speed）
- 拡張フィールド（任意・省略可）: `sdp_ratio` / `noise` / `noise_w` / `pitch_scale` /
  `intonation_scale` / `line_split`（既定 false）/ `split_interval`
- `stream` などの未知フィールドは無視されます。スタイル（`style` / `instructions`）は非対応
- 成功時 `audio/flac` のバイナリ、エラーは OpenAI 形式の JSON（`{"error": {...}}`）

curl:

```bash
curl -s http://127.0.0.1:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"model":"my-model","input":"こんにちは、はじめまして。","voice":"my-model"}' \
  --output speech.flac
```

OpenAI SDK:

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="unused")  # 認証なしなので api_key は無視されます
with client.audio.speech.with_streaming_response.create(
    model="my-model",
    voice="my-model",
    input="こんにちは、はじめまして。",
    response_format="flac",
) as response:
    response.save("speech.flac")
```

### その他ルート

- `GET /v1/models` — 配置済みモデル一覧（`{"object":"list","data":[...]}`）
- `GET /v1/audio/voices` — 話者一覧（非標準。モデル名との対応付き）
- `GET /health` — 生体確認（常駐モデル・BERT ロード状態つき）

### インタラクティブドキュメント（Swagger UI / ReDoc）

FastAPI 組み込みの API ドキュメントがそのまま使えます（サーバー起動後にブラウザで開くだけ。
リクエストボディのスキーマ・値域（speed 0.25〜4.0、response_format=flac）は実装と同じ
Pydantic モデルから自動生成されるため、常に実挙動と一致します）:

| URL | 内容 |
| --- | --- |
| `http://127.0.0.1:8000/docs` | Swagger UI。`Try it out` で実際に合成リクエストを実行できる（音声はモデル初回ロード後のため数秒かかる） |
| `http://127.0.0.1:8000/redoc` | ReDoc。読み専向けのリファレンス |
| `http://127.0.0.1:8000/openapi.json` | OpenAPI 3.1 仕様書（機械可読）。他ツールとの連携用 |

- ドキュメントページ自体に認証はありません。`--host 0.0.0.0` で公開する場合は、
  到達できる相手がブラウザから合成を実行できる点も踏まえて利用環境を選んでください

## テスト

```bash
uv run pytest                       # 偽モデル/偽BERTで完結（ネットワーク・GPU不要）
SBV2_REAL_TEST=1 SBV2_ASSETS_ROOT=$PWD/model_assets uv run pytest -m gpu   # 実モデル統合テスト
```

## 制約・注意

- 合成はプロセス内でロックにより直列化されます（単一 GPU 前提のため workers/レプリカで水平分散しないでください）
- SIGTERM（systemd / docker stop）ではモデルと pyopenjtalk ワーカーを解放してから終了します
  （uvicorn のシャットダウンを握りつぶさないよう、起動時に SIGTERM ハンドラを復元しています。詳細は VENDORED.md）
- 音声合成コアは**日本語のみ**。ローマ字のままの英単語は 1 文字ずつ読まれるため、カタカナ表記を推奨します
- **ライセンス**: 同梱コアは AGPL-3.0-or-later（一部 LGPL）。本サーバーも同ライセンスの方針です。
  再配布や公のサービス提供をする場合はライセンス条件を確認してください
