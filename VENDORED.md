# Vendored Style-Bert-VITS2 Core

本リポジトリの `style_bert_vits2/` は、上流リポジトリのフォーク（学習コア）から
推論に必要なライブラリ部分をコピー（ベンダー同梱）したものである。
パス・設定の参照先をすべて本リポジトリ内に閉じるため、外部リポジトリへの path 依存は持たない。

## 取得元

| 項目 | 値 |
| --- | --- |
| 上流リポジトリ | https://github.com/estelf/Style-Bert-VITS2 |
| ベースコミット | `aae2d6233df9a5dec483592e56063e40945076e5` |
| 取得日 | 2026-10-29 |
| ライセンス | AGPL-3.0-or-later（`LICENSE` / `LGPL_LICENSE` を同梱） |

## 同梱したもの

- `style_bert_vits2/` … 上流からコピー（`__pycache__` は除く）。学習・前処理を呼ぶ
  `train/`・`preprocess/`・`gradio_tabs/`・`tests/` は含めない（推論パスのみが必要）
- `dict_data/` … pyopenjtalk 用ユーザー辞書のデフォルト（`constants.DEFAULT_USER_DICT_DIR` が参照）
- `LICENSE` / `LGPL_LICENSE` … 上流由来のライセンス表記

## 上流との差分

同梱コアへ加えた変更は以下の 1 件のみ（更新時は上流の該当コミットを再コピーし、
上の「ベースコミット」を更新した上でこのパッチを再適用する）。

1. `style_bert_vits2/constants.py` — `VERSION = _pkg_version("style-bert-vits2")` は
   「`style-bert-vits2` としてインストールされている」前提のため、当リポジトリ（サーバー本体と
   単一分布に同梱）ではメタデータ解決に失敗する。`PackageNotFoundError` を捕捉して
   ベースコミット一致の固定値 `"2.7.0-vendored+aae2d62"` へフォールバックするようパッチ済み
   （`VERSION` はコア内部で参照されない定数のため動作影響はない）

## 配置が別途必要なアセット（git 管理外）

- `pretrained/` … 事前学習済みモデル。**コードは参照しない**（取得は `style_bert_vits2.pretrained` が担当し、
  起動時に自動で揃う）。手動で揃えたい場合は `uv run -m style_bert_vits2.pretrained --only bert`
- `model_assets/` … 学習済み合成モデル（`<モデル名>/config.json` + `*.safetensors` + `style_vectors.npy`）。
  ユーザーが配置する。既定の走査先は本リポジトリの `./model_assets`（環境変数 `SBV2_ASSETS_ROOT` で変更可）
