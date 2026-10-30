"""記号・空白位置の「一呼吸」と文末記号直後の「文切れ」無音を挿入する仕組み。

Style-Bert-VITS2 の正規化（`normalize_text`）では空白は削除され、括弧類や区切り記号など
読み上げに寄与しない記号も削除（または `'` 音素へ縮退）されるため、記号や空白をはさんだ
文章が無呼吸で繋がって読まれがちだった。

本モジュールは発話をセグメントに分割する。境界（区切りグループ）は 2 種類で、
シンボル一覧と無音長はそれぞれ別々に設定できる（無音長の挿入自体は呼び出し側の
アプリ層が合成結果へ直接行う。`TTSModel.infer` の `line_split` は単一间隔の
無音しか挿入できないため使わない）:

- 文末級（sentence）: 文末表現のシンボル（既定 `。`。環境変数 `SENTENCE_SYMBOLS` で
  `。？` のように変更可）の直後。無音長は `sentence_interval`。
  シンボル自体はセグメント末尾に保持するので、モデル自身の文末ポーズ・下降調は
  残る（その上に固定無音が加算される）
- 非文末級（breath）: 一呼吸対象のシンボル（既定は括弧類・区切り記号類。環境変数
  `BREATH_SYMBOLS` で変更可）と空白全般の位置。無音長は `breath_interval`

空白（半角・全角・改行・タブ・NBSP 等）とゼロ幅文字はシンボル設定に関わらず
常に一呼吸の対象。同じシンボルを両グループに指定した場合は文末級が優先される。

既定のシンボル対象外（必要なら設定で追加する）:
- `、` `！` `？` `…` `：` `・` 等 → 正規化で `,` 等になり、モデル自身がポーズを入れる
- `ー` `～` `〜` `~` → 長音として発音される
- `$` → 「ドル」「円」等に正規化されて発音される（区切りとして使うと不自然な位置で切れる）
- ASCII の `'` `"` `` ` `` → 英語の省略形（don't）や引用の途中が切れるのを防ぐ
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Literal, NamedTuple

#: 境界の種類（最後のセグメントは ""）
Boundary = Literal["", "breath", "sentence"]


class Segment(NamedTuple):
    text: str
    boundary: Boundary  #: このセグメントの直後の無音の種類


#: 非文末級（一呼吸）の既定シンボル（括弧類・区切り記号類。半角・全角とも）
DEFAULT_BREATH_SYMBOLS = (
    "()（）[]［］{}｛｝<>〈〉《》【】〔〕「」『』“”"
    "#＃@＊*/／\\＼|｜_＿^&+＋=%"
)

#: 文末級の既定シンボル
DEFAULT_SENTENCE_SYMBOLS = "。"

#: 空白類全般（`\s` は改行・タブ・全角スペース `\u3000`・NBSP などを含む）とゼロ幅文字。
#: シンボル設定に関わらず常に一呼吸の対象
_WHITESPACE_CLASS = r"\s\u200b-\u200d\ufeff"

#: 文末シンボル run の直後に立てる内部マーカー。区切り種別の判定に使うだけで、
#: 本文には決して残さない（入力や設定に混入していたら除去する）
_SENTENCE_MARK = "\x00"

#: 内容を持たないセグメント（句読点だけの断片等）の除去に使う既定文字集合
#: （文末シンボルとして設定された文字も加える）
_CONTENTLESS_BASE = set("。．.、，,！!？?…")


class _Patterns(NamedTuple):
    both: re.Pattern[str]  #: 呼吸＋文末の区切り（キャプチャ付き。split 結果から種別を判定できる）
    breath_only: re.Pattern[str]
    sentence_only: re.Pattern[str] | None  #: 文末シンボル未設定なら None
    sentence_run: re.Pattern[str] | None  #: 文末シンボルの run（マーカー挿入用）


@lru_cache(maxsize=16)
def _compile_patterns(breath_symbols: str, sentence_symbols: str) -> _Patterns:
    """設定されたシンボル文字列から分割用正規表現を構築する（設定ごとキャッシュ）"""
    # 内部マーカーとの混同を防ぐ（NUL は正規化でも消える文字）
    breath_cls = re.escape(breath_symbols.replace(_SENTENCE_MARK, "")) + _WHITESPACE_CLASS
    mark = re.escape(_SENTENCE_MARK)
    sentence_cls = re.escape(sentence_symbols.replace(_SENTENCE_MARK, ""))
    return _Patterns(
        both=re.compile(f"([{breath_cls}{mark}]+)" if sentence_cls else f"([{breath_cls}]+)"),
        breath_only=re.compile(f"([{breath_cls}]+)"),
        sentence_only=re.compile(f"([{mark}]+)") if sentence_cls else None,
        sentence_run=re.compile(f"([{sentence_cls}]+)") if sentence_cls else None,
    )


def split_breath(
    text: str,
    *,
    with_breath: bool = True,
    with_sentence: bool = True,
    breath_symbols: str = DEFAULT_BREATH_SYMBOLS,
    sentence_symbols: str = DEFAULT_SENTENCE_SYMBOLS,
) -> list[Segment]:
    """一呼吸シンボル・空白と文末シンボルの位置で発話をセグメント分割する。

    Args:
        text (str): 合成リクエストの原文（正規化前）
        with_breath (bool, optional): 一呼吸シンボル・空白位置で分割する。Defaults to True.
        with_sentence (bool, optional): 文末シンボルの直後で分割する（シンボルはセグメント
            末尾に保持）。Defaults to True.
        breath_symbols (str, optional): 一呼吸対象のシンボル（空白類は常に含む）。
            Defaults to DEFAULT_BREATH_SYMBOLS.
        sentence_symbols (str, optional): 文末対象のシンボル。Defaults to DEFAULT_SENTENCE_SYMBOLS.

    Returns:
        list[Segment]: 分割結果。各要素の boundary は直後の無音の種類で、最後の
        セグメントは ""。区切りが無いか 1 セグメントしか残らない場合は空リストを返す
        ＝分割なしとして扱う（原文のまま 1 回で合成する）
    """
    # 内部マーカーと入力中の NUL 文字の混同を防ぐ
    text = text.replace(_SENTENCE_MARK, "")
    patterns = _compile_patterns(breath_symbols, sentence_symbols)
    if with_sentence and patterns.sentence_run is not None:
        # 文末シンボルの連続（「。。。」「？！」等）は 1 つの文切れとして扱う
        text = patterns.sentence_run.sub(lambda m: m.group() + _SENTENCE_MARK, text)
    else:
        with_sentence = False

    if with_breath and with_sentence:
        pattern = patterns.both
    elif with_breath:
        pattern = patterns.breath_only
    elif with_sentence and patterns.sentence_only is not None:
        pattern = patterns.sentence_only
    else:
        return []

    # キャプチャ付き split なので parts は [本文, 区切り, 本文, 区切り, ..., 本文] 順
    parts = pattern.split(text)
    contentless = _CONTENTLESS_BASE | set(sentence_symbols)

    segments: list[Segment] = []
    pending: Boundary = ""  # 直前に読んだ区切りの種別（次本文の先頭に付与）
    for index, part in enumerate(parts):
        if index % 2 == 1:
            # 区切り文字列。マーカーを含むなら文末級（呼吸より長いので優先）
            pending = "sentence" if _SENTENCE_MARK in part else "breath"
            continue
        if part == "" or set(part) <= contentless:
            # 空・句読点のみの断片は合成対象にならないので境界として残さない
            continue
        if segments and pending:
            # 直前要素に付いていた空境界を、実際の区切り種別へ確定する
            prev_text, _ = segments[-1]
            segments[-1] = Segment(prev_text, pending)
        segments.append(Segment(part, ""))

    if len(segments) <= 1:
        return []
    return segments
