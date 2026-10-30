"""セグメント分割（breath.split_breath）のユニットテスト。モデル・GPU 不要。"""

from __future__ import annotations

from style_bert_vits2_openai_tts_api.breath import Segment, split_breath


def test_half_width_space_splits_as_breath():
    assert split_breath("KernelSU Issue #2113") == [
        Segment("KernelSU", "breath"),
        Segment("Issue", "breath"),
        Segment("2113", ""),
    ]


def test_various_whitespace_splits():
    assert split_breath("今日は 明日") == [Segment("今日は", "breath"), Segment("明日", "")]
    assert split_breath("a\tb") == [Segment("a", "breath"), Segment("b", "")]
    assert split_breath("x\u200by") == [Segment("x", "breath"), Segment("y", "")]
    assert split_breath("複数  連続\u00a0空白") == [
        Segment("複数", "breath"),
        Segment("連続", "breath"),
        Segment("空白", ""),
    ]


def test_brackets_and_separators_split():
    assert split_breath("列の置換（permutation）です") == [
        Segment("列の置換", "breath"),
        Segment("permutation", "breath"),
        Segment("です", ""),
    ]
    assert split_breath("A|B C/D") == [
        Segment("A", "breath"),
        Segment("B", "breath"),
        Segment("C", "breath"),
        Segment("D", ""),
    ]


def test_run_collapsed_into_single_breath():
    # 記号・空白の連続は 1 呼吸にまとまる
    assert split_breath("単語 （かっこと スペース） 次") == [
        Segment("単語", "breath"),
        Segment("かっこと", "breath"),
        Segment("スペース", "breath"),
        Segment("次", ""),
    ]


def test_period_splits_as_sentence_and_kept():
    # 「。」はセグメント末尾に保持し、境界は sentence（呼吸とは別扱い）
    assert split_breath("今日はいい天気です。外出しましょう。") == [
        Segment("今日はいい天気です。", "sentence"),
        Segment("外出しましょう。", ""),
    ]
    # 文末（最後）の「。」の後ろには無音を挿さない（boundary は ""）


def test_mixed_boundaries_sentence_wins():
    # 「。」直後のスペースは 1 つの区切りになり sentence が優先
    assert split_breath("問題が発生しました。 続きをどうぞ") == [
        Segment("問題が発生しました。", "sentence"),
        Segment("続きをどうぞ", ""),
    ]
    # 呼吸 → 文切れ の混在
    assert split_breath("A B。C D") == [
        Segment("A", "breath"),
        Segment("B。", "sentence"),
        Segment("C", "breath"),
        Segment("D", ""),
    ]


def test_consecutive_periods_single_sentence_boundary():
    assert split_breath("えーと。。。つまり") == [
        Segment("えーと。。。", "sentence"),  # run を丸ごと末尾に保持し 1 文切れ
        Segment("つまり", ""),
    ]


def test_model_punctuation_and_pronounced_symbols_not_split():
    # 「。」以外でモデル自身がポーズを入れる記号・長音・通貨・ASCII 引用符は対象外
    for text in [
        "こんにちは、世界！元気？……そう思います",  # 「、」「！」「？」だけ → 分割なし
        "さっき一気に読んだ〜",
        "価格は$100です",
        "重要：次へ",
        "don't",
        "1対1対応の入れ替え",
    ]:
        assert split_breath(text) == [], text


def test_no_split_returns_empty():
    assert split_breath("こんにちは") == []


def test_symbols_only_returns_empty():
    # 内容あるセグメントが残らない場合は分割なし（空テキストで infer が落ちる経路を避ける）
    assert split_breath("（　）#＊") == []
    assert split_breath("。。。") == []  # 「。」のみの断片は合成対象にしない


def test_with_sentence_disabled():
    # 呼吸のみ有効なら「。」では分割しない
    assert split_breath("天気です。外出", with_breath=True, with_sentence=False) == []
    assert split_breath("天気 です。外出", with_breath=True, with_sentence=False) == [
        Segment("天気", "breath"),
        Segment("です。外出", ""),
    ]


def test_with_breath_disabled():
    # 文切れのみ有効なら空白では分割しない（空白は本文内に残る。正規化で消える）
    assert split_breath("天気 です", with_breath=False, with_sentence=True) == []
    assert split_breath("天気 です。外出", with_breath=False, with_sentence=True) == [
        Segment("天気 です。", "sentence"),
        Segment("外出", ""),
    ]


def test_both_disabled_returns_empty():
    assert split_breath("天気です。 外出", with_breath=False, with_sentence=False) == []


def test_nul_in_input_never_becomes_boundary():
    # 内部マーカーと同じ NUL 文字が入力に混入しても文切れとして扱わない（除去される）
    assert split_breath("あ\x00い") == []


def test_custom_sentence_symbols():
    # 文末シンボルを設定で拡張（例: SENTENCE_SYMBOLS=。？）
    kw = {"breath_symbols": ""}
    assert split_breath("本気？はい。了解", with_breath=False, sentence_symbols="。？", **kw) == [
        Segment("本気？", "sentence"),
        Segment("はい。", "sentence"),
        Segment("了解", ""),
    ]
    # 既定では「？」は文切れにならない（文末が末尾 1 箇所のみ → 分割なし）
    assert split_breath("本気？はい。", with_breath=False, sentence_symbols="。") == []


def test_custom_breath_symbols():
    # 一呼吸シンボルを設定で追加（既定では「・」は対象外）
    assert split_breath("赤・青・緑") == []
    assert split_breath("赤・青・緑", breath_symbols="・") == [
        Segment("赤", "breath"),
        Segment("青", "breath"),
        Segment("緑", ""),
    ]


def test_overlapping_symbols_sentence_wins():
    # 同じシンボルを両グループに指定したら無音種別は文末級が優先
    assert split_breath("赤・青", breath_symbols="・", sentence_symbols="・") == [
        Segment("赤", "sentence"),
        Segment("青", ""),
    ]


def test_whitespace_always_breath_regardless_of_symbols():
    # 空白類はシンボル設定に関わらず常に一呼吸
    assert split_breath("A B", breath_symbols="", sentence_symbols="。") == [
        Segment("A", "breath"),
        Segment("B", ""),
    ]


def test_regex_special_chars_in_symbols():
    # 正規表現特殊文字を含む設定でも壊れない（「|」「.」「+」等）
    assert split_breath("a|b.c+d", breath_symbols="", sentence_symbols=".+") == [
        Segment("a|b.", "sentence"),
        Segment("c+", "sentence"),
        Segment("d", ""),
    ]
