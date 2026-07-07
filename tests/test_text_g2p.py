"""g2p フロントエンド（日/英/中）と ID 変換の契約。

g2p の厳密な音素列は依存ライブラリ（pyopenjtalk-plus / eng-to-ipa / jieba+pypinyin）の版に
依存する（volatile）。よって「決定論性・list 型・非空・全トークン ∈ symbols」を常時ガードし、
厳密ゴールデンは uv.lock でピンされた現行環境に対する回帰ガードとして別テストに分離する
（ライブラリ更新で音素が変わったら、その変更を意図的に検知してゴールデンを更新する）。
"""

import pytest

from text import cleaned_text_to_sequence, symbols
from text.english import english_to_ipa2
from text.japanese import japanese_to_ipa2
from text.mandarin import chinese_to_cnm3

_SYMSET = set(symbols)

# --- 現行環境（uv.lock ピン）に対する厳密ゴールデン。lib 更新で変わりうる ---
JP_TEXT, JP_GOLDEN = "こんにちは", ["k", "o", "↑", "n", "n", "^", "i", "t", "ʃ", "i", "w", "a"]
EN_TEXT, EN_GOLDEN = "Hello world.", ["h", "ɛ", "ˈ", "l", "o", "ʊ", " ", "w", "ə", "ɹ", "ɫ", "d", "."]
ZH_TEXT, ZH_GOLDEN = "你好，世界。", ["n3", "i3", "h3", "A03", "O03", ",", "sh4", "ir4", "j4", "ie4", "."]

G2P_CASES = [
    pytest.param(japanese_to_ipa2, JP_TEXT, id="japanese"),
    pytest.param(english_to_ipa2, EN_TEXT, id="english"),
    pytest.param(chinese_to_cnm3, ZH_TEXT, id="chinese"),
]


@pytest.mark.parametrize(("g2p", "text"), G2P_CASES)
def test_g2p_returns_nonempty_list(g2p, text):
    out = g2p(text)
    assert isinstance(out, list)
    assert len(out) > 0


@pytest.mark.parametrize(("g2p", "text"), G2P_CASES)
def test_g2p_is_deterministic(g2p, text):
    # 同一入力で常に同じ音素列（乱数・時刻依存が無いこと）
    assert g2p(text) == g2p(text)


@pytest.mark.parametrize(("g2p", "text"), G2P_CASES)
def test_g2p_tokens_are_in_symbols(g2p, text):
    # 全トークンが記号表に含まれる = cleaned_text_to_sequence で silent-drop されない
    out = g2p(text)
    unknown = [t for t in out if t not in _SYMSET]
    assert unknown == []


@pytest.mark.parametrize(
    ("g2p", "text", "golden"),
    [
        pytest.param(japanese_to_ipa2, JP_TEXT, JP_GOLDEN, id="japanese"),
        pytest.param(english_to_ipa2, EN_TEXT, EN_GOLDEN, id="english"),
        pytest.param(chinese_to_cnm3, ZH_TEXT, ZH_GOLDEN, id="chinese"),
    ],
)
def test_g2p_golden_pinned(g2p, text, golden):
    # uv.lock の依存版に対するゴールデン。失敗した場合は依存更新で音素が変化した可能性 → 意図確認の上更新する
    assert g2p(text) == golden


def test_cleaned_text_to_sequence_maps_tokens():
    toks = ["n3", "i3"]
    assert cleaned_text_to_sequence(toks) == [symbols.index("n3"), symbols.index("i3")]


def test_cleaned_text_to_sequence_drops_unknown_silently():
    # 記号表に無いトークンは例外を投げず除外されるだけ（本番経路の挙動）
    assert cleaned_text_to_sequence(["n3", "___NOT_A_SYMBOL___", "i3"]) == [
        symbols.index("n3"),
        symbols.index("i3"),
    ]
