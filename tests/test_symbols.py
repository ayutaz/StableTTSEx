"""記号表と n_vocab の不変条件（チェックポイント互換の最重要ガード）。

symbols は nn.Embedding(n_vocab) と既存 checkpoint の埋め込み行数の唯一の真実源。
値が変わると load_state_dict が静かに失敗/非互換になる。純 Python なので text パッケージの
重量 import（jieba/pyopenjtalk）を避けて symbols.py を直接ロードし高速・決定論的に検証する。
"""

import importlib.util
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _load_symbols_isolated():
    spec = importlib.util.spec_from_file_location("_symbols_isolated", REPO / "text" / "symbols.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_n_vocab_is_401():
    # StableTTS(len(symbols), ...) と checkpoint_0.pt の encoder.emb=(401,256) の前提
    assert len(_load_symbols_isolated().symbols) == 401


def test_no_duplicate_symbols():
    s = _load_symbols_isolated().symbols
    assert len(set(s)) == len(s)


def test_pad_is_first_symbol():
    # intersperse の blank=0 が pad '_' の ID と一致する前提
    assert _load_symbols_isolated().symbols[0] == "_"


def test_symbol_composition():
    m = _load_symbols_isolated()
    assert [len(m._pad), len(m._punctuation), len(m._IPA_letters), len(m._CNM3_letters), len(m._additional)] == [
        1,
        8,
        60,
        330,
        2,
    ]
    # symbols.py の構成式（symbols = [_pad] + list(_punctuation) + list(_IPA_letters) + _CNM3_letters + _additional）
    assert m.symbols == [m._pad] + list(m._punctuation) + list(m._IPA_letters) + m._CNM3_letters + m._additional


def test_space_id():
    m = _load_symbols_isolated()
    assert m.symbols.index(" ") == 68
    assert m.SPACE_ID == 68


def test_symbol_id_bijection():
    # 本番経路（text/__init__.py）の _symbol_to_id / _id_to_symbol が互いに逆写像
    import text
    from text import symbols as sym_list

    assert text._symbol_to_id["_"] == 0
    for i, sym in enumerate(sym_list):
        assert text._symbol_to_id[sym] == i
        assert text._id_to_symbol[i] == sym
