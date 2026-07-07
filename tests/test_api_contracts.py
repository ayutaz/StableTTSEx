"""推論・前処理の契約。g2p_mapping は api.py と preprocess.py の2箇所に重複定義されており、
言語追加時に両方の更新が必要（CLAUDE.md の既知の落とし穴）。preprocess.py は import 時に
mkdir 等の副作用があるため、AST 静的解析で両者の一致を検証する（import しない）。
"""

import ast
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _extract_g2p_mapping(source_path):
    """`g2p_mapping = {...}` / `self.g2p_mapping = {...}` の辞書リテラルを {str_key: 関数名} として抽出。"""
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Dict):
            continue
        for target in node.targets:
            is_g2p = (isinstance(target, ast.Name) and target.id == "g2p_mapping") or (
                isinstance(target, ast.Attribute) and target.attr == "g2p_mapping"
            )
            if is_g2p:
                return {
                    key.value: value.id
                    for key, value in zip(node.value.keys, node.value.values, strict=True)
                    if isinstance(key, ast.Constant) and isinstance(value, ast.Name)
                }
    raise AssertionError(f"g2p_mapping の辞書リテラルが {source_path} に見つからない")


def test_g2p_mapping_is_in_sync_between_api_and_preprocess():
    api_mapping = _extract_g2p_mapping(REPO / "api.py")
    pre_mapping = _extract_g2p_mapping(REPO / "preprocess.py")
    assert api_mapping == pre_mapping


def test_g2p_mapping_covers_expected_languages():
    api_mapping = _extract_g2p_mapping(REPO / "api.py")
    assert set(api_mapping.keys()) == {"chinese", "japanese", "english"}


def test_g2p_mapping_points_to_expected_functions():
    # 値（対応する g2p 関数）まで固定。両ファイルに同一の誤マッピングが入っても検知する
    expected = {"chinese": "chinese_to_cnm3", "japanese": "japanese_to_ipa2", "english": "english_to_ipa2"}
    assert _extract_g2p_mapping(REPO / "api.py") == expected
    assert _extract_g2p_mapping(REPO / "preprocess.py") == expected
