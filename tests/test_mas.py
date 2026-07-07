"""GPU ネイティブ MAS（utils/mas.py::maximum_path_torch）が numba 版
（monotonic_align.maximum_path）と **ビット同一**のアラインメントを返すことを検証する。
これが崩れると学習の教師アラインメントが静かに変わるため、厳密一致を不変条件として固定する。
"""

import torch

import monotonic_align
from utils.mas import maximum_path_torch


def _random_case(b, max_ty, max_tx, seed):
    # 実データの regime（mel フレーム >= text トークン）に合わせ、矩形の有効領域マスクを作る。
    # mask は attn_mask（x_mask ⊗ y_mask）と同じく [b, t_y, t_x] の矩形ブロック
    g = torch.Generator().manual_seed(seed)
    neg_cent = torch.randn(b, max_ty, max_tx, generator=g)
    mask = torch.zeros(b, max_ty, max_tx)
    for i in range(b):
        tx = int(torch.randint(1, max_tx + 1, (1,), generator=g))
        ty = int(torch.randint(tx, max_ty + 1, (1,), generator=g))  # ty >= tx（単調全被覆が可能）
        mask[i, :ty, :tx] = 1.0
    return neg_cent, mask


def _assert_matches(b, max_ty, max_tx, seed):
    neg_cent, mask = _random_case(b, max_ty, max_tx, seed)
    ref = monotonic_align.maximum_path(neg_cent.clone(), mask.clone())
    got = maximum_path_torch(neg_cent.clone(), mask.clone())
    assert got.shape == ref.shape
    assert got.dtype == ref.dtype
    assert torch.equal(got, ref), f"MAS mismatch at b={b}, ty={max_ty}, tx={max_tx}, seed={seed}"


def test_gpu_mas_matches_numba_various_shapes():
    for seed in range(20):
        _assert_matches(b=4, max_ty=48, max_tx=13, seed=seed)


def test_gpu_mas_matches_numba_batched_varied_lengths():
    # バッチ内で長さがばらつくケース（padding 領域の扱いを検証）
    for seed in range(20):
        _assert_matches(b=8, max_ty=64, max_tx=20, seed=100 + seed)


def test_gpu_mas_matches_numba_edge_single_token():
    # t_x=1（全 mel フレームが単一トークンに写像）
    for seed in range(5):
        _assert_matches(b=3, max_ty=16, max_tx=1, seed=200 + seed)


def test_gpu_mas_matches_numba_square():
    # t_x ≈ t_y（対角バンドが狭い、対角一直線に近い）
    for seed in range(10):
        _assert_matches(b=2, max_ty=12, max_tx=12, seed=300 + seed)


def test_gpu_mas_path_is_valid_alignment():
    # 各 mel フレームがちょうど 1 つの text トークンに割り当たる（行和=1、有効領域内）
    neg_cent, mask = _random_case(b=4, max_ty=48, max_tx=13, seed=7)
    path = maximum_path_torch(neg_cent, mask)
    t_y_len = mask.sum(1)[:, 0].long()
    for i in range(path.shape[0]):
        valid_rows = path[i, : t_y_len[i]]
        assert torch.equal(valid_rows.sum(1), torch.ones(int(t_y_len[i])))
