"""データ供給の契約（GPU/DDP 不要の部品）: intersperse・collate_fn・random_slice_tensor・
DistributedBucketSampler。乱数は stdlib random（collate/slice）と torch.Generator（sampler）を
使うため、それぞれ random.seed / num_replicas+rank 明示で決定論化する。
"""

import random

import torch

from datas.dataset import collate_fn, intersperse, random_slice_tensor
from datas.sampler import DistributedBucketSampler


# --- intersperse（GlowTTS 由来の blank 挿入） ---
def test_intersperse_golden():
    assert intersperse([5, 6, 7], 0) == [0, 5, 0, 6, 0, 7, 0]


def test_intersperse_empty():
    assert intersperse([], 0) == [0]


def test_intersperse_properties():
    lst = [3, 1, 4, 1, 5, 9]
    r = intersperse(lst, 0)
    assert len(r) == 2 * len(lst) + 1
    assert r[::2] == [0] * (len(lst) + 1)  # 偶数位置は全て blank
    assert r[1::2] == lst  # 奇数位置は元列


# --- random_slice_tensor（reference encoder 用のランダム断片） ---
def test_random_slice_short_returns_full():
    x = torch.randn(4, 8)  # length 8 < 12
    assert random_slice_tensor(x) is x


def test_random_slice_within_bounds_and_deterministic():
    x = torch.randn(4, 120)
    random.seed(0)
    s1 = random_slice_tensor(x)
    random.seed(0)
    s2 = random_slice_tensor(x)
    assert torch.equal(s1, s2)  # random.seed で完全一致（shape だけでなく値も）
    # segment は [length//12, length//3] の範囲
    assert 120 // 12 <= s1.size(-1) <= 120 // 3
    assert s1.size(0) == 4  # 非時間軸は保持
    # 返り値が x の連続窓 x[..., start:start+seg] であること（同一 seed で start/seg を再現して照合）。
    # x[...,0:seg] や範囲/オフバイワン mutation を検知する
    random.seed(0)
    seg = random.randint(120 // 12, 120 // 3)
    start = random.randint(0, 120 - seg)
    assert torch.equal(s1, x[..., start : start + seg])


# --- collate_fn（パディング + 参照スライス） ---
def test_collate_fn_shapes_and_lengths():
    random.seed(0)
    n_mels = 8
    batch = [
        (torch.randn(n_mels, 40), torch.tensor([1, 2, 3], dtype=torch.long)),
        (torch.randn(n_mels, 60), torch.tensor([1, 2, 3, 4, 5], dtype=torch.long)),
    ]
    texts_padded, text_lengths, mels_padded, mel_lengths, sliced_padded, sliced_lengths = collate_fn(batch)

    assert texts_padded.shape == (2, 5)  # max text length
    assert torch.equal(text_lengths, torch.tensor([3, 5]))
    assert mels_padded.shape == (2, n_mels, 60)  # max mel length
    assert torch.equal(mel_lengths, torch.tensor([40, 60]))
    # 参照スライスは元の mel 長以下・1 以上、非時間軸は n_mels
    assert sliced_padded.shape[:2] == (2, n_mels)
    assert torch.all(sliced_lengths >= 1)
    assert torch.all(sliced_lengths <= mel_lengths)


# --- DistributedBucketSampler ---
class _LenDataset:
    def __init__(self, lengths):
        self.lengths = lengths

    def __len__(self):
        return len(self.lengths)


def _make_sampler(lengths, boundaries, batch_size=2, shuffle=False):
    # num_replicas/rank を明示すれば dist プロセスグループ不要で CPU 構築可能
    return DistributedBucketSampler(
        _LenDataset(lengths),
        batch_size=batch_size,
        boundaries=list(boundaries),  # in-place pop 破壊を避けるためコピーを渡す
        num_replicas=1,
        rank=0,
        shuffle=shuffle,
    )


def test_bisect_assigns_bucket_by_length():
    s = _make_sampler([150] * 4, [100, 200])
    # boundaries=[100,200] は1バケット: 100 < x <= 200 → idx 0、範囲外 → -1
    assert s._bisect(150) == 0
    assert s._bisect(100) == -1  # 下限は含まない（length <= b1 は捨てる）
    assert s._bisect(250) == -1  # 上限超えは捨てる


def test_sampler_iterates_full_batches():
    s = _make_sampler([150] * 8, [100, 200], batch_size=2)
    s.set_epoch(0)
    batches = list(iter(s))
    assert len(batches) == len(s)
    assert len(batches) > 0
    for b in batches:
        assert len(b) == 2
        assert all(0 <= idx < 8 for idx in b)


def test_sampler_mutates_boundaries_in_place_when_low_bucket_empty():
    # 既知ハザードの characterization: 空バケットがあると _create_buckets が boundaries を in-place pop する。
    # train.py はプロセス毎にリテラルを渡すので実害はないが、共有リストを渡すと2回目以降で境界が壊れる。
    boundaries = [0, 100, 200]  # 下バケット(0,100] は空（全 length=150）
    DistributedBucketSampler(_LenDataset([150] * 4), batch_size=2, boundaries=boundaries, num_replicas=1, rank=0)
    assert boundaries != [0, 100, 200]  # 破壊されている（現状挙動の記録）
