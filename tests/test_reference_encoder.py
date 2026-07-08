"""MelStyleEncoder の系列返却経路（Phase 3 MRTE 用）。return_sequence=False が現行と byte-identical
であること、True で pool 前の系列 [B,gin,T_ref] が返り、その time 平均が pooled c と一致することを検証する。
"""

import torch

from models.reference_encoder import MelStyleEncoder


def _encoder(n_mels=8, gin=16):
    torch.manual_seed(0)
    m = MelStyleEncoder(n_mels, style_vector_dim=gin, style_kernel_size=5, dropout=0.0)
    m.eval()
    return m


def test_return_sequence_false_is_byte_identical():
    # 既定経路（pooled のみ）は現行と完全一致
    m = _encoder()
    x = torch.randn(2, 8, 40)
    torch.manual_seed(1)
    w_default = m(x)
    torch.manual_seed(1)
    w_flag = m(x, return_sequence=False)
    assert torch.equal(w_default, w_flag)


def test_return_sequence_shapes():
    m = _encoder(gin=16)
    x = torch.randn(2, 8, 40)
    out = m(x, return_sequence=True)
    assert isinstance(out, tuple) and len(out) == 2
    w, ref_seq = out
    assert w.shape == (2, 16)  # pooled c
    assert ref_seq.shape == (2, 16, 40)  # 系列 [B, gin, T_ref]


def test_pooled_equals_mean_of_sequence():
    # ref_seq が pool 前の表現である保証: mask 無し時 w == ref_seq の time 平均
    m = _encoder(gin=16)
    x = torch.randn(2, 8, 40)
    w, ref_seq = m(x, return_sequence=True)
    assert torch.allclose(w, ref_seq.mean(dim=-1), atol=1e-5)
