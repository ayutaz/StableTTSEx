"""Phase 2 施策6: EMA の数値・resume・デバイス整合。CPU で決定論的に検証する
（GPU cross-device 分岐は CPU では no-op のため、ここでは値・dtype・resume・device 属性まで保証）。
"""

import torch

from utils.ema import EMA


def _lin_with_int_buffer(fill=0.0):
    lin = torch.nn.Linear(4, 4, bias=False)
    with torch.no_grad():
        lin.weight.fill_(fill)
    lin.register_buffer("int_buf", torch.tensor([1, 2], dtype=torch.long))
    return lin


def test_decay_warmup_formula():
    # decay(step) = min(decay_max, (1+n)/(warmup+n))
    ema = EMA(_lin_with_int_buffer(), decay=0.9, warmup=10)
    ema.num_updates = 0
    assert ema._decay() == min(0.9, 1 / 10)
    ema.num_updates = 5
    assert ema._decay() == min(0.9, 6 / 15)
    ema.num_updates = 1000
    assert ema._decay() == 0.9  # 上限に張り付く


def test_float_update_math():
    lin = _lin_with_int_buffer(fill=0.0)
    ema = EMA(lin, decay=0.9, warmup=0)
    with torch.no_grad():
        lin.weight.fill_(1.0)
    ema.update(lin)
    # n=1, warmup=0 → decay=min(0.9,(1+1)/(0+1))=0.9 → shadow = 0.9*0 + 0.1*1 = 0.1
    w = ema.state_dict()["weight"]
    assert torch.allclose(w, torch.full_like(w, 0.1))


def test_exponential_average_uses_prior_shadow():
    # 非ゼロ shadow から更新し、保持係数 mul_(d) を検証（fill=0 だと d*0=0 で係数がすり抜ける）。
    # shadow 初期=2.0（param 由来）→ weight を 1.0 にして update → 0.9*2.0 + 0.1*1.0 = 1.9
    lin = _lin_with_int_buffer(fill=2.0)
    ema = EMA(lin, decay=0.9, warmup=0)
    with torch.no_grad():
        lin.weight.fill_(1.0)
    ema.update(lin)
    w = ema.state_dict()["weight"]
    assert torch.allclose(w, torch.full_like(w, 1.9))


def test_int_buffer_dtype_preserved_and_follows_latest():
    lin = _lin_with_int_buffer()
    ema = EMA(lin, decay=0.9, warmup=0)
    with torch.no_grad():
        lin.int_buf.fill_(9)
    ema.update(lin)
    ib = ema.state_dict()["int_buf"]
    assert ib.dtype == torch.long
    assert torch.equal(ib, torch.tensor([9, 9]))


def test_state_dict_keys_match_module():
    lin = _lin_with_int_buffer()
    ema = EMA(lin, decay=0.9, warmup=0)
    assert set(ema.state_dict().keys()) == set(lin.state_dict().keys())


def test_shadow_is_fp32_even_if_param_half():
    lin = torch.nn.Linear(4, 4, bias=False).half()
    ema = EMA(lin, decay=0.9, warmup=0)
    assert ema.state_dict()["weight"].dtype == torch.float32


def test_resume_roundtrip_restores_values_and_num_updates():
    lin = _lin_with_int_buffer(fill=0.0)
    ema = EMA(lin, decay=0.9, warmup=0)
    with torch.no_grad():
        lin.weight.fill_(1.0)
    ema.update(lin)
    state = ema.ema_training_state()
    assert state["num_updates"] == 1 and "shadow" in state

    ema2 = EMA(_lin_with_int_buffer(fill=5.0), decay=0.9, warmup=0)
    ema2.load_ema_training_state(state)
    assert ema2.num_updates == 1
    assert torch.allclose(ema2.state_dict()["weight"], torch.full((4, 4), 0.1))


def test_device_attr_and_shadow_colocated():
    # HIGH 修正の回帰ガード: shadow は self.device に載る（resume 復元でも同一デバイスへ戻す）。
    # CPU CI では device=cpu。CPU 保存 → CPU ロード → update がクラッシュしないことも確認。
    lin = _lin_with_int_buffer(fill=0.0)
    ema = EMA(lin, decay=0.9, warmup=0)
    assert ema.device == torch.device("cpu")
    assert all(v.device == ema.device for v in ema.state_dict().values())

    saved = {"shadow": {k: v.detach().cpu() for k, v in ema.ema_training_state()["shadow"].items()}, "num_updates": 3}
    ema.load_ema_training_state(saved)
    assert all(v.device == ema.device for v in ema.state_dict().values())
    ema.update(lin)  # cross-device なら例外。CPU 一致なら通る
