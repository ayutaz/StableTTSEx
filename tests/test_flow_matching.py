"""CFMDecoder（flow matching）の数値契約。Phase 1（sway/CFG）と Phase 2（logit-normal）の
回帰ガードを含む。値ゴールデンは版依存で volatile なため、有限性・shape・同一 seed 内一致・
関係性（cfg=1.0 で誘導なし = cond 出力）・数学的性質（sway warp）を検証する。
"""

import math

import pytest
import torch

from models.flow_matching import CFMDecoder

N_MELS = 8


def _inputs(t_len=20):
    x1 = torch.randn(1, N_MELS, t_len)
    mask = torch.ones(1, 1, t_len)
    mu = torch.randn(1, N_MELS, t_len)
    c = torch.randn(1, 16)
    return x1, mask, mu, c


@pytest.mark.parametrize("sampling", ["cosine", "logit_normal"])
def test_compute_loss_finite_and_shape(tiny_cfm, sampling):
    dec = tiny_cfm(timestep_sampling=sampling)
    x1, mask, mu, c = _inputs()
    torch.manual_seed(0)
    loss, y = dec.compute_loss(x1, mask, mu, c)
    assert torch.isfinite(loss)
    assert loss.ndim == 0
    assert y.shape == x1.shape


def test_compute_loss_cosine_is_deterministic(tiny_cfm):
    dec = tiny_cfm(timestep_sampling="cosine")
    x1, mask, mu, c = _inputs()

    def run():
        torch.manual_seed(123)
        return dec.compute_loss(x1, mask, mu, c)[0]

    assert torch.equal(run(), run())


def test_invalid_timestep_sampling_raises():
    with pytest.raises(ValueError, match="timestep_sampling"):
        CFMDecoder(N_MELS, N_MELS, 16, N_MELS, 32, 2, 2, 3, 0.0, 16, timestep_sampling="lognormal")


def test_cfg_strength_one_equals_cond_output(tiny_cfm):
    # Phase 1 ビット不変: cfg=1.0 は uncond 前向きを省略し cond 出力そのものになる（誘導なし）
    dec = tiny_cfm()
    t = torch.tensor(0.5)
    x = torch.randn(1, N_MELS, 20)
    mask = torch.ones(1, 1, 20)
    mu = torch.randn(1, N_MELS, 20)
    c = torch.randn(1, 16)
    cfg_kwargs = {
        "fake_speaker": torch.zeros(1, 16),
        "fake_content": torch.zeros(1, N_MELS, 1),
        "cfg_strength": 1.0,
        "cfg_rescale": 0.0,
        "cfg_interval": None,
        "slg_scale": 0.0,
        "slg_layers": (1,),
        "slg_t_range": (0.0, 0.5),
    }
    torch.manual_seed(1)
    cond = dec.estimator(t, x, mask, mu, c)
    torch.manual_seed(1)
    wrapped = dec.cfg_wrapper(t, x, mask, mu, c, cfg_kwargs)
    assert torch.equal(cond, wrapped)


def test_sway_warp_s_minus_one_matches_cosine_schedule():
    # sway warp: t_span + s*(cos(pi/2 t)-1+t)。s=-1 で学習時 cosine スケジュール 1-cos(pi/2 t) に一致
    # （flow_matching.py の warp 式に対応する数学的性質の回帰ガード）
    t_span = torch.linspace(0, 1, 21)
    s = -1.0
    warp = t_span + s * (torch.cos(torch.pi / 2 * t_span) - 1 + t_span)
    cosine_sched = 1 - torch.cos(torch.pi / 2 * t_span)
    assert torch.allclose(warp, cosine_sched, atol=1e-6)
    # 端点保存
    assert warp[0].item() == pytest.approx(0.0, abs=1e-6)
    assert warp[-1].item() == pytest.approx(1.0, abs=1e-6)


@pytest.mark.parametrize("s", [-1.0, 0.0, 1.75])
def test_sway_warp_is_monotonic(s):
    # 実装のクランプ範囲 [-1, 1.75] 内で warp は単調非減少（ソルバー t グリッドの順序を壊さない）
    t_span = torch.linspace(0, 1, 101)
    warp = t_span + s * (torch.cos(torch.pi / 2 * t_span) - 1 + t_span)
    assert torch.all(torch.diff(warp) >= -1e-6)


def test_sway_forward_runs_and_is_deterministic(tiny_cfm):
    # sway_coef が forward に配線され、euler で有限・正しい shape・決定論（temperature=0 で z=0）
    dec = tiny_cfm()
    mu = torch.randn(1, N_MELS, 16)
    mask = torch.ones(1, 1, 16)
    c = torch.randn(1, 16)

    def run():
        return dec(mu, mask, n_timesteps=4, temperature=0.0, c=c, solver="euler", sway_coef=-1.0)

    out = run()
    assert out.shape == mu.shape
    assert torch.isfinite(out).all()
    assert torch.equal(run(), out)


def test_logit_normal_formula_in_open_unit_interval():
    # logit-normal は sigmoid(m + s*eps) で常に (0,1) 開区間（端点で補間が退化しない前提）
    torch.manual_seed(0)
    t = torch.sigmoid(0.0 + 1.0 * torch.randn(10000))
    assert torch.all(t > 0) and torch.all(t < 1)
    # m=0 は 0.5 対称 → 平均はおよそ 0.5
    assert t.mean().item() == pytest.approx(0.5, abs=0.02)
    assert math.isclose(torch.sigmoid(torch.tensor(0.0)).item(), 0.5)


def test_sway_changes_trajectory(tiny_cfm):
    # 実配線ガード: sway_coef=-1.0 は t グリッドを歪め、sway なし(None)と異なる出力になる
    # （warp を no-op 化/削除すると両者が一致してこのテストが落ちる）。temperature=0 で z=0 決定論
    dec = tiny_cfm()
    mu = torch.randn(1, N_MELS, 16)
    mask = torch.ones(1, 1, 16)
    c = torch.randn(1, 16)
    kw = {"n_timesteps": 4, "temperature": 0.0, "c": c, "solver": "euler"}
    out_none = dec(mu, mask, sway_coef=None, **kw)
    out_sway = dec(mu, mask, sway_coef=-1.0, **kw)
    assert not torch.allclose(out_none, out_sway)


def test_logit_normal_sampling_uses_sigmoid_of_m_plus_s_eps(tiny_cfm, monkeypatch):
    # 実配線ガード: compute_loss の logit_normal 分岐が t=sigmoid(m + s*eps) を実際に使うことを検証。
    # torch.sigmoid をスパイして (b,1,1) 形状の t サンプリング呼び出しの引数平均が m に一致するか見る
    # （sigmoid 削除 → 呼び出し消失、m<->s 入替 → 引数平均がズレて検知）
    b = 64
    dec = tiny_cfm(timestep_sampling="logit_normal", logit_normal_m=10.0, logit_normal_s=1.0)
    x1 = torch.randn(b, N_MELS, 12)
    mask = torch.ones(b, 1, 12)
    mu = torch.randn(b, N_MELS, 12)
    c = torch.randn(b, 16)

    calls = []
    real_sigmoid = torch.sigmoid

    def spy(inp):
        calls.append(inp)
        return real_sigmoid(inp)

    monkeypatch.setattr(torch, "sigmoid", spy)
    torch.manual_seed(0)
    dec.compute_loss(x1, mask, mu, c)

    t_calls = [inp for inp in calls if tuple(inp.shape) == (b, 1, 1)]
    assert t_calls, "logit_normal の t=sigmoid(...) 呼び出しが無い（sigmoid が外された可能性）"
    # arg = m + s*eps, eps~N(0,1) を b 個平均 → ほぼ m（=10）
    assert t_calls[0].float().mean().item() == pytest.approx(10.0, abs=0.5)


def _cfg_inputs():
    x = torch.randn(1, N_MELS, 20)
    mask = torch.ones(1, 1, 20)
    mu = torch.randn(1, N_MELS, 20)
    c = torch.randn(1, 16)
    return x, mask, mu, c


def _cfg_kwargs(cfg_strength=3.0, cfg_rescale=0.0, cfg_interval=None):
    return {
        "fake_speaker": torch.zeros(1, 16),
        "fake_content": torch.zeros(1, N_MELS, 1),
        "cfg_strength": cfg_strength,
        "cfg_rescale": cfg_rescale,
        "cfg_interval": cfg_interval,
        "slg_scale": 0.0,
        "slg_layers": (1,),
        "slg_t_range": (0.0, 0.5),
    }


def test_cfg_guidance_blends_uncond_and_cond(tiny_cfm):
    # Phase 1 の CFG 本体（推奨既定 cfg=3）: output = uncond + cfg*(cond-uncond)。cfg=1.0 の no-op ではなく
    # 誘導が効く経路を検証する（eval + p_dropout=0 で estimator は決定論なので手計算と一致）
    dec = tiny_cfm()
    t = torch.tensor(0.5)
    x, mask, mu, c = _cfg_inputs()
    kw = _cfg_kwargs(cfg_strength=3.0)
    cond = dec.estimator(t, x, mask, mu, c)
    uncond = dec.estimator(t, x, mask, kw["fake_content"].repeat(1, 1, x.size(-1)), kw["fake_speaker"].repeat(1, 1))
    out = dec.cfg_wrapper(t, x, mask, mu, c, kw)
    assert torch.allclose(out, uncond + 3.0 * (cond - uncond), atol=1e-5)
    assert not torch.allclose(out, cond)  # 誘導が実際に効いている


def test_cfg_rescale_matches_cond_std(tiny_cfm):
    # cfg_rescale=1.0 → output = blend * (std_cond/std_blend) なので出力 std が cond の std に一致（飽和抑制）
    dec = tiny_cfm()
    t = torch.tensor(0.5)
    x, mask, mu, c = _cfg_inputs()
    cond = dec.estimator(t, x, mask, mu, c)
    out = dec.cfg_wrapper(t, x, mask, mu, c, _cfg_kwargs(cfg_strength=3.0, cfg_rescale=1.0))
    assert torch.allclose(out.std(dim=(1, 2)), cond.std(dim=(1, 2)), atol=1e-4)


def test_cfg_interval_gates_guidance(tiny_cfm):
    # cfg_interval=(0.6,0.9): 区間外の t では誘導を省略して cond のまま、区間内では誘導が効く
    dec = tiny_cfm()
    x, mask, mu, c = _cfg_inputs()
    kw = _cfg_kwargs(cfg_strength=3.0, cfg_interval=(0.6, 0.9))

    t_out = torch.tensor(0.5)
    cond_out = dec.estimator(t_out, x, mask, mu, c)
    assert torch.equal(dec.cfg_wrapper(t_out, x, mask, mu, c, kw), cond_out)  # 区間外 = cond

    t_in = torch.tensor(0.7)
    cond_in = dec.estimator(t_in, x, mask, mu, c)
    assert not torch.allclose(dec.cfg_wrapper(t_in, x, mask, mu, c, kw), cond_in)  # 区間内 = 誘導
