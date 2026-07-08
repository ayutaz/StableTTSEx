"""StableTTS のテンソル契約。推論経路（synthesise）の shape、学習経路（forward: MAS + 3損失）の
有限性、Phase 1 の「既定 = 機能オフ」ビット不変性、および full-config でのパラメータ数
（既存チェックポイント互換の回帰ガード）を検証する。値は版依存なので shape/有限/等価に寄せる。
"""

import torch

from tests.conftest import TINY_MODEL_KWARGS, TINY_N_MELS, TINY_N_VOCAB


def _text_inputs(length=5):
    x = torch.arange(1, length + 1, dtype=torch.long).unsqueeze(0)  # 1..length（< n_vocab）
    x_lengths = torch.tensor([length], dtype=torch.long)
    return x, x_lengths


def test_synthesise_output_shapes(tiny_stabletts):
    model = tiny_stabletts()
    x, x_lengths = _text_inputs()
    y_ref = torch.randn(1, TINY_N_MELS, 16)
    torch.manual_seed(0)
    out = model.synthesise(x, x_lengths, n_timesteps=4, temperature=0.0, y=y_ref, solver="euler")
    assert set(out.keys()) == {"encoder_outputs", "decoder_outputs", "attn"}
    enc, dec = out["encoder_outputs"], out["decoder_outputs"]
    assert enc.shape == dec.shape
    assert enc.shape[:2] == (1, TINY_N_MELS)
    assert enc.shape[2] > 0
    assert torch.isfinite(dec).all()
    # attn の最終軸は生成 mel 長に一致
    assert out["attn"].shape[-1] == dec.shape[-1]


def test_synthesise_defaults_equal_explicit_off(tiny_stabletts):
    # Phase 1 ビット不変: 既定（sway/cfg/slg 未指定）と明示オフ値が同一出力
    model = tiny_stabletts()
    x, x_lengths = _text_inputs()
    y_ref = torch.randn(1, TINY_N_MELS, 16)

    torch.manual_seed(7)
    default_out = model.synthesise(x, x_lengths, n_timesteps=4, temperature=0.0, y=y_ref, solver="euler")

    torch.manual_seed(7)
    explicit_out = model.synthesise(
        x,
        x_lengths,
        n_timesteps=4,
        temperature=0.0,
        y=y_ref,
        solver="euler",
        cfg=1.0,
        sway_coef=None,
        cfg_rescale=0.0,
        cfg_interval=None,
        slg_scale=0.0,
    )
    assert torch.equal(default_out["decoder_outputs"], explicit_out["decoder_outputs"])


def test_synthesise_cfg_guidance_changes_output(tiny_stabletts):
    # webui/api の推奨・既定推論パス（cfg=3 で cfg_wrapper 経由の誘導 + sway）を実行し、
    # cfg=1.0（誘導なし）と異なる出力になることを確認（CFG/sway の配線ガード）
    model = tiny_stabletts()
    x, x_lengths = _text_inputs()
    y_ref = torch.randn(1, TINY_N_MELS, 16)

    torch.manual_seed(3)
    plain = model.synthesise(x, x_lengths, n_timesteps=4, temperature=0.0, y=y_ref, solver="euler", cfg=1.0)
    torch.manual_seed(3)
    guided = model.synthesise(
        x, x_lengths, n_timesteps=4, temperature=0.0, y=y_ref, solver="euler", cfg=3.0, cfg_rescale=0.7, sway_coef=-1.0
    )
    assert torch.isfinite(guided["decoder_outputs"]).all()
    assert guided["decoder_outputs"].shape == plain["decoder_outputs"].shape
    assert not torch.allclose(plain["decoder_outputs"], guided["decoder_outputs"])


def test_synthesise_length_scale_extends_output(tiny_stabletts):
    # length_scale は継続長に比例し生成 mel 長を伸ばす（webui 露出のユーザーノブ）
    model = tiny_stabletts()
    x, x_lengths = _text_inputs()
    y_ref = torch.randn(1, TINY_N_MELS, 16)

    torch.manual_seed(0)
    short = model.synthesise(x, x_lengths, n_timesteps=4, temperature=0.0, y=y_ref, solver="euler", length_scale=1.0)
    torch.manual_seed(0)
    long = model.synthesise(x, x_lengths, n_timesteps=4, temperature=0.0, y=y_ref, solver="euler", length_scale=2.0)
    assert long["decoder_outputs"].shape[-1] > short["decoder_outputs"].shape[-1]


def test_forward_returns_three_finite_losses(tiny_stabletts):
    model = tiny_stabletts()
    x, x_lengths = _text_inputs()
    y = torch.randn(1, TINY_N_MELS, 48)
    y_lengths = torch.tensor([48], dtype=torch.long)
    z = torch.randn(1, TINY_N_MELS, 16)
    z_lengths = torch.tensor([16], dtype=torch.long)
    torch.manual_seed(0)
    dur_loss, diff_loss, prior_loss, attn = model.forward(x, x_lengths, y, y_lengths, z, z_lengths)
    for loss in (dur_loss, diff_loss, prior_loss):
        assert loss.ndim == 0 and torch.isfinite(loss)
    # MAS アラインメント attn: (batch, text_len, mel_len)
    assert attn.shape == (1, x.shape[1], y.shape[-1])


def test_forward_bf16_autocast_path_runs(tiny_stabletts):
    # Tier 1 最適化: bf16 autocast 下でも学習経路が有限損失を返す（MAS の neg_cent は fp32 保護される）。
    # conftest が CPU 固定するため device_type="cpu" の bf16 autocast で検証する
    model = tiny_stabletts(timestep_sampling="logit_normal")
    x, x_lengths = _text_inputs()
    y = torch.randn(1, TINY_N_MELS, 48)
    y_lengths = torch.tensor([48], dtype=torch.long)
    z = torch.randn(1, TINY_N_MELS, 16)
    z_lengths = torch.tensor([16], dtype=torch.long)
    torch.manual_seed(0)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        dur_loss, diff_loss, prior_loss, attn = model.forward(x, x_lengths, y, y_lengths, z, z_lengths)
    for loss in (dur_loss, diff_loss, prior_loss):
        assert loss.ndim == 0 and torch.isfinite(loss)
    # MAS は fp32 で計算されるため attn は bf16 でも従来と同じ shape/整数境界を保つ
    assert attn.shape == (1, x.shape[1], y.shape[-1])


def test_forward_gpu_mas_matches_numba_alignment(tiny_stabletts):
    # Tier 2 最適化: use_gpu_mas=True の GPU MAS が numba 版と同一の教師アラインメントを返す。
    # neg_cent はモデル重み + 入力から決定的に決まる（MAS は no_grad・RNG 非依存）ため、両者の attn は厳密一致
    x, x_lengths = _text_inputs()
    y = torch.randn(1, TINY_N_MELS, 48)
    y_lengths = torch.tensor([48], dtype=torch.long)
    z = torch.randn(1, TINY_N_MELS, 16)
    z_lengths = torch.tensor([16], dtype=torch.long)

    model_numba = tiny_stabletts(use_gpu_mas=False)
    model_gpu = tiny_stabletts(use_gpu_mas=True)  # 同一 seed で重みは一致
    torch.manual_seed(0)
    attn_ref = model_numba.forward(x, x_lengths, y, y_lengths, z, z_lengths)[3]
    torch.manual_seed(0)
    attn_got = model_gpu.forward(x, x_lengths, y, y_lengths, z, z_lengths)[3]
    assert torch.equal(attn_ref, attn_got)


def test_forward_logit_normal_training_path_runs(tiny_stabletts):
    # Phase 2 施策5: logit_normal でも学習経路が有限損失を返す
    model = tiny_stabletts(timestep_sampling="logit_normal")
    x, x_lengths = _text_inputs()
    y = torch.randn(1, TINY_N_MELS, 48)
    y_lengths = torch.tensor([48], dtype=torch.long)
    z = torch.randn(1, TINY_N_MELS, 16)
    z_lengths = torch.tensor([16], dtype=torch.long)
    torch.manual_seed(0)
    losses = model.forward(x, x_lengths, y, y_lengths, z, z_lengths)[:3]
    assert all(torch.isfinite(loss) for loss in losses)


def test_compiled_estimator_preserves_state_dict_keys():
    # Tier 2 最適化: use_compile の in-place compile（nn.Module.compile）は state_dict のキーを変えない。
    # module = torch.compile(...) の再代入だと _orig_mod. 接頭辞が付き既存チェックポイントと非互換になるため、
    # in-place であることをチェックポイント互換の回帰ガードとして固定する（forward は呼ばず compile はしない）
    from models.model import StableTTS

    model = StableTTS(TINY_N_VOCAB, TINY_N_MELS, **TINY_MODEL_KWARGS)
    before = set(model.state_dict().keys())
    model.decoder.estimator.compile(dynamic=True)
    after = set(model.state_dict().keys())
    assert before == after
    # 生モデルへ strict=True で相互ロード可能
    fresh = StableTTS(TINY_N_VOCAB, TINY_N_MELS, **TINY_MODEL_KWARGS)
    fresh.load_state_dict(model.state_dict(), strict=True)


def test_mrte_full_config_param_count():
    # Phase 3 MRTE: use_mrte=True の full config パラメータ数（6ブロック× cross_attn conv q/k/v/o + cross_gate
    # + fake_ref）。既存チェックポイントとの非互換を意図的に厳密固定する不変条件
    from dataclasses import asdict

    from config import MelConfig, ModelConfig
    from models.model import StableTTS
    from text import symbols

    model = StableTTS(len(symbols), MelConfig().n_mels, **{**asdict(ModelConfig()), "use_mrte": True})
    assert sum(p.numel() for p in model.parameters()) == 33_225_345


def test_mrte_zero_gate_bit_identical(tiny_stabletts):
    # zero-init ゲートにより、MRTE モデルに baseline 重みを strict=False 部分ロードした直後の synthesise 出力は
    # cross-attn を持たない baseline と byte-identical（追加項 = cross_gate(=0) * cross_attn = 0）
    base = tiny_stabletts(use_mrte=False)
    mrte = tiny_stabletts(use_mrte=True)
    missing, unexpected = mrte.load_state_dict(base.state_dict(), strict=False)
    assert unexpected == []  # baseline は MRTE の真部分集合
    assert all(("cross" in k or "fake_ref" in k) for k in missing)  # 欠落は MRTE キーのみ

    x, x_lengths = _text_inputs()
    y_ref = torch.randn(1, TINY_N_MELS, 16)
    torch.manual_seed(5)
    base_out = base.synthesise(x, x_lengths, n_timesteps=4, temperature=0.0, y=y_ref, solver="euler")
    torch.manual_seed(5)
    mrte_out = mrte.synthesise(x, x_lengths, n_timesteps=4, temperature=0.0, y=y_ref, solver="euler")
    assert torch.equal(base_out["decoder_outputs"], mrte_out["decoder_outputs"])


def test_mrte_zero_gate_bit_identical_cfg_path(tiny_stabletts):
    # zero-gate の byte-identity を webui/api 既定の CFG 経路（cfg=3 → cfg_wrapper の fake_ref uncond）でも固定する。
    # cfg=1.0 の直行経路は cfg_wrapper を通らないため、fake_ref.expand / uncond の ref_seq null 化配線を別途ガードする
    base = tiny_stabletts(use_mrte=False)
    mrte = tiny_stabletts(use_mrte=True)
    mrte.load_state_dict(base.state_dict(), strict=False)  # cross_gate=0 のまま

    x, x_lengths = _text_inputs()
    y_ref = torch.randn(1, TINY_N_MELS, 16)
    kw = dict(n_timesteps=4, temperature=0.0, y=y_ref, solver="euler", cfg=3.0, cfg_rescale=0.7, sway_coef=-1.0)
    torch.manual_seed(8)
    base_out = base.synthesise(x, x_lengths, **kw)
    torch.manual_seed(8)
    mrte_out = mrte.synthesise(x, x_lengths, **kw)
    assert torch.equal(base_out["decoder_outputs"], mrte_out["decoder_outputs"])


def test_mrte_state_dict_superset(tiny_stabletts):
    base = tiny_stabletts(use_mrte=False)
    mrte = tiny_stabletts(use_mrte=True)
    bk, mk = set(base.state_dict().keys()), set(mrte.state_dict().keys())
    assert bk < mk  # 真部分集合
    assert all(("cross" in k or "fake_ref" in k) for k in (mk - bk))  # 差分は MRTE キーのみ


def test_mrte_gate_perturb_changes_output(tiny_stabletts):
    # cross_gate を非零にすると出力が変わる（cross-attn 配線が生きている保証）
    mrte = tiny_stabletts(use_mrte=True)
    x, x_lengths = _text_inputs()
    y_ref = torch.randn(1, TINY_N_MELS, 16)
    torch.manual_seed(6)
    out0 = mrte.synthesise(x, x_lengths, n_timesteps=4, temperature=0.0, y=y_ref, solver="euler")
    with torch.no_grad():
        for blk in mrte.decoder.estimator.blocks:
            blk.block.cross_gate.fill_(0.5)
    torch.manual_seed(6)
    out1 = mrte.synthesise(x, x_lengths, n_timesteps=4, temperature=0.0, y=y_ref, solver="euler")
    assert not torch.allclose(out0["decoder_outputs"], out1["decoder_outputs"])


def test_mrte_forward_training_finite(tiny_stabletts):
    # use_mrte=True の学習経路（参照系列 drop 込み）が有限3損失を返す
    model = tiny_stabletts(use_mrte=True)
    x, x_lengths = _text_inputs()
    y = torch.randn(1, TINY_N_MELS, 48)
    y_lengths = torch.tensor([48], dtype=torch.long)
    z = torch.randn(1, TINY_N_MELS, 16)
    z_lengths = torch.tensor([16], dtype=torch.long)
    torch.manual_seed(0)
    losses = model.forward(x, x_lengths, y, y_lengths, z, z_lengths)[:3]
    assert all(torch.isfinite(loss) for loss in losses)


def test_default_kwargs_do_not_add_parameters():
    # Phase 1/2 の新引数は plain 属性で nn.Parameter/buffer を増やさない
    # → 既存チェックポイントが strict=True でロード可能（キー集合が不変）
    from models.model import StableTTS

    base = StableTTS(
        TINY_N_VOCAB,
        TINY_N_MELS,
        hidden_channels=16,
        filter_channels=32,
        n_heads=2,
        n_enc_layers=2,
        n_dec_layers=2,
        kernel_size=3,
        p_dropout=0.0,
        gin_channels=16,
    )
    withopts = StableTTS(
        TINY_N_VOCAB,
        TINY_N_MELS,
        hidden_channels=16,
        filter_channels=32,
        n_heads=2,
        n_enc_layers=2,
        n_dec_layers=2,
        kernel_size=3,
        p_dropout=0.0,
        gin_channels=16,
        timestep_sampling="logit_normal",
        logit_normal_m=0.3,
        logit_normal_s=0.8,
    )
    assert set(base.state_dict().keys()) == set(withopts.state_dict().keys())
    # 実際に相互ロード可能
    withopts.load_state_dict(base.state_dict(), strict=True)


def test_full_config_param_count_and_embedding():
    # チェックポイント互換の強いガード: アーキ定義から一意に決まるパラメータ数と埋め込み形状
    from dataclasses import asdict

    from config import MelConfig, ModelConfig
    from models.model import StableTTS
    from text import symbols

    # use_mrte=False の正準アーキ（MRTE 有り 33,225,345 は test_mrte_full_config_param_count で別途固定）。
    # ModelConfig の既定が MRTE ラン用に True でも、この互換ガードは常に正準値を検証する
    model = StableTTS(len(symbols), MelConfig().n_mels, **{**asdict(ModelConfig()), "use_mrte": False})
    assert sum(p.numel() for p in model.parameters()) == 31_644_545
    assert tuple(model.encoder.emb.weight.shape) == (401, 256)


def test_use_tla_sa_does_not_change_state_dict():
    # Phase 3 TLA-SA: use_tla_sa は plain 属性で submodule/Parameter を足さない
    # → state_dict キー集合・パラメータ数が不変（TLASAHead は train.py 側の独立モジュール）
    from models.model import StableTTS

    base = StableTTS(TINY_N_VOCAB, TINY_N_MELS, **TINY_MODEL_KWARGS)
    withtla = StableTTS(TINY_N_VOCAB, TINY_N_MELS, **TINY_MODEL_KWARGS, use_tla_sa=True)
    assert set(base.state_dict().keys()) == set(withtla.state_dict().keys())
    assert sum(p.numel() for p in base.parameters()) == sum(p.numel() for p in withtla.parameters())
    withtla.load_state_dict(base.state_dict(), strict=True)  # 相互ロード可


def test_forward_return_tla_shapes_and_noop_equivalence(tiny_stabletts):
    # return_tla=True で 5-tuple。tla_feats の hiddens は n_dec_layers 本・各 [B,hidden,T]・有限。
    # かつ同一 seed で diff_loss が非 tla 経路と一致（no-op 等価: return_tla は estimator を1回呼ぶだけで数値不変）
    model = tiny_stabletts(use_tla_sa=True)
    x, x_lengths = _text_inputs()
    y = torch.randn(1, TINY_N_MELS, 48)
    y_lengths = torch.tensor([48], dtype=torch.long)
    z = torch.randn(1, TINY_N_MELS, 16)
    z_lengths = torch.tensor([16], dtype=torch.long)

    torch.manual_seed(0)
    _, diff1, _, _ = model.forward(x, x_lengths, y, y_lengths, z, z_lengths)
    torch.manual_seed(0)
    out = model.forward(x, x_lengths, y, y_lengths, z, z_lengths, return_tla=True)
    assert len(out) == 5
    _, diff2, _, _, tla = out
    assert torch.equal(diff1, diff2)  # no-op 等価
    n_dec_layers = TINY_MODEL_KWARGS["n_dec_layers"]
    assert len(tla["hiddens"]) == n_dec_layers
    for h in tla["hiddens"]:
        assert h.shape[:2] == (1, TINY_MODEL_KWARGS["hidden_channels"])
        assert torch.isfinite(h).all()
    assert tla["valid"].shape == (1,)


def test_tla_sa_head_forward_scalar_finite():
    # TLASAHead: スカラ有限、uniform フォールバック、全 cfg ドロップ時の 0 割保護を検証
    from models.tla_sa import TLASAHead

    n_layers, d_hidden, d_teacher, b, t = 6, 16, 192, 3, 20
    hiddens = [torch.randn(b, d_hidden, t) for _ in range(n_layers)]
    ts = torch.rand(b)
    e_sa = torch.randn(b, d_teacher)
    ymask = torch.ones(b, 1, t)
    valid = torch.ones(b, dtype=torch.bool)

    head = TLASAHead(n_layers=n_layers, d_hidden=d_hidden, d_teacher=d_teacher)
    loss = head(hiddens, ts, e_sa, ymask, valid)
    assert loss.ndim == 0 and torch.isfinite(loss)

    head_u = TLASAHead(n_layers=n_layers, d_hidden=d_hidden, d_teacher=d_teacher, uniform=True)
    loss_u = head_u(hiddens, ts, e_sa, ymask, valid)
    assert loss_u.ndim == 0 and torch.isfinite(loss_u)

    # 全サンプル cfg ドロップ（valid 全 False）でも 0 割せず有限
    loss_zero = head(hiddens, ts, e_sa, ymask, torch.zeros(b, dtype=torch.bool))
    assert torch.isfinite(loss_zero)
