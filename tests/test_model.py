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

    model = StableTTS(len(symbols), MelConfig().n_mels, **asdict(ModelConfig()))
    assert sum(p.numel() for p in model.parameters()) == 31_644_545
    assert tuple(model.encoder.emb.weight.shape) == (401, 256)
