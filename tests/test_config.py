"""config.py の不変条件。MelConfig/ModelConfig の値は既存チェックポイント互換の前提であり、
壊れると静かに非互換になる。TrainConfig は「次に回す学習ランの設定」で、現在は Phase 2 R2
（logit_normal + EMA）を選択している。チェックポイント互換のビット不変性の本質はモデル層
（CFMDecoder/StableTTS の kwargs 既定 = cosine/off、推論は asdict(ModelConfig) 経由で不変）
にあり、test_model.py / test_flow_matching.py が担保する。
"""

from config import MelConfig, ModelConfig, TrainConfig


def test_mel_config_defaults():
    m = MelConfig()
    assert (m.sample_rate, m.n_fft, m.hop_length, m.win_length, m.n_mels, m.mel_scale) == (
        44100,
        2048,
        512,
        2048,
        128,
        "slaney",
    )


def test_mel_config_pad_sentinel():
    # pad=0 はセンチネルで (n_fft - hop_length)//2 = 768 に導出される（明示的に 0 にはできない）
    assert MelConfig().pad == (2048 - 512) // 2 == 768
    assert MelConfig(pad=0).pad == 768
    assert MelConfig(pad=5).pad == 5


def test_model_config_architecture_constants():
    c = ModelConfig()
    assert (c.hidden_channels, c.filter_channels, c.n_heads) == (256, 1024, 4)
    assert (c.n_enc_layers, c.n_dec_layers) == (3, 6)
    assert (c.kernel_size, c.gin_channels, c.p_dropout) == (3, 256, 0.1)
    # use_lsc（デコーダの long skip）は n_dec_layers 偶数が前提
    assert c.n_dec_layers % 2 == 0


def test_train_config_phase2_r2_recipe():
    t = TrainConfig()
    # 現在の学習設定は Phase 2 R2: logit_normal timestep + EMA
    assert t.timestep_sampling in ("cosine", "logit_normal")
    assert t.timestep_sampling == "logit_normal"
    # m=0 は t 規約が SD3 と逆でも対称（中間 t 重点のみ効かせる）。s は標準
    assert (t.logit_normal_m, t.logit_normal_s) == (0.0, 1.0)
    assert t.use_ema is True
    assert (t.ema_decay, t.ema_warmup) == (0.9995, 10)
    # Tier 1 学習最適化: bf16 AMP + 勾配クリップ + fused AdamW。数値精度のみ変わり、チェックポイント形式・
    # n_vocab・パラメータ数・推論経路は不変（互換性はモデル層で担保）
    assert t.use_amp is True
    assert t.grad_clip == 1.0
    assert t.use_fused_optimizer is True
