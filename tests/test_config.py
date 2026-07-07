"""config.py の不変条件。MelConfig/ModelConfig の値は既存チェックポイント互換の前提であり、
壊れると静かに非互換になる。TrainConfig の Phase 2 既定は「既存挙動をビット維持」の前提。
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


def test_train_config_phase2_defaults_preserve_behavior():
    t = TrainConfig()
    # 既定は既存挙動（cosine スケジューラ・EMA なし）をビット維持する
    assert t.timestep_sampling == "cosine"
    assert (t.logit_normal_m, t.logit_normal_s) == (0.0, 1.0)
    assert t.use_ema is False
    assert (t.ema_decay, t.ema_warmup) == (0.9995, 10)
