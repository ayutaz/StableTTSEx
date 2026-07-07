"""config.py の不変条件。MelConfig/ModelConfig の値は既存チェックポイント互換の前提であり、
壊れると静かに非互換になる。TrainConfig は「次に回す学習ランの設定」で、現在は Phase 3 第一弾
（cosine + TLA-SA 補助話者整列損失、EMA/logit_normal は不使用）を選択している。チェックポイント
互換のビット不変性の本質はモデル層（CFMDecoder/StableTTS の kwargs 既定 = cosine/off、推論は
asdict(ModelConfig) 経由で不変）にあり、test_model.py / test_flow_matching.py が担保する。
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


def test_train_config_phase3_tla_sa_recipe():
    t = TrainConfig()
    # 現在の学習設定は Phase 3 第一弾 = TLA-SA（cosine + EMA無 + 補助話者整列損失）。
    # baseline japanese-378h(cosine) と揃え、logit_normal(Phase 2 で不採用)は cosine に戻して切り分ける
    assert t.timestep_sampling == "cosine"
    assert t.use_ema is False
    # TLA-SA 有効。教師はスモーク用 wavlm_sv(512次元)。出力は vast_run3 に隔離（R1/R2 と対称）
    assert t.use_tla_sa is True
    assert (t.tla_sa_lambda, t.tla_sa_alpha) == (0.5, 0.01)
    assert t.tla_sa_teacher == "wavlm_sv"
    assert t.tla_sa_teacher_dim == 512
    assert t.tla_sa_uniform_weight is False
    assert t.model_save_path == "./checkpoints/vast_run3"
    # Tier 1/2 学習最適化は据え置き（数値精度のみ変わり、チェックポイント形式・n_vocab・param数・推論経路は不変）
    assert t.use_amp is True
    assert t.grad_clip == 1.0
    assert t.use_fused_optimizer is True
    assert (t.num_workers, t.prefetch_factor) == (8, 4)
    assert t.use_compile is False
    assert t.use_gpu_mas is False
