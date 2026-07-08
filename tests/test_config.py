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


def test_phase3_closed_config_is_inference_safe():
    t = TrainConfig()
    m = ModelConfig()
    # Phase 3 はクローズ（Phase 2/3-1/3-2 いずれも話者類似性を改善できず不採用、docs/phase3-plan.md §13）。
    # 推論・webui が既存チェックポイントを strict ロードできるよう、アーキ変更フラグは全て off に戻してある
    assert m.use_mrte is False  # MRTE off = param 31,644,545・state_dict キー不変
    assert t.use_tla_sa is False  # TLA-SA off = state_dict 不変
    # 学習レシピは baseline 相当（cosine / EMA無 / logit_normal は Phase 2 で不採用）
    assert t.timestep_sampling == "cosine"
    assert t.use_ema is False
    # Tier 1/2 学習最適化は据え置き（数値精度のみ変わり、チェックポイント形式・n_vocab・推論経路は不変）
    assert t.use_amp is True
    assert t.grad_clip == 1.0
    assert t.use_fused_optimizer is True
    assert (t.num_workers, t.prefetch_factor) == (8, 4)
    assert t.use_compile is False
    assert t.use_gpu_mas is False
