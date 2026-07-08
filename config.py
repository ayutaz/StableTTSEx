from dataclasses import dataclass


@dataclass
class MelConfig:
    sample_rate: int = 44100
    n_fft: int = 2048
    win_length: int = 2048
    hop_length: int = 512
    f_min: float = 0.0
    f_max: float = None
    pad: int = 0
    n_mels: int = 128
    center: bool = False
    pad_mode: str = "reflect"
    mel_scale: str = "slaney"

    def __post_init__(self):
        if self.pad == 0:
            self.pad = (self.n_fft - self.hop_length) // 2


@dataclass
class ModelConfig:
    hidden_channels: int = 256
    filter_channels: int = 1024
    n_heads: int = 4
    n_enc_layers: int = 3
    n_dec_layers: int = 6
    kernel_size: int = 3
    p_dropout: float = 0.1
    gin_channels: int = 256
    # Phase 3 MRTE: 参照 mel 系列への cross-attention（デコーダに cross-attn を追加）。既定 False で現行と
    # 完全一致（param 数 31,644,545 不変）。True で state_dict にキーが増える（=既存チェックポイントとは
    # strict=False 部分ロード + zero-init ゲートで初期挙動一致 → 継続学習）。ModelConfig 側に置くのは
    # チェックポイント構造を変えるアーキ設定だから（TLA-SA は state_dict 不変なので TrainConfig）。
    # Phase 3 はクローズ（MRTE は spk_cos −0.007 で不採用、docs/phase3-plan.md §13）。False に戻して
    # 既存チェックポイント（tsukuyomi_ft200.pt 等）が api.py/webui.py で strict ロードできる状態を保つ。
    # MRTE をスクラッチ学習等で再評価する際に True にする
    use_mrte: bool = False


@dataclass
class TrainConfig:
    train_dataset_path: str = "filelists/filelist.json"
    test_dataset_path: str = "filelists/filelist.json"  # not used
    # R2 ランは R1(378h・batch 32) と学習ダイナミクスを揃え、既存 japanese-378h との 3-way 比較を清潔に
    # するため 32 に据える。bf16 でメモリに余裕はあるが、増やすと総ステップ/warmup 比/EMA 更新回数が変わる
    batch_size: int = 32
    learning_rate: float = 1e-4
    num_epochs: int = 15
    # Phase 3 第二弾 MRTE ラン。専用ディレクトリへ隔離する。初期値は **japanese-378h の checkpoint_14 を
    # このディレクトリ直下に checkpoint_0.pt として置く**（continue_training が model_save_path 直下から初期値を
    # 探し、strict=False 部分ロードで MRTE cross-attn を zero-init 立ち上げる）
    model_save_path: str = "./checkpoints/vast_run4"
    log_dir: str = "./runs"
    log_interval: int = 16
    save_interval: int = 1
    warmup_steps: int = 200
    # Phase 2 施策5: 学習時 timestep サンプリング。"cosine" = 既存 CosyVoice スケジューラ（ビット不変）、
    # "logit_normal" = SD3 式 logit-normal(m, s)（中間 t 重点）。Phase 3 TLA-SA ラン: cosine
    # （logit_normal は Phase 2 で不採用。baseline japanese-378h=cosine と揃え TLA-SA の効果だけを切り分ける）
    timestep_sampling: str = "cosine"
    logit_normal_m: float = 0.0
    logit_normal_s: float = 1.0
    # Phase 2 施策6: EMA 重み。use_ema=False で既存挙動。decay は warmup 付き上限（utils/ema.py 参照）。
    # Phase 3 TLA-SA ラン: False（EMA の微改善を混ぜず TLA-SA 単独の効果を baseline と切り分ける）
    use_ema: bool = False
    ema_decay: float = 0.9995
    ema_warmup: int = 10
    # Tier 1 学習最適化（GPU スループット向上）。いずれも数値精度は変わるが、チェックポイント形式・
    # n_vocab・パラメータ数・推論経路には影響しない。R2 はゼロからの再事前学習なので導入の最適タイミング。
    # use_amp: bf16 自動混合精度（autocast）。False で従来の純 FP32。bf16 は GradScaler 不要（fp16 と違い縮尺不要）
    use_amp: bool = True
    # grad_clip: 勾配ノルムのクリップ上限。0 以下で無効。bf16 + logit_normal は勾配分散が変わるため既定で有効化
    grad_clip: float = 1.0
    # use_fused_optimizer: fused AdamW（CUDA 専用でカーネル起動を融合）。CPU 実行時は自動で無効化される
    use_fused_optimizer: bool = True
    # Tier 2 学習最適化: DataLoader / DDP / compile / GPU MAS。
    # num_workers/prefetch_factor: .pt mel ロードは I/O バウンド。GPU が速いほど worker と先読みを増やす
    num_workers: int = 8
    prefetch_factor: int = 4
    # use_compile: torch.compile で decoder.estimator（DiT 本体）をコンパイル。バケットで系列長が変わるため
    # dynamic=True で再コンパイルを抑える。GPU 依存で効果が変わるため既定オフ（A/B して有効化）
    use_compile: bool = False
    # use_gpu_mas: MAS を GPU ネイティブ実装にして毎ステップの GPU→CPU 同期を除去する。numba 版と
    # ビット同一のアラインメントを返す（tests/test_mas.py で担保）。速度は GPU/系列長依存のため既定オフ
    use_gpu_mas: bool = False
    # Phase 3 第一弾: TLA-SA（補助話者整列損失、arXiv:2511.09995）。use_tla_sa=False で現行とビット一致
    # （TLASAHead は train.py 側の独立モジュールで、StableTTS の state_dict には一切足さない）。
    # 教師 SV エンコーダは評価の ECAPA とは別系統にする（テストに教えるバイアス回避）。
    # 埋め込みは precompute_spk_emb.py で事前計算し filelist に spk_emb_path を追加しておく。
    # Phase 3 第二弾 MRTE ランでは off（MRTE 単独の効果を切り分ける。TLA-SA は WavLM 教師で不発）
    use_tla_sa: bool = False
    tla_sa_lambda: float = 0.5  # L = L_CFM(等) + λ·L_TLA-SA。学習初期に sa/diff 比を見て再調整する
    tla_sa_alpha: float = 0.01  # 層重み w のエントロピー正則係数
    tla_sa_teacher: str = "wavlm_sv"  # スモークは vendor 不要の wavlm_sv(512次元)。本命は campplus(192次元, Apache-2.0)
    tla_sa_teacher_dim: int = 512  # teacher と必ず整合（campplus=192 / wavlm_sv=512）
    tla_sa_uniform_weight: bool = False  # True で層重み w_i=1/N 固定（timestep MLP・entropy 正則を無効化）


@dataclass
class VocosConfig:
    input_channels: int = 128
    dim: int = 512
    intermediate_dim: int = 1536
    num_layers: int = 8
