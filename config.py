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


@dataclass
class TrainConfig:
    train_dataset_path: str = "filelists/filelist.json"
    test_dataset_path: str = "filelists/filelist.json"  # not used
    # bf16 でアクティベーションメモリが約半減したぶん増やせる。OOM する場合は下げる（VRAM 依存の調整値）
    batch_size: int = 48
    learning_rate: float = 1e-4
    num_epochs: int = 15
    model_save_path: str = "./checkpoints"
    log_dir: str = "./runs"
    log_interval: int = 16
    save_interval: int = 1
    warmup_steps: int = 200
    # Phase 2 施策5: 学習時 timestep サンプリング。"cosine" = 既存 CosyVoice スケジューラ（ビット不変）、
    # "logit_normal" = SD3 式 logit-normal(m, s)（中間 t 重点）。R2 ラン: logit_normal
    # 注: このリポジトリの t 規約は SD3 と逆（t=0→noise, t=1→data）。m=0 は対称なので影響なし。
    # m≠0 で調整する際は SD3 の符号を反転させること
    timestep_sampling: str = "logit_normal"
    logit_normal_m: float = 0.0
    logit_normal_s: float = 1.0
    # Phase 2 施策6: EMA 重み。use_ema=False で既存挙動。decay は warmup 付き上限（utils/ema.py 参照）。R2 ラン: True
    use_ema: bool = True
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


@dataclass
class VocosConfig:
    input_channels: int = 128
    dim: int = 512
    intermediate_dim: int = 1536
    num_layers: int = 8
