import os

# 既に設定されていれば尊重する（未設定時のみ既定で GPU 0,1 を使う。使用GPU数は device_count() で決まる）
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0,1")

from dataclasses import asdict

import torch
import torch.distributed as dist
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from config import MelConfig, ModelConfig, TrainConfig
from datas.dataset import StableDataset, collate_fn
from datas.sampler import DistributedBucketSampler
from models.model import StableTTS
from text import symbols
from utils.ema import EMA
from utils.load import continue_training
from utils.scheduler import get_cosine_schedule_with_warmup

torch.backends.cudnn.benchmark = True
# Tier 1 最適化: Ampere 以降の Tensor Core を使うため matmul の TF32 を許可する。
# PyTorch は matmul の TF32 が既定 False（純 FP32）なので、明示有効化で純 FP32 比 数倍。精度劣化は無視できる範囲。
torch.set_float32_matmul_precision("high")
torch.backends.cudnn.allow_tf32 = True


def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12345"
    dist.init_process_group("gloo" if os.name == "nt" else "nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def _init_config(model_config: ModelConfig, mel_config: MelConfig, train_config: TrainConfig):

    if not os.path.exists(train_config.model_save_path):
        print(f"Creating {train_config.model_save_path}")
        os.makedirs(train_config.model_save_path, exist_ok=True)


def train(rank, world_size):
    setup(rank, world_size)
    torch.cuda.set_device(rank)

    model_config = ModelConfig()
    mel_config = MelConfig()
    train_config = TrainConfig()

    _init_config(model_config, mel_config, train_config)

    model = StableTTS(
        len(symbols),
        mel_config.n_mels,
        **asdict(model_config),
        timestep_sampling=train_config.timestep_sampling,
        logit_normal_m=train_config.logit_normal_m,
        logit_normal_s=train_config.logit_normal_s,
        use_gpu_mas=train_config.use_gpu_mas,
    ).to(rank)

    # パラメータ数を計算
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # rank 0のプロセスでのみパラメータ数を表示
    if rank == 0:
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")

    # Tier 2 最適化: DiT 本体（estimator）を torch.compile。MAS 等の graph break を避けるためサブモジュール
    # 単位でコンパイルする。系列長がバケットで変わるので dynamic=True で再コンパイル爆発を抑える。
    # in-place の nn.Module.compile を使う（module = torch.compile(...) の再代入は state_dict に _orig_mod.
    # 接頭辞を付けチェックポイント互換を壊すため。in-place ならキーが不変）
    if train_config.use_compile:
        model.decoder.estimator.compile(dynamic=True)

    # Tier 2 最適化: DDP の通信オーバーヘッド削減。
    # gradient_as_bucket_view=True で勾配バッファのコピーを省く。使用パラメータが毎ステップ一定なので
    # static_graph=True が安全（cfg dropout はマスク乗算で両経路とも常時計算される）。
    # LayerNorm(affine 無し)のみで同期対象バッファが無いため broadcast_buffers=False で毎ステップの同期を省く
    model = DDP(
        model,
        device_ids=[rank],
        gradient_as_bucket_view=True,
        static_graph=True,
        broadcast_buffers=False,
    )

    train_dataset = StableDataset(train_config.train_dataset_path, mel_config.hop_length)
    # 上限はデータセットの最長 mel 長を包含させる（境界外のサンプルは黙って捨てられるため。moe-speech は最長 ~1291）
    train_sampler = DistributedBucketSampler(
        train_dataset,
        train_config.batch_size,
        [32, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300],
        num_replicas=world_size,
        rank=rank,
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        num_workers=train_config.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
        persistent_workers=True,
        prefetch_factor=train_config.prefetch_factor,
    )

    if rank == 0:
        writer = SummaryWriter(train_config.log_dir)

    # fused AdamW は CUDA かつ全パラメータが CUDA 上にある場合のみ有効（CPU では使えないためフォールバック）
    use_fused = train_config.use_fused_optimizer and torch.cuda.is_available()
    optimizer = optim.AdamW(model.parameters(), lr=train_config.learning_rate, fused=use_fused)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(train_config.warmup_steps),
        num_training_steps=train_config.num_epochs * len(train_dataloader),
    )

    # load latest checkpoints if possible
    current_epoch = continue_training(train_config.model_save_path, model, optimizer)

    # Phase 2 施策6: EMA は rank 0 でのみ維持する（DDP が全 rank の重みを同期するため代表となる）。
    # continue_training が生重みをロードした後に生成するので、シャドウは直近チェックポイント基準で初期化される。
    ema = None
    if rank == 0 and train_config.use_ema:
        ema = EMA(model.module, decay=train_config.ema_decay, warmup=train_config.ema_warmup)
        ema_resume_path = os.path.join(train_config.model_save_path, f"ema_state_{current_epoch - 1}.pt")
        if current_epoch > 0 and os.path.exists(ema_resume_path):
            ema.load_ema_training_state(torch.load(ema_resume_path, map_location="cpu"))
            print(f"resume EMA from {current_epoch - 1} epoch")
        elif current_epoch > 0:
            print(f"warning: EMA state for epoch {current_epoch - 1} not found; initializing EMA from current weights")

    model.train()
    for epoch in range(current_epoch, train_config.num_epochs):  # loop over the train_dataset multiple times
        train_dataloader.batch_sampler.set_epoch(epoch)
        if rank == 0:
            dataloader = tqdm(train_dataloader)
        else:
            dataloader = train_dataloader

        for batch_idx, datas in enumerate(dataloader):
            datas = [data.to(rank, non_blocking=True) for data in datas]
            x, x_lengths, y, y_lengths, z, z_lengths = datas
            optimizer.zero_grad()
            # Tier 1 最適化: bf16 autocast。matmul/conv/attention を bf16 で走らせ ~1.5-2x・メモリ半減。
            # bf16 は fp16 と違いダイナミックレンジが広く GradScaler 不要。損失関数（mse_loss 等）と MAS は
            # autocast の fp32 ポリシー / model 側の明示 fp32 で保護される
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=train_config.use_amp):
                dur_loss, diff_loss, prior_loss, _ = model(x, x_lengths, y, y_lengths, z, z_lengths)
            loss = dur_loss + diff_loss + prior_loss
            loss.backward()
            if train_config.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.grad_clip)
            optimizer.step()
            scheduler.step()
            if ema is not None:
                ema.update(model.module)

            if rank == 0 and batch_idx % train_config.log_interval == 0:
                steps = epoch * len(dataloader) + batch_idx
                writer.add_scalar("training/diff_loss", diff_loss.item(), steps)
                writer.add_scalar("training/dur_loss", dur_loss.item(), steps)
                writer.add_scalar("training/prior_loss", prior_loss.item(), steps)
                writer.add_scalar("learning_rate/learning_rate", scheduler.get_last_lr()[0], steps)

        if rank == 0 and epoch % train_config.save_interval == 0:
            torch.save(model.module.state_dict(), os.path.join(train_config.model_save_path, f"checkpoint_{epoch}.pt"))
            torch.save(optimizer.state_dict(), os.path.join(train_config.model_save_path, f"optimizer_{epoch}.pt"))
            if ema is not None:
                # 推論用の EMA 重み（module.state_dict 互換で api.py にそのままロード可）とレジューム用状態。
                # ファイル名は "checkpoint" 始まりにしない（continue_training の走査が生重みと衝突するため）
                torch.save(ema.state_dict(), os.path.join(train_config.model_save_path, f"ema_checkpoint_{epoch}.pt"))
                torch.save(
                    ema.ema_training_state(), os.path.join(train_config.model_save_path, f"ema_state_{epoch}.pt")
                )
        print(f"Rank {rank}, Epoch {epoch}, Loss {loss.item()}")

    cleanup()


torch.set_num_threads(1)
torch.set_num_interop_threads(1)

if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    torch.multiprocessing.spawn(train, args=(world_size,), nprocs=world_size)
