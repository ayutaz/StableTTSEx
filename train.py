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
    ).to(rank)

    # パラメータ数を計算
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # rank 0のプロセスでのみパラメータ数を表示
    if rank == 0:
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")

    model = DDP(model, device_ids=[rank])

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
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn,
        persistent_workers=True,
    )

    if rank == 0:
        writer = SummaryWriter(train_config.log_dir)

    optimizer = optim.AdamW(model.parameters(), lr=train_config.learning_rate)
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
            dur_loss, diff_loss, prior_loss, _ = model(x, x_lengths, y, y_lengths, z, z_lengths)
            loss = dur_loss + diff_loss + prior_loss
            loss.backward()
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
