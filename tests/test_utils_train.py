"""学習ユーティリティ（GPU/DDP 不要）: sequence_mask・LR スケジューラ・continue_training。
continue_training のテストは Phase 2 の EMA ファイル命名（ema_* は生重みレジュームに誤マッチしない）
の回帰ガードを兼ねる。
"""

import math
import types

import pytest
import torch
from torch import optim

from utils.load import continue_training
from utils.mask import sequence_mask
from utils.scheduler import get_cosine_schedule_with_warmup


def test_sequence_mask_golden():
    mask = sequence_mask(torch.tensor([2, 3]), 4)
    assert mask.tolist() == [[True, True, False, False], [True, True, True, False]]


def test_sequence_mask_infers_max_length():
    mask = sequence_mask(torch.tensor([1, 3]))
    assert mask.shape == (2, 3)
    assert mask.tolist() == [[True, False, False], [True, True, True]]


def test_cosine_schedule_warmup_then_decay():
    param = torch.nn.Parameter(torch.zeros(1))
    opt = optim.SGD([param], lr=1.0)
    warmup, total = 5, 20
    sched = get_cosine_schedule_with_warmup(opt, num_warmup_steps=warmup, num_training_steps=total)

    lrs = []
    for _ in range(total):
        lrs.append(opt.param_groups[0]["lr"])
        opt.step()
        sched.step()

    assert lrs[0] < 1e-9  # step0 は概ね 0
    assert lrs[warmup] == max(lrs)  # warmup 終了でピーク（= 初期 lr 1.0）
    assert lrs[warmup] == 1.0
    assert all(lrs[i] <= lrs[i + 1] + 1e-12 for i in range(warmup))  # warmup 中は非減少
    assert lrs[-1] < lrs[warmup]  # 以降は cosine 減衰
    # 減衰の形状を閉形式で固定（num_cycles=0.5 の half-cosine）。'2.0*' 係数欠落等の形状バグを検知
    progress = (total - 1 - warmup) / (total - warmup)
    expected_last = 0.5 * (1.0 + math.cos(math.pi * 0.5 * 2.0 * progress))
    assert lrs[-1] == pytest.approx(expected_last, abs=1e-4)


def _save_tiny_checkpoint(tmp_path, epoch, with_optimizer=True, with_ema=True):
    model = torch.nn.Linear(4, 4)
    torch.nn.init.normal_(model.weight)  # 非ゼロの実重み（ロード検証・EMA との識別のため）
    opt = optim.AdamW(model.parameters(), lr=1e-4)
    torch.save(model.state_dict(), tmp_path / f"checkpoint_{epoch}.pt")
    if with_optimizer:
        torch.save(opt.state_dict(), tmp_path / f"optimizer_{epoch}.pt")
    if with_ema:
        # Phase 2 が出力する EMA ファイル。continue_training はこれらを無視すべき。
        # 生重みと識別できるよう「ゼロ」で保存する（誤って EMA を拾ったら重み比較が落ちる）
        zero_state = {k: torch.zeros_like(v) for k, v in model.state_dict().items()}
        torch.save(zero_state, tmp_path / f"ema_checkpoint_{epoch}.pt")
        torch.save({"shadow": zero_state, "num_updates": 3}, tmp_path / f"ema_state_{epoch}.pt")
    return model.state_dict()


def test_continue_training_resumes_and_ignores_ema_files(tmp_path):
    saved_state = _save_tiny_checkpoint(tmp_path, epoch=0, with_optimizer=True, with_ema=True)

    model = torch.nn.Linear(4, 4)
    wrapper = types.SimpleNamespace(module=model)  # DDP の .module を模擬
    opt = optim.AdamW(model.parameters(), lr=1e-4)

    next_epoch = continue_training(str(tmp_path), wrapper, opt)

    assert next_epoch == 1  # max_epoch(0) + 1。ema_* を checkpoint として拾っていない
    # 生重み checkpoint_0.pt（非ゼロ）がロードされている。誤って ema_checkpoint_0.pt（ゼロ）を拾えば落ちる
    for k in saved_state:
        assert torch.equal(model.state_dict()[k], saved_state[k])
        assert not torch.equal(model.state_dict()[k], torch.zeros_like(saved_state[k]))


def test_continue_training_loads_pretrained_only_when_no_optimizer(tmp_path):
    # optimizer が無い場合は事前学習チェックポイントとして重みだけロードし epoch 0 から開始
    saved_state = _save_tiny_checkpoint(tmp_path, epoch=5, with_optimizer=False, with_ema=False)
    model = torch.nn.Linear(4, 4)
    with torch.no_grad():  # 既知の初期値（全ゼロ）→ ロードされたか確実に判定できる
        model.weight.zero_()
        model.bias.zero_()
    wrapper = types.SimpleNamespace(module=model)
    opt = optim.AdamW(model.parameters(), lr=1e-4)

    next_epoch = continue_training(str(tmp_path), wrapper, opt)
    assert next_epoch == 0
    # 事前学習重みが実際にロードされた（load をスキップする mutation なら初期ゼロのままで落ちる）
    for k in saved_state:
        assert torch.equal(model.state_dict()[k], saved_state[k])
