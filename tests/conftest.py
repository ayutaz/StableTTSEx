"""pytest 共有設定とフィクスチャ。

方針（docs と調査の注意点に基づく）:
- GPU/実チェックポイント/ネットワークを要求しない。すべて CPU・決定論的に回す。
- モデル出力は torch/torchaudio のバージョンで微小に揺れるため、値ゴールデンではなく
  shape・有限性・同一 seed 内の一致（torch.equal）・関係性（単調性）・no-op 等価を検証する。
- train.py / preprocess.py はモジュールレベル副作用（CUDA_VISIBLE_DEVICES 設定・mkdir 等）が
  あるため import しない。テスト対象はそれらが組み立てる部品（config/models/utils/datas/text）。
"""

import os
import sys

# torch import より前に CPU 固定（GPU 機での device 選択・cudnn.benchmark 非決定を排除）。
# 既存値を尊重する setdefault だと GPU 機/CI で事前 export された値により CPU 強制が無効化されるため無条件代入する
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Windows の cp932 コンソールで IPA/日本語を含む失敗表示が UnicodeEncodeError にならないよう UTF-8 化
if sys.platform == "win32":
    for _stream in (sys.stdout, sys.stderr):
        try:
            _stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass

import pytest
import torch

# tiny config 制約:
#   hidden_channels は偶数（SinusoidalPosEmb）/ hidden_channels % n_heads == 0（Attention）/
#   n_dec_layers は偶数（use_lsc の U-Net 風 long skip）。p_dropout=0 + eval() で決定論にする。
TINY_MODEL_KWARGS = {
    "hidden_channels": 16,
    "filter_channels": 32,
    "n_heads": 2,
    "n_enc_layers": 2,
    "n_dec_layers": 2,
    "kernel_size": 3,
    "p_dropout": 0.0,
    "gin_channels": 16,
}
TINY_N_MELS = 8
TINY_N_VOCAB = 20  # テストで使う音素 ID の上限より十分大きければよい（実 n_vocab=401 は full-model テストのみ）


@pytest.fixture(scope="session")
def tiny_stabletts():
    """CPU 上の極小 StableTTS を返す factory（eval 済み）。timestep_sampling を差し替え可能。"""
    from models.model import StableTTS  # monotonic_align(numba JIT) を引くので遅延 import

    def _build(timestep_sampling="cosine", logit_normal_m=0.0, logit_normal_s=1.0):
        torch.manual_seed(0)
        model = StableTTS(
            TINY_N_VOCAB,
            TINY_N_MELS,
            **TINY_MODEL_KWARGS,
            timestep_sampling=timestep_sampling,
            logit_normal_m=logit_normal_m,
            logit_normal_s=logit_normal_s,
        )
        model.eval()
        return model

    return _build


@pytest.fixture(scope="session")
def tiny_cfm():
    """CPU 上の極小 CFMDecoder を返す factory（eval 済み）。monotonic_align を引かず軽量。"""
    from models.flow_matching import CFMDecoder

    def _build(timestep_sampling="cosine", logit_normal_m=0.0, logit_normal_s=1.0):
        torch.manual_seed(0)
        dec = CFMDecoder(
            TINY_N_MELS,  # noise_channels
            TINY_N_MELS,  # cond_channels
            16,  # hidden_channels
            TINY_N_MELS,  # out_channels
            32,  # filter_channels
            2,  # n_heads
            2,  # n_layers（偶数: use_lsc）
            3,  # kernel_size
            0.0,  # p_dropout
            16,  # gin_channels
            timestep_sampling=timestep_sampling,
            logit_normal_m=logit_normal_m,
            logit_normal_s=logit_normal_s,
        )
        dec.eval()
        return dec

    return _build
