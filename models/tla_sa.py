"""TLA-SA: Timestep-Layer Aligned Speaker Alignment（Phase 3 第一弾、arXiv:2511.09995 / REPA 2410.06940）。

flow-matching デコーダの各 DiT ブロックの中間表現を、凍結事前学習話者検証(SV)エンコーダの埋め込みへ
cosine 整列させる**学習時のみ**の補助損失。話者情報が「初期 denoising step・浅い層」に偏在する観察に基づき、
denoising timestep から生成した層重み w で層別損失を動的加重する。推論時はこのヘッドを丸ごと捨てる
（`api.py` は TLASAHead を構築しない）ため、推論経路・チェックポイント形式は一切変わらない。

設計判断（設計調査ワークフロー wf_205c00ab-204 で確定）:
- TLASAHead は StableTTS の submodule にせず **train.py 側の独立モジュール**として独立 DDP でラップする。
  → checkpoint_{epoch}.pt は baseline とキー集合完全一致、api.py/utils/load.py/utils/ema.py は無改修。
- 教師 SV 埋め込みは **precompute_spk_emb.py でオフライン事前計算**（dataset は raw 音声を持たない）。
  学習ループでは事前計算済みの [B, D'] を受け取るだけで、SV エンコーダ自体は呼ばない。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.estimator import SinusoidalPosEmb


class TLASAHead(nn.Module):
    """層別 projection + timestep 依存の層重み + cosine 整列損失。

    forward の入力:
        hiddens: list[Tensor]、各 [B, d_hidden, T]（CFMDecoder の各ブロック出力、n_layers 本）
        t:       [B]、サンプルされた denoising timestep
        e_sa:    [B, d_teacher]、凍結 SV 教師埋め込み（事前計算済み・detach 済み）
        ymask:   [B, 1, T]、mel の有効長マスク
        valid:   [B]、cfg ドロップされていない（real speaker）サンプルの bool マスク
    戻り値: スカラ損失（cfg ドロップサンプルは除外して平均）
    """

    def __init__(
        self,
        n_layers=6,
        d_hidden=256,
        d_teacher=192,
        head_hidden=512,
        time_dim=256,
        alpha=0.01,
        uniform=False,
    ):
        super().__init__()
        self.n_layers = n_layers
        self.alpha = alpha
        self.uniform = uniform
        # 層ごとに独立した 3 層 MLP（d_hidden -> head_hidden -> head_hidden -> d_teacher）
        self.proj = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(d_hidden, head_hidden),
                    nn.SiLU(),
                    nn.Linear(head_hidden, head_hidden),
                    nn.SiLU(),
                    nn.Linear(head_hidden, d_teacher),
                )
                for _ in range(n_layers)
            ]
        )
        # uniform=True のときは層重み固定（timestep MLP・entropy 正則を持たない）
        if not uniform:
            self.time_emb = SinusoidalPosEmb(time_dim)
            self.time_mlp = nn.Sequential(nn.Linear(time_dim, 128), nn.SiLU(), nn.Linear(128, n_layers))

    @staticmethod
    def _masked_mean(e, ymask):
        # e: [B, d, T]、ymask: [B, 1, T] -> [B, d]。bf16 の中間表現を fp32 に昇格して整列を安定化
        e = e.float()
        ymask = ymask.float()
        return (e * ymask).sum(-1) / ymask.sum(-1).clamp_min(1e-5)

    def forward(self, hiddens, t, e_sa, ymask, valid):
        e_sa = F.normalize(e_sa.float(), dim=-1)  # [B, D']
        layer_losses = []
        for i in range(self.n_layers):
            pooled = self._masked_mean(hiddens[i], ymask)  # [B, d_hidden] fp32
            projected = F.normalize(self.proj[i](pooled), dim=-1)  # [B, D'] fp32
            layer_losses.append(1.0 - (projected * e_sa).sum(-1))  # [B]、1 - cos
        L = torch.stack(layer_losses, dim=1)  # [B, N]

        if self.uniform:
            w = torch.full_like(L, 1.0 / self.n_layers)
            reg = torch.zeros(L.size(0), device=L.device, dtype=L.dtype)
        else:
            logits = self.time_mlp(self.time_emb(t.float().reshape(-1)))  # [B, N]
            w = torch.softmax(logits, dim=-1)
            reg = (w * (w + 1e-8).log()).sum(-1)  # [B]、= -Entropy(w)（正則で加重の一極集中を抑制）

        per_sample = (w * L).sum(-1) + self.alpha * reg  # [B]
        v = valid.float()  # [B]
        # cfg ドロップされた（real speaker でない）サンプルは整列対象外。全ドロップ時の 0 割を clamp で保護
        return (per_sample * v).sum() / v.sum().clamp_min(1.0)


def load_sv_teacher(name, device="cpu"):
    """凍結話者検証エンコーダをロードする（TLA-SA の整列教師）。

    **学習ループでは呼ばない**（precompute_spk_emb.py で埋め込みを事前計算するため）。
    評価器（ECAPA）とは別系統のアーキにすること（テストに教えるバイアス回避）。
    """
    if name == "wavlm_sv":
        from transformers import WavLMForXVector

        model = WavLMForXVector.from_pretrained("microsoft/wavlm-base-plus-sv")
    elif name == "campplus":
        from models.sv_teacher_campplus import load_campplus

        model = load_campplus()
    else:
        raise ValueError(f"unknown tla_sa_teacher: {name!r} (expected 'campplus' or 'wavlm_sv')")
    return model.to(device).eval().requires_grad_(False)
