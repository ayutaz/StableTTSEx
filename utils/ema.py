import torch


class EMA:
    """モデル重みの指数移動平均（Phase 2 施策6）。

    DDP では rank 0 でのみ生成する（DDP が全 rank の重みを同期するため rank 0 の複製が代表となる）。
    学習が低精度でも安定するよう float 側は fp32 シャドウで保持し、整数バッファは dtype を保って追従する。

    decay は warmup 付き上限で、総ステップが少ない短期学習でも初期の追従を確保する:
        decay(step) = min(decay_max, (1 + step) / (warmup + step))

    state_dict() は module.state_dict() と同一キー構造の重み辞書を返すため、推論側（api.py）に
    そのままロードできる。ema_training_state()/load_ema_training_state() はレジューム用（num_updates 込み）。
    """

    def __init__(self, model: torch.nn.Module, decay: float = 0.9995, warmup: int = 10):
        self.decay_max = decay
        self.warmup = warmup
        self.num_updates = 0
        # DDP ラップ前の module を渡すこと（キーに "module." 前置が付かないようにする）
        self.shadow = {
            name: (p.detach().clone().float() if p.dtype.is_floating_point else p.detach().clone())
            for name, p in model.state_dict().items()
        }
        # シャドウはモデルと同一デバイスに置く（update の in-place 演算は cross-device を許さない）。
        # レジューム時は CPU からロードするため、この device へ戻す必要がある
        self.device = next(iter(self.shadow.values())).device if self.shadow else torch.device("cpu")

    def _decay(self) -> float:
        return min(self.decay_max, (1 + self.num_updates) / (self.warmup + self.num_updates))

    @torch.no_grad()
    def update(self, model: torch.nn.Module):
        self.num_updates += 1
        d = self._decay()
        for name, p in model.state_dict().items():
            s = self.shadow[name]
            if p.dtype.is_floating_point:
                s.mul_(d).add_(p.detach().float(), alpha=1.0 - d)
            else:
                s.copy_(p.detach())  # 整数バッファ等は EMA せず最新値に追従

    def state_dict(self) -> dict:
        """推論用の重み辞書（module.state_dict() 互換）。"""
        return self.shadow

    def ema_training_state(self) -> dict:
        """レジューム用の完全状態（シャドウ + 更新回数）。"""
        return {"shadow": self.shadow, "num_updates": self.num_updates}

    def load_ema_training_state(self, state: dict):
        # ロード元は CPU テンソル（train.py は map_location="cpu"）。update の cross-device クラッシュを避けるため
        # 必ずモデルと同一デバイスへ戻す
        self.shadow = {
            name: (t.float().to(self.device) if t.dtype.is_floating_point else t.to(self.device))
            for name, t in state["shadow"].items()
        }
        self.num_updates = state["num_updates"]
