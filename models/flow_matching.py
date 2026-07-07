import functools

import torch
import torch.nn.functional as F
from torchdiffeq import odeint

from models.estimator import Decoder


# modified from https://github.com/shivammehta25/Matcha-TTS/blob/main/matcha/models/components/flow_matching.py
class CFMDecoder(torch.nn.Module):
    def __init__(
        self,
        noise_channels,
        cond_channels,
        hidden_channels,
        out_channels,
        filter_channels,
        n_heads,
        n_layers,
        kernel_size,
        p_dropout,
        gin_channels,
        timestep_sampling="cosine",
        logit_normal_m=0.0,
        logit_normal_s=1.0,
    ):
        super().__init__()
        self.noise_channels = noise_channels
        self.cond_channels = cond_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.filter_channels = filter_channels
        self.gin_channels = gin_channels
        self.sigma_min = 1e-4
        # Phase 2 施策5: 学習時 timestep サンプリング（推論には無関係）
        if timestep_sampling not in ("cosine", "logit_normal"):
            raise ValueError(f"timestep_sampling must be 'cosine' or 'logit_normal', got {timestep_sampling!r}")
        self.timestep_sampling = timestep_sampling
        self.logit_normal_m = logit_normal_m
        self.logit_normal_s = logit_normal_s

        self.estimator = Decoder(
            noise_channels,
            cond_channels,
            hidden_channels,
            out_channels,
            filter_channels,
            p_dropout,
            n_layers,
            n_heads,
            kernel_size,
            gin_channels,
        )

    @torch.inference_mode()
    def forward(self, mu, mask, n_timesteps, temperature=1.0, c=None, solver=None, cfg_kwargs=None, sway_coef=None):
        """Forward diffusion

        Args:
            mu (torch.Tensor): output of encoder
                shape: (batch_size, n_feats, mel_timesteps)
            mask (torch.Tensor): output_mask
                shape: (batch_size, 1, mel_timesteps)
            n_timesteps (int): number of diffusion steps
            temperature (float, optional): temperature for scaling noise. Defaults to 1.0.
            c (torch.Tensor, optional): speaker embedding
                shape: (batch_size, gin_channels)
            solver: see https://github.com/rtqichen/torchdiffeq for supported solvers
            cfg_kwargs: used for cfg inference
            sway_coef (float, optional): sway sampling coefficient. Only effective with fixed-step
                solvers (euler/midpoint/rk4 etc.); adaptive solvers like dopri5 treat t_span as
                output evaluation points only, so it has no effect there.

        Returns:
            sample: generated mel-spectrogram
                shape: (batch_size, n_feats, mel_timesteps)
        """

        z = torch.randn_like(mu) * temperature
        t_span = torch.linspace(0, 1, n_timesteps + 1, device=mu.device)
        if sway_coef is not None and sway_coef != 0.0:
            # Sway Sampling (F5-TTS, arXiv:2410.06885). fixed-step solvers only; s=-1 matches the training-time cosine schedule
            # monotonicity (dt'/dt = (1+s) - s*(pi/2)*sin(pi*t/2) >= 0) requires -1 <= s <= 2/(pi-2) (~1.752); clamp slightly inside for float32 grid safety
            s = min(max(float(sway_coef), -1.0), 1.75)
            t_span = t_span + s * (torch.cos(torch.pi / 2 * t_span) - 1 + t_span)

        # cfg control
        if cfg_kwargs is None:
            estimator = functools.partial(self.estimator, mask=mask, mu=mu, c=c)
        else:
            estimator = functools.partial(self.cfg_wrapper, mask=mask, mu=mu, c=c, cfg_kwargs=cfg_kwargs)

        trajectory = odeint(estimator, z, t_span, method=solver, rtol=1e-5, atol=1e-5)
        return trajectory[-1]

    # cfg inference
    def cfg_wrapper(self, t, x, mask, mu, c, cfg_kwargs):
        cfg_strength = cfg_kwargs["cfg_strength"]
        cfg_interval = cfg_kwargs.get("cfg_interval")  # None or (t_min, t_max)
        cfg_rescale = cfg_kwargs.get("cfg_rescale", 0.0)  # 0.0 = off (Lin+ 2023 arXiv:2305.08891 §3.4)
        slg_scale = cfg_kwargs.get("slg_scale", 0.0)  # 0.0 = off (SD3.5 Skip Layer Guidance)
        slg_layers = cfg_kwargs.get("slg_layers", (2,))
        slg_t_range = cfg_kwargs.get("slg_t_range", (0.0, 0.5))

        cond_output = self.estimator(t, x, mask, mu, c)
        t_now = float(t)

        output = cond_output
        # cfg_strength == 1.0 reduces to cond_output, so skip the uncond forward (also enables SLG-only inference at cfg=1.0)
        if cfg_strength != 1.0 and (cfg_interval is None or cfg_interval[0] <= t_now <= cfg_interval[1]):
            fake_speaker = cfg_kwargs["fake_speaker"].repeat(x.size(0), 1)
            fake_content = cfg_kwargs["fake_content"].repeat(x.size(0), 1, x.size(-1))
            uncond_output = self.estimator(t, x, mask, fake_content, fake_speaker)
            output = uncond_output + cfg_strength * (cond_output - uncond_output)
            if cfg_rescale > 0.0:
                std_cond = cond_output.std(dim=(1, 2), keepdim=True)
                std_cfg = output.std(dim=(1, 2), keepdim=True)
                rescaled = output * (std_cond / (std_cfg + 1e-8))
                output = cfg_rescale * rescaled + (1.0 - cfg_rescale) * output
        if slg_scale > 0.0 and slg_t_range[0] <= t_now < slg_t_range[1]:
            skip_output = self.estimator(t, x, mask, mu, c, skip_layers=slg_layers)
            output = output + slg_scale * (cond_output - skip_output)
        return output

    def compute_loss(self, x1, mask, mu, c):
        """Computes diffusion loss

        Args:
            x1 (torch.Tensor): Target
                shape: (batch_size, n_feats, mel_timesteps)
            mask (torch.Tensor): target mask
                shape: (batch_size, 1, mel_timesteps)
            mu (torch.Tensor): output of encoder
                shape: (batch_size, n_feats, mel_timesteps)
            c (torch.Tensor, optional): speaker condition.

        Returns:
            loss: conditional flow matching loss
            y: conditional flow
                shape: (batch_size, n_feats, mel_timesteps)
        """
        b, _, t = mu.shape

        # random timestep
        if self.timestep_sampling == "logit_normal":
            # Phase 2 施策5: SD3 (arXiv:2403.03206) の logit-normal(m, s)。中間 t を重点サンプリング。
            # sigmoid は有限入力で (0,1) 開区間に収まるためクランプ不要（端点で y/u が退化しない）
            eps = torch.randn([b, 1, 1], device=mu.device, dtype=mu.dtype)
            t = torch.sigmoid(self.logit_normal_m + self.logit_normal_s * eps)
        else:
            # use cosine timestep scheduler from cosyvoice: https://github.com/FunAudioLLM/CosyVoice/blob/main/cosyvoice/flow/flow_matching.py
            t = torch.rand([b, 1, 1], device=mu.device, dtype=mu.dtype)
            t = 1 - torch.cos(t * 0.5 * torch.pi)

        # sample noise p(x_0)
        z = torch.randn_like(x1)

        y = (1 - (1 - self.sigma_min) * t) * z + t * x1
        u = x1 - (1 - self.sigma_min) * z

        loss = F.mse_loss(self.estimator(t.squeeze(), y, mask, mu, c), u, reduction="sum") / (
            torch.sum(mask) * u.size(1)
        )
        return loss, y
