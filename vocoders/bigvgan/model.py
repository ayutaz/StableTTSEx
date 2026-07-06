import torch
import torch.nn as nn

# config.json of nvidia/bigvgan_v2_44khz_128band_512x (generator-relevant keys).
# use_tanh_at_final / use_bias_at_final default to True upstream, so False must be set explicitly.
config_dict = {
    "num_mels": 128,
    "upsample_rates": [8, 4, 2, 2, 2, 2],
    "upsample_kernel_sizes": [16, 8, 4, 4, 4, 4],
    "upsample_initial_channel": 1536,
    "resblock": "1",
    "resblock_kernel_sizes": [3, 7, 11],
    "resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
    "activation": "snakebeta",
    "snake_logscale": True,
    "use_tanh_at_final": False,
    "use_bias_at_final": False,
    "use_cuda_kernel": False,
}

# download_link: https://huggingface.co/nvidia/bigvgan_v2_44khz_128band_512x/resolve/main/bigvgan_generator.pt
class BigVGANWrapper(nn.Module):
    def __init__(self, model_path):
        super().__init__()
        from .bigvgan import BigVGAN
        from .env import AttrDict

        self.model = BigVGAN(AttrDict(config_dict), use_cuda_kernel=False)
        sd = torch.load(model_path, map_location='cpu', weights_only=True)
        self.model.load_state_dict(sd['generator'])  # checkpoint keys are weight_g/weight_v; load before removing
        self.model.remove_weight_norm()
        self.model.eval()

    @ torch.inference_mode()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # mel (B, num_mels, T) -> waveform (B, T), matching ffgan/vocos wrappers
        return self.model(x).squeeze(1)
