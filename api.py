from dataclasses import asdict

import torch
import torch.nn as nn

from config import MelConfig, ModelConfig
from datas.dataset import intersperse
from models.model import StableTTS
from text import cleaned_text_to_sequence, symbols
from text.english import english_to_ipa2
from text.japanese import japanese_to_ipa2
from text.mandarin import chinese_to_cnm3
from utils.audio import LogMelSpectrogram, load_and_resample_audio


def get_vocoder(model_path, model_name="ffgan") -> nn.Module:
    if model_name == "ffgan":
        # training or changing ffgan config is not supported in this repo
        # you can train your own model at https://github.com/fishaudio/vocoder
        from vocoders.ffgan.model import FireflyGANBaseWrapper

        vocoder = FireflyGANBaseWrapper(model_path)

    elif model_name == "vocos":
        from config import MelConfig, VocosConfig
        from vocoders.vocos.models.model import Vocos

        vocoder = Vocos(VocosConfig(), MelConfig())
        vocoder.load_state_dict(torch.load(model_path, weights_only=True, map_location="cpu"))
        vocoder.eval()

    elif model_name == "bigvgan":
        from vocoders.bigvgan.model import BigVGANWrapper

        vocoder = BigVGANWrapper(model_path)

    else:
        raise NotImplementedError(f"Unsupported model: {model_name}")

    return vocoder


class StableTTSAPI(nn.Module):
    def __init__(self, tts_model_path, vocoder_model_path, vocoder_name="ffgan"):
        super().__init__()

        self.mel_config = MelConfig()
        self.tts_model_config = ModelConfig()

        self.mel_extractor = LogMelSpectrogram(**asdict(self.mel_config))

        # text to mel spectrogram
        self.tts_model = StableTTS(len(symbols), self.mel_config.n_mels, **asdict(self.tts_model_config))
        self.tts_model.load_state_dict(torch.load(tts_model_path, map_location="cpu", weights_only=True))
        self.tts_model.eval()

        # mel spectrogram to waveform
        self.vocoder_model = get_vocoder(vocoder_model_path, vocoder_name)
        self.vocoder_model.eval()

        self.g2p_mapping = {
            "chinese": chinese_to_cnm3,
            "japanese": japanese_to_ipa2,
            "english": english_to_ipa2,
        }
        self.supported_languages = self.g2p_mapping.keys()

    @torch.inference_mode()
    def get_style_vector(self, ref_audio, ref_window_seconds=None, ref_window_hop_seconds=None):
        """Compute a style vector (1, gin_channels) from one or more reference audio files.

        ref_window_seconds: if set, each file is sliced into fixed-length windows which are
        encoded separately and averaged. Otherwise each file is encoded as a single window.
        """
        device = next(self.parameters()).device
        if isinstance(ref_audio, str):
            ref_audio = [ref_audio]

        mels = []
        for audio_path in ref_audio:
            audio = load_and_resample_audio(audio_path, self.mel_config.sample_rate)
            if audio is None:
                continue
            mels.append(self.mel_extractor(audio.to(device)))  # (1, n_mels, T)
        if not mels:
            raise ValueError("no valid reference audio files")

        if ref_window_seconds is not None:
            win = int(ref_window_seconds * self.mel_config.sample_rate / self.mel_config.hop_length)
            hop = (
                int(ref_window_hop_seconds * self.mel_config.sample_rate / self.mel_config.hop_length)
                if ref_window_hop_seconds is not None
                else win // 2
            )

        styles_per_file = [[] for _ in mels]
        windows, window_file_ids = [], []  # equal-length windows, batched into one forward
        for i, mel in enumerate(mels):
            mel_length = mel.size(-1)
            if ref_window_seconds is None or mel_length < win:
                # whole file as a single window (variable length, forwarded individually)
                styles_per_file[i].append(self.tts_model.ref_encoder(mel, None))
            else:
                starts = list(range(0, mel_length - win + 1, hop))
                if mel_length - (starts[-1] + win) >= win // 2:
                    starts.append(mel_length - win)  # end-aligned window for the residual
                for start in starts:
                    windows.append(mel[:, :, start : start + win])
                    window_file_ids.append(i)
        if windows:
            window_styles = self.tts_model.ref_encoder(torch.cat(windows, dim=0), None)
            for i, style in zip(window_file_ids, window_styles.split(1, dim=0), strict=True):
                styles_per_file[i].append(style)

        # two-stage average: within each file, then across files (files weighted equally, not by window count)
        file_styles = [torch.cat(styles, dim=0).mean(dim=0, keepdim=True) for styles in styles_per_file]
        return torch.cat(file_styles, dim=0).mean(dim=0, keepdim=True)

    @torch.inference_mode()
    def inference(
        self,
        text,
        ref_audio,
        language,
        step,
        temperature=1.0,
        length_scale=1.0,
        solver=None,
        cfg=3.0,
        sway_coef=None,
        cfg_rescale=0.0,
        cfg_interval=None,
        ref_window_seconds=None,
        slg_scale=0.0,
        slg_layers=(2,),
        slg_t_range=(0.0, 0.5),
    ):
        device = next(self.parameters()).device
        phonemizer = self.g2p_mapping.get(language)

        text = phonemizer(text)
        text = torch.tensor(
            intersperse(cleaned_text_to_sequence(text), item=0), dtype=torch.long, device=device
        ).unsqueeze(0)
        text_length = torch.tensor([text.size(-1)], dtype=torch.long, device=device)

        if isinstance(ref_audio, (list, tuple)) or ref_window_seconds is not None:
            c = self.get_style_vector(ref_audio, ref_window_seconds)
            ref_audio = None
        else:
            c = None
            ref_audio = load_and_resample_audio(ref_audio, self.mel_config.sample_rate).to(device)
            ref_audio = self.mel_extractor(ref_audio)

        mel_output = self.tts_model.synthesise(
            text,
            text_length,
            step,
            temperature,
            ref_audio,
            length_scale,
            solver,
            cfg,
            sway_coef=sway_coef,
            cfg_rescale=cfg_rescale,
            cfg_interval=cfg_interval,
            c=c,
            slg_scale=slg_scale,
            slg_layers=slg_layers,
            slg_t_range=slg_t_range,
        )["decoder_outputs"]
        audio_output = self.vocoder_model(mel_output)
        return audio_output.cpu(), mel_output.cpu()

    def get_params(self):
        tts_param = sum(p.numel() for p in self.tts_model.parameters()) / 1e6
        vocoder_param = sum(p.numel() for p in self.vocoder_model.parameters()) / 1e6
        return tts_param, vocoder_param


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tts_model_path = "./checkpoints/checkpoint_0.pt"
    vocoder_model_path = "./vocoders/pretrained/vocos.pt"

    model = StableTTSAPI(tts_model_path, vocoder_model_path, "vocos")
    model.to(device)

    text = "樱落满殇祈念集……殇歌花落集思祈……樱花满地集于我心……揲舞纷飞祈愿相随……"
    audio = "./audio_1.wav"

    audio_output, mel_output = model.inference(text, audio, "chinese", 10, solver="dopri5", cfg=3)
    print(audio_output.shape)
    print(mel_output.shape)

    import torchaudio

    torchaudio.save("output.wav", audio_output, MelConfig().sample_rate)
