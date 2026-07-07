"""Phase 1: StableTTSAPI.get_style_vector の複数参照平均の契約。
「各ファイル内で平均 → ファイル間は等重み平均（窓数で重み付けしない）」を検証する。
実チェックポイント/ボコーダ不要。unbound メソッドをスタブ化した self に束縛して呼ぶ。
"""

import types

import torch

import api as api_module
from api import StableTTSAPI

GIN = 4
N_MELS = 4


def _fake_ref_encoder(mel, mask):
    # (B, n_mels, T) → (B, gin)。各窓を最初のフレームの先頭 gin ちゃんねるで一意に符号化する（決定論）
    return mel[:, :GIN, 0]


def _make_api_stub():
    tts = types.SimpleNamespace(ref_encoder=_fake_ref_encoder)
    return types.SimpleNamespace(
        tts_model=tts,
        mel_config=types.SimpleNamespace(sample_rate=44100, hop_length=512),
        mel_extractor=lambda audio: audio,  # ロード済みテンソルをそのまま mel として使う
        parameters=lambda: iter([torch.zeros(1)]),  # device 判定用（CPU）
    )


def _call_get_style_vector(stub, ref_audio, mel_by_path, monkeypatch, **kwargs):
    # load_and_resample_audio をスタブし、パスごとに用意した mel テンソルを返す
    monkeypatch.setattr(api_module, "load_and_resample_audio", lambda path, sr: mel_by_path[path])
    return StableTTSAPI.get_style_vector(stub, ref_audio, **kwargs)


def test_single_str_and_singleton_list_are_equivalent(monkeypatch):
    stub = _make_api_stub()
    mel = torch.randn(1, N_MELS, 20)
    mel_by_path = {"a.wav": mel}
    as_str = _call_get_style_vector(stub, "a.wav", mel_by_path, monkeypatch)
    as_list = _call_get_style_vector(stub, ["a.wav"], mel_by_path, monkeypatch)
    assert torch.equal(as_str, as_list)
    assert as_str.shape == (1, GIN)


def test_two_files_equal_weight_average(monkeypatch):
    # 単一窓（ref_window_seconds=None）: 結果 = (style_A + style_B)/2
    stub = _make_api_stub()
    mel_a = torch.ones(1, N_MELS, 10) * 1.0  # ref_encoder → 全要素 1.0
    mel_b = torch.ones(1, N_MELS, 10) * 3.0  # ref_encoder → 全要素 3.0
    out = _call_get_style_vector(stub, ["a.wav", "b.wav"], {"a.wav": mel_a, "b.wav": mel_b}, monkeypatch)
    assert torch.allclose(out, torch.full((1, GIN), 2.0))  # (1+3)/2


def test_files_weighted_equally_not_by_window_count(monkeypatch):
    # 窓分割ありで窓数が偏っても、ファイル間は等重み（窓数重み付けでない）ことを検証。
    # A: 長さ20 → 4窓（全て値1.0）、B: 長さ8 → 1窓（値3.0）。
    # 等重み: (mean_A + mean_B)/2 = (1+3)/2 = 2.0。窓重み付けなら (4*1+1*3)/5 = 1.4。
    stub = _make_api_stub()
    mel_a = torch.ones(1, N_MELS, 20) * 1.0
    mel_b = torch.ones(1, N_MELS, 8) * 3.0
    out = _call_get_style_vector(
        stub,
        ["a.wav", "b.wav"],
        {"a.wav": mel_a, "b.wav": mel_b},
        monkeypatch,
        ref_window_seconds=0.1,  # win = int(0.1*44100/512) = 8 フレーム
    )
    assert torch.allclose(out, torch.full((1, GIN), 2.0))
