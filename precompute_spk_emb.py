"""TLA-SA 教師埋め込みのオフライン事前計算（Phase 3、one-off スクリプト）。

`filelist.json`（各行に `audio_path` を含む JSON Lines）を読み、原音声を 16kHz にリサンプルして凍結
SV（話者検証）エンコーダに通し、発話ごとの埋め込みを sidecar `.pt` に保存、filelist の各行に
`spk_emb_path` を追記する。学習ループ（datas/dataset.py）はこの `.pt` を読むだけで SV エンコーダ自体は
呼ばない（重い依存と計算を学習から排除する設計）。

- 教師は評価の ECAPA とは別系統にすること（`config.tla_sa_teacher` と `Config.teacher` を整合させる）。
- SV 重み・キャッシュは `model_save_path` の外に置く（continue_training が model_save_path 直下の
  全 *.pt に int(epoch) を実行するため、混ぜると壊れる）。
- 実行前に下記 Config を編集する（設定は CLI 引数でなく dataclass を直接編集する方針）。

    uv run python precompute_spk_emb.py
"""

import hashlib
import json
import os
from dataclasses import dataclass

import torch
import torchaudio
import torchaudio.functional as AF

from models.tla_sa import load_sv_teacher


@dataclass
class Config:
    filelist_path: str = "filelists/filelist.json"
    # config.tla_sa_teacher と一致させる（campplus=192次元 / wavlm_sv=512次元）。
    # Phase 3 スモークは vendor 不要な wavlm_sv。本番は Apache-2.0 の campplus に切り替える
    teacher: str = "wavlm_sv"
    emb_dir: str = "spk_emb"  # sidecar .pt の保存先（model_save_path の外）
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def _campplus_features(wav16k):
    # CAM++ は kaldi fbank(80次元) + 発話平均正規化(CMN) が入力。TTS の 44.1kHz/128-mel slaney とは
    # 非互換なので raw 音声から再計算する
    feat = torchaudio.compliance.kaldi.fbank(wav16k, num_mel_bins=80, sample_frequency=16000)
    feat = feat - feat.mean(dim=0, keepdim=True)
    return feat.unsqueeze(0)  # [1, T, 80]


def _embed(teacher_name, model, wav, sr, device):
    wav = wav.mean(0, keepdim=True)  # downmix to mono, shape [1, T]
    if sr != 16000:
        wav = AF.resample(wav, sr, 16000)
    with torch.no_grad():
        if teacher_name == "campplus":
            emb = model(_campplus_features(wav).to(device))
        else:  # wavlm_sv (transformers WavLMForXVector)
            emb = model(wav.to(device)).embeddings
    return emb.reshape(-1).cpu()  # [D']


def main():
    cfg = Config()
    os.makedirs(cfg.emb_dir, exist_ok=True)
    model = load_sv_teacher(cfg.teacher, device=cfg.device)

    with open(cfg.filelist_path, encoding="utf-8") as f:
        lines = [json.loads(line.strip()) for line in f]

    for i, item in enumerate(lines):
        wav, sr = torchaudio.load(item["audio_path"])
        emb = _embed(cfg.teacher, model, wav, sr, cfg.device)
        # content-addressed 命名（audio_path の hash）。run/filelist をまたいでも衝突せず、再実行も冪等
        # （enumerate 連番だと別 filelist を同一 emb_dir で処理したとき silent に上書きされる）
        emb_path = os.path.join(cfg.emb_dir, hashlib.sha1(item["audio_path"].encode("utf-8")).hexdigest()[:16] + ".pt")
        torch.save(emb, emb_path)
        item["spk_emb_path"] = emb_path
        if i % 500 == 0:
            print(f"{i}/{len(lines)} (emb dim={emb.numel()})", flush=True)

    with open(cfg.filelist_path, "w", encoding="utf-8") as f:
        for item in lines:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"done: wrote {len(lines)} spk_emb sidecars to {cfg.emb_dir}/ and updated {cfg.filelist_path}")


if __name__ == "__main__":
    main()
