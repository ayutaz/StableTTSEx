import json
import random

import torch
from torch.utils.data import Dataset

from text import cleaned_text_to_sequence


def intersperse(lst: list, item: int):
    """
    putting a blank token between any two input tokens to improve pronunciation
    see https://github.com/jaywalnut310/glow-tts/issues/43 for more details
    """
    result = [item] * (len(lst) * 2 + 1)
    result[1::2] = lst
    return result


class StableDataset(Dataset):
    def __init__(self, filelist_path, hop_length):
        self.filelist_path = filelist_path
        self.hop_length = hop_length

        self._load_filelist(filelist_path)

    def _load_filelist(self, filelist_path):
        filelist, lengths = [], []
        with open(filelist_path, encoding="utf-8") as f:
            for line in f:
                line = json.loads(line.strip())
                filelist.append((line["mel_path"], line["phone"], line.get("spk_emb_path")))
                lengths.append(line["mel_length"])

        self.filelist = filelist
        self.lengths = lengths  # length is used for DistributedBucketSampler

    def __len__(self):
        return len(self.filelist)

    def __getitem__(self, idx):
        mel_path, phone, spk_emb_path = self.filelist[idx]
        mel = torch.load(mel_path, map_location="cpu", weights_only=True)
        phone = torch.tensor(intersperse(cleaned_text_to_sequence(phone), 0), dtype=torch.long)
        # TLA-SA 教師埋め込み（precompute_spk_emb.py で事前計算済み）。spk_emb_path が無い filelist では None
        spk_emb = torch.load(spk_emb_path, map_location="cpu", weights_only=True) if spk_emb_path else None
        return mel, phone, spk_emb


def collate_fn(batch):
    texts = [item[1] for item in batch]
    mels = [item[0] for item in batch]
    mels_sliced = [random_slice_tensor(mel) for mel in mels]

    text_lengths = torch.tensor([text.size(-1) for text in texts], dtype=torch.long)
    mel_lengths = torch.tensor([mel.size(-1) for mel in mels], dtype=torch.long)
    mels_sliced_lengths = torch.tensor([mel_sliced.size(-1) for mel_sliced in mels_sliced], dtype=torch.long)

    # pad to the same length
    texts_padded = torch.nested.to_padded_tensor(torch.nested.nested_tensor(texts), padding=0)
    mels_padded = torch.nested.to_padded_tensor(torch.nested.nested_tensor(mels), padding=0)
    mels_sliced_padded = torch.nested.to_padded_tensor(torch.nested.nested_tensor(mels_sliced), padding=0)

    # TLA-SA 教師埋め込み。全サンプルに有れば [B, D']、いずれか欠ければ None（use_tla_sa=False 経路では未使用）
    spk_embs = [item[2] for item in batch]
    spk_emb = torch.stack(spk_embs) if all(e is not None for e in spk_embs) else None

    return texts_padded, text_lengths, mels_padded, mel_lengths, mels_sliced_padded, mels_sliced_lengths, spk_emb


# random slice mel for reference encoder to prevent overfitting
def random_slice_tensor(x: torch.Tensor):
    length = x.size(-1)
    if length < 12:
        return x
    segmnt_size = random.randint(length // 12, length // 3)
    start = random.randint(0, length - segmnt_size)
    return x[..., start : start + segmnt_size]
