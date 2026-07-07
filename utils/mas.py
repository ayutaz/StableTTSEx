"""GPU ネイティブの Monotonic Alignment Search（MAS）。

`monotonic_align`（numba・CPU）の `maximum_path` は毎ステップ `neg_cent.cpu().numpy()` で
GPU→CPU 同期を起こし、学習パイプラインを直列化する。この実装は同じ DP をすべて GPU テンソル演算
（バッチと text 軸をベクトル化し、mel 軸 y のみ逐次）で行い、その同期を除去する。

numba 版（monotonic_align/core.py::maximum_path_jit）と **ビット同一の 0/1 アラインメント**を返す。
値の DP には対角バンド制約 [max(0, t_x + y - t_y), min(t_x, y + 1)) を厳密に再現しており、
バンド外の value は生の neg_cent を保持する（後段のバックトラックが読むのはバンド内のみ）。
"""

import torch

_NEG = -1e9  # numba 版の max_neg_val と同値（-inf ではなく厳密一致のため -1e9 を使う）


@torch.no_grad()
def maximum_path_torch(neg_cent: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """MAS を GPU 上で解く。

    Args:
        neg_cent: アラインメントコスト [b, t_y, t_x]（t_y=mel フレーム, t_x=text トークン）
        mask: 有効領域 {0,1} の [b, t_y, t_x]

    Returns:
        path: 0/1 のアラインメント [b, t_y, t_x]（neg_cent と同じ dtype/device）
    """
    device, dtype = neg_cent.device, neg_cent.dtype
    b, t_y, t_x = neg_cent.shape

    # numba 版が値配列として使う float32 の raw neg_cent（in-place で DP 累積する）
    value = neg_cent.float().clone()

    # 各バッチの有効長（numba: t_ys = mask.sum(1)[:,0], t_xs = mask.sum(2)[:,0]）
    t_y_len = mask.sum(1)[:, 0].long()  # [b]
    t_x_len = mask.sum(2)[:, 0].long()  # [b]

    arange_b = torch.arange(b, device=device)
    x_idx = torch.arange(t_x, device=device)  # [t_x]

    # --- forward DP ---
    # value[y, x] += max(v_prev, v_cur)
    #   v_cur  = value[y-1, x]（ただし x==y は _NEG）
    #   v_prev = value[y-1, x-1]（x==0 は y>0 のとき _NEG）
    # y=0 はバンドが {0} で += max(0, _NEG)=0 の no-op なので省略できる。
    for y in range(1, t_y):
        prev = value[:, y - 1, :]  # [b, t_x]
        v_cur = prev.clone()
        if y < t_x:
            v_cur[:, y] = _NEG
        # v_prev[x] = prev[x-1], v_prev[0] = _NEG
        v_prev = torch.nn.functional.pad(prev, (1, 0), value=_NEG)[:, :-1]
        update = torch.maximum(v_prev, v_cur)  # [b, t_x]

        # 対角バンド [lo, hi) を厳密再現。バンド外は value を据え置く（生の neg_cent のまま）
        lo = torch.clamp(t_x_len + y - t_y_len, min=0)  # [b]
        hi = torch.clamp(torch.full_like(t_x_len, y + 1), max=t_x_len)  # min(t_x_len, y+1)
        band = (x_idx.unsqueeze(0) >= lo.unsqueeze(1)) & (x_idx.unsqueeze(0) < hi.unsqueeze(1))  # [b, t_x]
        value[:, y, :] = torch.where(band, value[:, y, :] + update, value[:, y, :])

    # --- backtrack ---
    path = torch.zeros(b, t_y, t_x, device=device, dtype=dtype)
    index = (t_x_len - 1).clamp(min=0)  # [b] 末尾の text トークンから開始
    for y in range(t_y - 1, -1, -1):
        active = y < t_y_len  # [b] このバッチの有効 mel 行か
        path[arange_b, y, index] = torch.where(active, torch.ones_like(path[:, 0, 0]), path[arange_b, y, index])
        if y >= 1:
            v_a = value[arange_b, y - 1, index]  # value[y-1, index]
            v_b = value[arange_b, y - 1, (index - 1).clamp(min=0)]  # value[y-1, index-1]
            move = (index != 0) & ((index == y) | (v_a < v_b)) & active
            index = index - move.long()
    return path
