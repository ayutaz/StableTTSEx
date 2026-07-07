# Phase 2 実装計画 — レシピのみの再事前学習

作成日: 2026-07-07 / 対象: StableTTSEx（upstream v1.1、31.6M）/ 親文書: [architecture-improvement-research.md](architecture-improvement-research.md) §7 / 関連: [pretraining-report.md](pretraining-report.md)

## 0. 位置づけと原則

Phase 1（推論のみ・再学習なし、[architecture-improvement-research.md §9](architecture-improvement-research.md)）は完了。Phase 2 は**モデル構造を一切変えず、学習の手続き（レシピ）だけ**を変えて moe-speech 378h を再事前学習するフェーズ（施策 5・6、任意で 7）。

- **チェックポイント互換**: 重みの形状は不変。`checkpoint_0.pt`（upstream）から継続学習でき、出力される state_dict は既存の `api.py` / `webui.py` にそのままロードできる（施策7 の projection head を除く。7 は学習時のみ追加し推論では捨てる）。
- **Phase 1 との違い**: 再学習が必要。前回実績（378h・15 epochs ≈ 2.5h / $2.5）を維持中の vast インスタンスで再現する。
- **評価**: 現行 `japanese-378h` チェックポイントとの聴感 A/B + 定量指標（CER / ECAPA cos / mel 飽和 proxy）。Phase 1 の評価枠組み（`temps/phase1_eval/eval_phase1.py`）を再利用する。

対象施策（[改善候補マップ §6](architecture-improvement-research.md) より）:

| # | 施策 | 効果 | コスト | 互換性 |
|---|---|---|---|---|
| 5 | logit-normal(0,1) timestep サンプリング | 中 | 低（実質1行 + 再学習） | 互換（レシピのみ） |
| 6 | EMA 重み | 小〜中 | 低 | 互換（レシピのみ） |
| 7 | （任意）TLA-SA 話者整列補助損失 | 中 | 中 | ほぼ互換（学習時のみ head 追加） |

---

## 1. 施策5: logit-normal timestep サンプリング（本命）

### 現状

学習時の timestep サンプリングは CosyVoice 由来の cosine スケジューラで、noise 側（t≈0）を重点サンプリングしている。

```python
# models/flow_matching.py:137-139  compute_loss 内
t = torch.rand([b, 1, 1], device=mu.device, dtype=mu.dtype)
t = 1 - torch.cos(t * 0.5 * torch.pi)   # noise側(t≈0)重点
```

SD3 論文の ablation では **logit-normal(0,1)（中間 t 重点）が一貫して最良**で、現行 cosine は中間軽視の逆向きバイアス。

### 変更

`logit-normal(m, s)` は `t = sigmoid(m + s·ε), ε~N(0,1)`。SD3 の最良設定は `(m, s) = (0, 1)`。

```python
# 置換後（timestep_sampling == "logit_normal" のとき）
eps = torch.randn([b, 1, 1], device=mu.device, dtype=mu.dtype)
t = torch.sigmoid(logit_normal_m + logit_normal_s * eps)   # (0,1) 開区間、クランプ不要
```

`sigmoid` は有限入力で 0/1 に到達しないため、`y = (1-(1-σ_min)t)z + t·x1` の端点問題は起きない。

> **t 規約の反転に注意（`m≠0` に調整する場合）**: 本リポの補間は `y = (1-(1-σ_min)t)z + t·x1`（`z`=ノイズ, `x1`=データ）なので **t=0→ノイズ / t=1→データ**。SD3 の慣用（t=1→ノイズ）とは向きが逆。既定の `m=0` は `sigmoid` が 0.5 対称なため向きに依存せず問題ないが、将来ノイズ側/データ側どちらかに寄せたくて `m≠0` にする場合、**SD3 の推奨符号を本リポでは反転**させる必要がある（例: SD3 でノイズ側重点の正の `m` は、本リポでは負の `m`）。

### フラグ設計（A/B 再現性のため）

学習手続きの選択なので **`TrainConfig` に追加**し、`CFMDecoder` へコンストラクタ経由で渡す。デフォルトは `"cosine"` で既存挙動をビット単位維持する。

- `config.py` `TrainConfig`: `timestep_sampling: str = "cosine"`（`"cosine"` | `"logit_normal"`）、`logit_normal_m: float = 0.0`、`logit_normal_s: float = 1.0` を追加。
- `models/flow_matching.py` `CFMDecoder.__init__`: 引数 `timestep_sampling="cosine"`, `logit_normal_m=0.0`, `logit_normal_s=1.0` を受けて `self.` に保持。`compute_loss` で分岐。
- `models/model.py` `StableTTS.__init__`: `CFMDecoder(...)` 構築時に上記3値を渡す。`ModelConfig` ではなく引数で受ける（`ModelConfig` はチェックポイント意味論に紐づくため混ぜない）。
- `train.py`: `StableTTS(...)` 構築時に `train_config` の3値を渡す。推論側（`api.py`）は既定 `"cosine"` のままでよい（推論では compute_loss を呼ばない＝影響なし）。

> **注意（Phase 1 sway_coef との相互作用・要再検証）**: Phase 1 で推奨した `sway_coef=-1.0` は「euler の t_span を**学習時 cosine スケジュール**に一致させる」warp だった。学習側を logit-normal に変えると、その一致関係が崩れる。**Phase 2 モデルの推論では sway_coef / step を再スイープする**こと（`sway=None`（=一様 t_span）や別の s が最適になる可能性がある）。評価は Phase 1 プリセットに固定せず、複数の推論設定で比較する。

---

## 2. 施策6: EMA（指数移動平均）重み

### 現状

学習ループは EMA を持たず、生の重みをそのまま保存する。

```python
# train.py:110-125（抜粋）
loss = dur_loss + diff_loss + prior_loss
loss.backward(); optimizer.step(); scheduler.step()
...
torch.save(model.module.state_dict(), f"checkpoint_{epoch}.pt")
```

### 変更

学習中に重みの EMA シャドウを維持し、**生重みと EMA 重みの両方を保存**して A/B する。EMA チェックポイントは同一形状の state_dict なので、推論側は保存ファイルを差し替えるだけでよい（`api.py` 無改修）。

実装コンポーネント（実装済み。当初計画から下記2点を改善して確定）:

1. **`utils/ema.py`（新規）**: `EMA` クラス。`__init__(model)` で `state_dict()` 全キー（params+buffers）の fp32 シャドウを確保（整数バッファは dtype 保持）、`update(model)` で浮動小数は `shadow = decay·shadow + (1-decay)·param`・整数は最新値追従、`state_dict()`（推論用の module 互換重み）/ `ema_training_state()` / `load_ema_training_state()`（レジューム用、num_updates 込み）を持つ。シャドウはモデルと同一デバイスに置き、レジューム復元時も `self.device` へ戻す（`update` の in-place 演算は cross-device 不可のため）。
2. **`train.py`**:
   - DDP ラップ・`continue_training`（生重みロード）後、**rank 0 かつ `use_ema` のみ** `ema = EMA(model.module, decay, warmup)` を生成（生重みロード後に生成するのでシャドウは直近チェックポイント基準）。
   - `optimizer.step()` / `scheduler.step()` の直後に `ema.update(model.module)`。
   - 保存時に生重み（従来どおり）に加え、推論用 `ema_checkpoint_{epoch}.pt`（＝ `ema.state_dict()`、`api.py` にそのままロード可）とレジューム用 `ema_state_{epoch}.pt`（シャドウ + num_updates）を保存。**ファイル名を `checkpoint` 始まりにしない**（`continue_training` の `startswith("checkpoint")` 走査に誤マッチし生重みレジュームを汚染するため。当初計画の `checkpoint_ema_*` から変更）。
   - レジューム時は `ema_state_{current_epoch-1}.pt` があれば `load_ema_training_state` で復元、無ければ現重みから初期化して警告。
3. **`utils/load.py` は変更しない**（当初計画では `continue_training` に `ema` 引数追加を想定したが、EMA 復元を train.py 側で完結させる方が変更面が小さく安全。`continue_training` は生重み `checkpoint_*` / `optimizer_*` のみを走査し、`ema_*` は名前で無視される）。

> **decay の調整（短期学習の要点）**: 総ステップ = `15 epochs × len(dataloader)`。378h・batch 32・2GPU では総ステップが少なく、`decay=0.9999` だと EMA が初期値からほぼ動かない。**decay warmup**（例 `decay = min(decay_max, (1+step)/(10+step))`）を入れるか、`decay_max` を `0.999`〜`0.9995` に下げる。学習開始時の実 step 数を見て決める（TensorBoard の step 数で確認可能）。

---

## 3. 施策7: TLA-SA 話者整列補助損失（任意・実験的）

ゼロショット類似性（[§4 で特定した最大の残課題](architecture-improvement-research.md)）への低リスク bolt-on。**5+6 の効果を測ってから採否を判断**し、まず 5+6 を先行させる。

### 概要

flow-matching デコーダの中間表現を、事前学習済み話者埋め込み（ECAPA-TDNN、speechbrain、192次元）に整列させる補助損失。推論時は projection head を捨てる。

### 変更（実装時に詳細化）

- **デコーダ中間表現の露出**: `estimator`（`models/estimator.py`）が hidden を返せるようにする（最も侵襲的な箇所）。どの層・どうプールするか（時間平均など）は実装時に決める。
- **projection head**: 中間表現 → 192次元の小 MLP を `StableTTS` に追加（学習時のみ）。ゼロ初期化不要（推論で捨てるため）。
- **教師埋め込み**: 凍結 ECAPA でターゲット音声（または z スライス）から抽出。学習ループ内で毎バッチ計算 or 事前計算。
- **損失**: cosine/MSE を重み λ で総和に加算（`train.py` の `loss = dur + diff + prior + λ·tla_sa`）。
- **推論側の互換**: head パラメータは推論モデルに存在しないため、`api.py` のロードは既存のまま（EMA/生 state_dict に head キーが混ざる場合は保存時に head を除外するか、ロード側で `strict=False`）。実装時に「保存時 head 除外」を採る想定。

### 依存

`speechbrain` + ECAPA モデル DL が必要。**`pyproject.toml` には評価用依存を入れていない方針**（Phase 1 評価と同様、`uv pip` で venv に足すだけで `uv sync` で消える）。採用が決まった段階で pyproject 管理に昇格するか判断する。

---

## 4. 学習・評価プロトコル

### 学習ラン（各 ≈ 2.5h / $2.5、維持中の vast インスタンス）

初期値は前回と同じく **upstream `checkpoint_0.pt`** から 15 epochs 継続学習し、既存 `japanese-378h` と直接比較できる条件に揃える。

**推奨: まず 1 ラン（R2）で 5・6 を同時評価する。**

| ラン | 設定 | 得られる checkpoint |
|---|---|---|
| （既存） | cosine + EMAなし | `japanese-378h`（再学習不要、比較基準） |
| **R2** | logit_normal(0,1) + EMA | 生重み（＝施策5のみ相当）+ EMA 重み（＝施策5+6） |

EMA は学習軌道に影響しない受動的シャドウなので、R2 の生重み ＝「logit-normal のみ・EMAなし」に厳密一致する。したがって R2 一本で **3-way 比較**（既存 vs R2-生 vs R2-EMA）が成立し、施策5・6 の寄与を個別に切り分けられる。5+6 が回帰した場合のみ追加アブレーション（例: cosine+EMA）を検討。

### 評価

- **枠組み再利用**: `temps/phase1_eval/eval_phase1.py`。held-out 3話者（学習除外）× 日本語5文。
- **指標**: CER（faster-whisper large-v3）／ECAPA spk_cos／mel_std・peak（飽和 proxy）／生成時間。
- **推論設定**: §1 の注意どおり **sway_coef / step を再スイープ**（Phase 1 プリセット固定にしない）。少なくとも {euler16 sway=−1.0, euler16 sway=None, dopri5-25} を比較し、logit-normal 学習に最適な推論スケジュールを再特定する。
- **聴感 A/B**: 既存 378h と R2-EMA を主に比較。
- **評価用依存**: faster-whisper / jiwer / librosa / speechbrain は `uv pip install` で venv に追加（`pyproject.toml` 非管理）。

### 判定

CER 維持かつ ECAPA cos・mel 飽和・聴感のいずれかで有意改善なら採用。改善が確認できたら:
- 公開モデルカードと `webui.py` 既定チェックポイントの更新を検討。
- `CLAUDE.md` / `README.md` にレシピ変更（logit-normal 既定化・EMA）を反映。
- 本文書と [architecture-improvement-research.md §7](architecture-improvement-research.md) に Phase 2 結果セクションを追記（Phase 1 §9 と同形式）。

---

## 5. 変更ファイル一覧（実装フェーズの着手点）

| ファイル | 施策 | 変更内容 |
|---|---|---|
| `config.py` | 5,6 | `TrainConfig` に `timestep_sampling` / `logit_normal_m` / `logit_normal_s` / `use_ema` / `ema_decay` / `ema_warmup` |
| `models/flow_matching.py` | 5 | `CFMDecoder.__init__` に3引数（未知値は `ValueError`）、`compute_loss` に分岐 |
| `models/model.py` | 5 (+7) | `StableTTS.__init__` で `CFMDecoder` に3値を渡す（7 採用時: projection head 追加） |
| `train.py` | 5,6 (+7) | サンプリング値を渡す／EMA 生成・更新・保存・レジューム復元（7 採用時: 補助損失加算） |
| `utils/ema.py` | 6 | 新規 `EMA` クラス |
| `models/estimator.py` | 7 | （任意）中間表現の露出 |

施策5+6 は実装・検証済み（5観点の敵対的レビュー + スモークテスト通過）。すべてデフォルト値（`timestep_sampling="cosine"` / `use_ema=False`）で既存挙動をビット維持し、推論（`api.py`/`webui.py`）は完全不変。`utils/load.py` は変更不要（§2-3）。施策7 の head を採用する場合のみ推論との互換に別途配慮する。

---

## 6. リスク・留意点

1. **sway_coef との相互作用（最重要）**: logit-normal 学習後は `sway_coef=-1.0`（cosine 一致 warp）が最適でなくなる可能性。推論スケジュールの再スイープを評価に必ず含める。
2. **EMA decay の短期学習向け調整**: 総ステップが少ないため高 decay は無効。warmup か低め decay に。
3. **効果見積りは画像/英中 TTS 由来**: 日本語 44.1kHz mel での検証値は無い（[調査の限界 §8](architecture-improvement-research.md)）。R2 の自前 A/B で確認してから Phase 3 へ。
4. **施策7 は投機的**: ECAPA 置換型の類似性転移には否定的報告もある（arXiv:2506.20190）。プロトタイプで効果が出なければ本採用しない。5+6 を優先。
