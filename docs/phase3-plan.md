# Phase 3 計画 — ゼロショット話者類似性の改善（日本語特化 StableTTS v2）

作成日: 2026-07-07（3視点の文献調査を反映して改訂）/ 対象: StableTTSEx（upstream v1.1、31.6M）/ 親文書: [architecture-improvement-research.md](architecture-improvement-research.md) §4,6,7 / 前フェーズ: [phase2-plan.md §7](phase2-plan.md)（Phase 2 は不採用）

> **本フェーズは §13 でクローズ済み（TLA-SA / MRTE とも不採用）。研究アーク全体（Phase 1→2→3）の俯瞰は [research-summary.md](research-summary.md) を参照。**

## 0. 位置づけと原則

Phase 1（推論のみ改善・採用）と Phase 2（レシピ変更 = logit-normal + EMA・**不採用**）を経て、**伸びしろは「ゼロショット話者類似性」にある**と定量+聴感で確定した。Phase 3 はこの一点に集中する。

- **原則**: 既存重みを **`strict=False` 部分ロード**して moe-speech 378h から継続学習し、`checkpoint_0` 系の資産を引き継ぐ。新規モジュールは **zero-init ゲート**で「追加直後は現行とビット一致」にして安全に立ち上げる（Phase 1 の「デフォルト無効＝既存挙動一致」設計思想を踏襲）。
- **Phase 2 の教訓**: 効果は日本語 44.1kHz mel で未検証。複数施策の同梱は切り分け困難。→ **第一弾は 1 施策に絞り、A/B で確認してから次段へ**。

## 1. 現状の正確な把握（コードで確認）

- **日本語アクセント記号（`↑↓`）+ 句境界は既に音素列に挿入済み**（`text/japanese.py` L116-123、`text/symbols.py` L44）。研究doc §5 の「捨てている・配管工事だけ」は**誤り**。当初本命に想定した「アクセント埋め込み」は撤回。既知のカタカナ語アクセント不安定は g2p 推定精度の問題で埋め込みでは直らない。
- **pooled style vector ボトルネックの所在**（3エージェントがコードで確認）:
  - `models/reference_encoder.py` `MelStyleEncoder.forward` の `temporal_avg_pool` が参照 mel を時間平均し **単一 256 次元ベクトル `c`** に潰す。
  - `models/model.py` でこの `c` が encoder / duration predictor / decoder の**全コンポーネントに同一ベクトルとして**渡る。
  - `models/estimator.py` / `diffusion_transformer.py` では `c` は各 `DiTConVBlock` の **`adaLN_modulation`（FiLM 変調）にしか入らない = 全時刻に同一 affine**。**参照 mel への cross-attention は存在しない**（attention は q=k=v=x の self のみ）。
- **反証（規模が一致）**: 同型構成の **DiFlow-TTS（LibriTTS 470h、global 埋め込み + DiT affine 変調）は SIM-o 0.45 止まり**で、著者自ら「SIM-o が低いのは話者条件付けが単純だから」と明記（[arXiv:2509.09631](https://arxiv.org/html/2509.09631v2)）。同じ 470-585h でも参照を**系列**で使う ZipVoice(123M) は 0.610、CAST-TTS(1360h) は 0.784。→ **単一ベクトル→参照フレーム列にするだけで大きく上がる**のが最も再現性のある知見。

## 2. ゴール

| 目標 | 指標 | 現状 → 目標 |
|---|---|---|
| **主: ゼロショット話者類似性の向上** | ECAPA spk_cos（held-out 3話者、dopri25） | 0.63-0.65 → **baseline 比 +0.03 以上**（特に長参照） |
| 副: 発音を悪化させない | CER（faster-whisper） | 0.001-0.003 を維持 |
| 副: 日本語韻律の維持 | 聴感 A/B、mel 飽和 | baseline 同等以上 |
| 成果物 | — | 「日本語特化 StableTTS v2」 |

## 3. 調査結論（3視点の総意、2026-07-07）

参照系列条件付け / 分離・補助損失 / 2025-2026 事例 の3視点で独立に調査し、同じ結論に収束した。話者類似性向けの候補比較:

| 施策 | 効果 | 推論互換 | 部分ロード | 実装 | 小規模実証 | 出典 |
|---|---|---|---|---|---|---|
| **TLA-SA**（補助損失で中間表現を事前学習話者埋め込みに整列） | +0.03〜0.06（実測） | **ビット不変** | ◎完全 | 小 | LibriTTS 585h ✓・収束2.9倍・F5型LM-freeでも実証 | [2511.09995](https://arxiv.org/html/2511.09995) |
| **MRTE**（参照mel系列への frame-level cross-attn。本命の構造解消） | 高 | 経路変更（zero-init で初期一致） | ○ | 中 | CAST-TTS 1360h が F5/MaskGCT/ZipVoice 超え | [2307.07218](https://arxiv.org/html/2307.07218v3) / [CAST-TTS](https://arxiv.org/html/2603.16280) |
| Perceiver 多トークン（IndexTTS2型、K個の話者トークン） | 中〜高 | 経路変更 | ○ | 中（小改造） | IndexTTS2 SS 0.87（大規模） | [2506.21619](https://arxiv.org/html/2506.21619v2) |
| in-context infilling（F5/E2/ZipVoice） | **最高** | 全面改造 | ✗ | 大 | 上限最高だが 60-100Kh 前提・378h で暗黙アラインメント不安定 | [F5](https://arxiv.org/html/2410.06885v1) |
| 事前学習埋め込みの単純置換（ECAPA等） | 小（頭打ち継続） | 容易 | ○ | 小 | 否定的（1.6B+級でしか成立） | [2506.20190](https://arxiv.org/abs/2506.20190) |

**重要な実装知見**:
- **zero-init tanh ゲート付き cross-attention**（StableVC の DualAGC、[2412.04724](https://arxiv.org/html/2412.04724)）なら追加直後の出力が現行とビット一致 → 構造変更でも低リスクに継続学習。
- コードに**未使用の `AttnMelStyleEncoder` が既にある** → Perceiver 多トークン案の足場。
- 事前学習埋め込みは「入力に差す」（否定的、[2506.20190](https://arxiv.org/abs/2506.20190)）のではなく **「TLA-SA の整列教師に使う」**のが安全（否定対象外）。**整列教師と評価指標は別系統にする**（教師 WavLM/CAM++、評価 ECAPA。テストに教えるバイアス回避）。

## 4. 戦略：二段構え（第一弾に TLA-SA を採用）

- **第一弾（本計画の実装対象）= TLA-SA**。推論経路ビット不変・pooled 温存・既存重みそのまま継続学習・実測 +0.03〜0.06。目標 +0.03 に届けば**推論を一切変えずに完了**。外部 SV エンコーダの教師導入のみで済む最小リスク。
- **第二弾 = MRTE**（TLA-SA で頭打ちの場合）。参照 mel を系列のまま cross-attention 参照し、pooling ボトルネックを構造的に解消。zero-init ゲートで継続学習。TLA-SA と**併用可**。
- **代替 = Perceiver 多トークン**（`AttnMelStyleEncoder` 足場、MRTE より小改造）。第二弾で MRTE のパラメータ増を嫌う場合の中間解。
- **in-context infilling は非推奨**（上限は最高だが既存 checkpoint を捨てるリビルド、378h で不安定）。将来の次期メジャー改訂で腰を据える場合のみ。

## 5. 第一弾 = TLA-SA の設計

REPA（画像 DiT で収束加速、[2410.06940](https://arxiv.org/abs/2410.06940)）の音声・話者版。flow-matching デコーダの中間表現を、凍結した事前学習話者エンコーダの埋め込みに **cosine 損失で整列**させる補助損失。話者情報が「初期 denoising step・浅い層」に偏在する観察に基づき、**層ごと・時刻ごとに動的加重**する。

### 損失

`L = L_CFM + λ · Σ_i w_i · L_i^SA`（`λ=0.5`）
- `L_i^SA`：第 i 層の中間表現 → time 平均プール → 層別 MLP 射影 → 凍結 SV 埋め込みとの **1 − cosine**
- `w_i`：denoising timestep 埋め込みから小 MLP + softmax で生成する層重み（`+ α·L_reg(w)` エントロピー正則化、`α=0.01`）
- 推論時はヘッド（層別 MLP・重み MLP）を**捨てる** → 推論経路は現行とビット不変

### 実装コンポーネント

| ファイル | 変更 |
|---|---|
| `config.py` `TrainConfig` | `use_tla_sa: bool=False`、`tla_sa_lambda: float=0.5`、`tla_sa_teacher: str`（"wavlm_sv" / "campplus"）、`tla_sa_alpha: float=0.01` を追加。既定 False で現行とビット一致 |
| `models/estimator.py` | 各 `DiTConVBlock` の出力 hidden を**学習時のみ**リストで返す経路を追加（`docs/phase2-plan.md §3` が特定済みの唯一の侵襲点。推論時は返さない＝経路不変） |
| `models/tla_sa.py`（新規） | 層別 projection head（MLP）+ timestep→層重み MLP + cosine 損失。学習時のみ生成 |
| `train.py` | 凍結 SV エンコーダをロード（教師）、ターゲット音声（または `z` スライス）から毎バッチ埋め込み抽出、`loss += λ·L_tla_sa`。**保存時にヘッドを除外**（`ema`/生 state_dict に混ぜない） |
| 依存 | 凍結 SV エンコーダ（**評価の ECAPA とは別系統**。WavLM-base-plus-sv か CAM++。`pyproject.toml` 非管理で `uv pip`/`pip` 導入、Phase 1/2 評価依存と同方針） |

### StableTTSEx 固有の注意

1. **デコーダが 6 層と浅い**（TLA-SA 原典は 9-12層）。層数が少ないほど「層別適応加重」の旨味は減る可能性。ただし SLG（実験D で 6層では層スキップが効かず）とは機序が別（層を**スキップ**せず**教師付け**するので浅さの影響は限定的と見込む）。効果が出なければ「単純平均加重」へのフォールバックも用意。
2. **整列教師と評価指標を分離**：教師 = WavLM系 or CAM++、評価 = ECAPA。同じにすると「テストに教える」バイアス。
3. **教師埋め込みの対象**：`collate_fn` が参照に使う `z`（ターゲット mel のランダムスライス）から抽出するか、ターゲット全体から抽出するか。実装時に決める（学習/推論の参照長分布の差に注意）。

## 6. 第二弾 = MRTE の設計（TLA-SA で不足した場合の概要）

- `models/reference_encoder.py`：`temporal_avg_pool` 前の系列表現を返す経路を追加（pooled `c` は CFG `fake_speaker`・global 用に併存）。
- `models/estimator.py` / `diffusion_transformer.py`：`DiTConVBlock` に **zero-init ゲート付き cross-attention**（query=noisy mel、key/value=参照 mel 系列）を追加。追加直後はビット一致。
- `models/model.py`：CFG に `fake_ref`（参照 key/value の null 差し替え）を追加。dataloader は参照 mel を系列のまま渡す（既存 `z` / Phase 1 の `ref_window_seconds` 経路を流用）。
- リーク対策：MRTE の「音素 query＝内容対応音色」設計、StableVC の「speaker embedding を key に連結」を踏襲し、韻律・内容のコピーを抑制。

## 7. 学習・評価プロトコル

- **学習**: 既存 japanese-378h（または upstream `checkpoint_0`）から `strict=False` 部分ロードで継続学習。moe-speech 378h、Tier1/2 高速化込み（実測 ~2.5h/15ep）。vast 2×RTX5090（インスタンス 43982092 維持）。
- **評価**: `temps/phase2_eval/eval_phase2_3way.py` を流用。**主指標 = ECAPA spk_cos**、副 = CER / mel 飽和 / 聴感。比較は baseline（japanese-378h）vs TLA-SA 版。推論は **dopri25 で揃える**（Phase 2 で確認した公平比較条件）。
- **参照長スイープ**を含める（2s/5s/9s）。TLA-SA は pooled のままなので長参照での伸びは限定的な想定だが、MRTE へ進む判断材料として測る。
- **判定**: CER 維持かつ spk_cos が baseline 比 +0.03 以上 → 採用（v2 候補）。届かなければ第二弾 MRTE へ。

## 8. リスク・留意点

1. **効果は日本語 44.1kHz mel で未検証**。第一弾 A/B で確認してから第二弾へ（Phase 2 と同じ慎重運用）。
2. **6 層デコーダの浅さ**（§5-1）。単純平均加重フォールバックを用意。
3. **教師エンコーダ選定**：WavLM-sv / CAM++ の導入容易性・日本語話者での妥当性を実装時に確認。評価 ECAPA との分離を厳守。
4. **保存時のヘッド混入**：TLA-SA ヘッドを生/EMA state_dict に混ぜない（`api.py` ロードは推論パラメータのみ）。保存時除外 or `strict=False`。
5. **MRTE へ進む場合**：pooled と系列の二重条件付けで韻律劣化の報告（[2210.16045](https://arxiv.org/abs/2210.16045)）。効果が出なければ pooled を段階的に外す or FACodec-lite（GRL）で分離。

## 9. 次アクション

1. **第一弾 = TLA-SA に決定**（2026-07-07、ユーザー合意）。
2. TLA-SA の実装詳細を詰める: 教師 SV エンコーダの選定（WavLM-sv vs CAM++）、中間表現の露出方法、ヘッド構造、教師埋め込みの対象（z スライス vs 全体）。
3. 実装 → 部分ロード継続学習（vast）→ dopri25 A/B（+参照長スイープ）→ 採否判定 → 必要なら第二弾 MRTE。

## 11. 第一弾 TLA-SA 実施結果（2026-07-08）— 主目標未達、第二弾 MRTE へ

WavLM-sv 教師で TLA-SA を実装・学習・評価した。**結論: 話者類似性は改善せず（spk_cos −0.009）。第二弾 MRTE へ進む。**

### 実装・学習

設計調査ワークフロー（GO_WITH_CONDITIONS）+ 実装レビュー（low 3件修正）で確定した設計を実装（§5）。教師=WavLM-base-plus-sv（512次元、vendor 不要）、upstream `checkpoint_0` から 15 epochs / cosine / EMA無（baseline japanese-378h と同条件で TLA-SA 有無だけを差にする）。教師埋め込みは `precompute_spk_emb.py` で 235,095 サンプル全てをオフライン事前計算。vast 2×RTX5090、実測 ~2.5h。成果物 `checkpoints/vast_run3/`。

学習は健全に完走: `tla_sa_loss` は 0.97 → ~0（ヘッドが整列を達成）、主タスク損失（diff/dur/prior）は baseline と同レンジで安定（λ=0.5 は過大でない）。

### 評価（dopri25 A/B、n=15）

| model | CER | spk_cos | mel_std | peak |
|---|---|---|---|---|
| baseline（既存378h） | 0.0013 | **0.6502** | 2.590 | 0.823 |
| tla_sa | 0.0199 | 0.6408 | 2.577 | 0.817 |

**spk_cos 差 = −0.009（目標 +0.03 に対し改善せず、むしろ微減）。CER も 0.001→0.020 と微悪化。** 話者別・文別でも改善は一貫しない。参照長スイープは win2 で tla_sa 0.652 > baseline 0.640（+0.012）と弱い兆候のみ（誤差レベル）。スクリプト・数値は `temps/phase3_eval/`（`eval_phase3_tla.py`, `results_tla.json`, `agg_tla.py`）。

### 解釈

- `tla_sa_loss` が ~0 に張り付いた ＝ **層別 projection head の表現力が十分で、デコーダ本体を弱くしか使わずに整列を達成できてしまった**。REPA が狙う「デコーダ本体への話者情報の押し込み」が起きず、下流 spk_cos に転移しなかった。
- WavLM は英語話者検証モデルで、**日本語話者への転移が弱い**（arXiv:2506.20190 の「事前学習埋め込みの転移は限定的」とも整合）。

### 判断

CAM++ 教師への差し替えや λ 増強も選択肢だが、TLA-SA は「補助損失で症状を教師付けする」間接策で上限が低い。**pooling ボトルネックを構造的に解消する第二弾 MRTE（§6）へ進む**（ユーザー決定 2026-07-08）。TLA-SA の実装（config フラグ・tla_sa.py・precompute）は残置（既定 False でビット一致、将来 CAM++ で再評価可能）。

---

## 12. 第二弾 MRTE 実装〜学習・評価完了（2026-07-08）— 不採用

設計調査ワークフロー（GO_WITH_CONDITIONS）+ 実装レビュー（confirmed 1件=CFG 経路テスト追加で対応）で、参照 mel 系列への cross-attention を実装した。**以下 §12 の実装記述は実装時点のスナップショット。学習・評価の結果は本節末尾の「MRTE 実施結果（2026-07-08）— 主目標未達」を参照（spk_cos −0.007 で不採用、Phase 3 は §13 でクローズ）。**

### 実装（Stage A/B/C 完了、実装時点 97 テスト通過 →〔Phase 3 クローズ設定テスト追加後〕最終 98 passed・ruff clean）

- **Stage A** `models/reference_encoder.py`: `MelStyleEncoder.forward(return_sequence=True)` で pool 前系列 `[B,gin,T_ref]` を返す（param 0、既定 byte-identical）。
- **Stage B** MRTE 本体:
  - `models/diffusion_transformer.py`: 新クラス `CrossAttention`（RoPE 無し、query=noisy mel hidden、key/value=参照 mel 系列、ref_mask を additive mask 化）。`DiTConVBlock` に `use_cross_attn` を追加し、True 時のみ `norm_cross`（0 param）・`cross_attn`・`cross_gate=Parameter(zeros(1,hidden,1))` を生成。self-attn 後・FFN 前に `x = x + cross_gate * cross_attn(norm_cross(x), ref_seq, ref_mask) * x_mask` を挿入。**cross_gate は zero-init**（conv_o は非 zero-init でゲート勾配を確保）。
  - `models/estimator.py` / `models/flow_matching.py`: `use_mrte` と `ref_seq`/`ref_mask` を全経路に透過。`cfg_wrapper` の uncond は `fake_ref`（null 参照）に落として CFG が話者を誘導。
  - `models/model.py`: `use_mrte`・`fake_ref`（use_mrte 時のみ生成）。学習 forward で参照系列を同一 `cfg_mask` で drop、`synthesise` で c 未指定時に参照系列を抽出。
  - `config.py`: **`ModelConfig.use_mrte: bool = False`**（TLA-SA と違い state_dict にキーが増えるアーキ設定）。
- **Stage C** `utils/load.py`: pretrained ブートストラップを strict=False 化（legacy checkpoint → MRTE モデルの部分ロード入口。missing/unexpected をログ）。本 resume は strict=True 据置。

### 保証された不変条件（テストで固定）

- `use_mrte=False`: state_dict キー・**param 数 31,644,545 不変**（現行完全維持）。
- `use_mrte=True`: full config **param 数 33,225,345**（+1,580,800）。
- **zero-init ゲートで baseline 重みを strict=False 部分ロード後の synthesise 出力が baseline と byte-identical**（cfg=1.0 直行経路・cfg=3.0 の CFG 経路の両方でテスト）。
- 配線ガード: cross_gate を非零にすると出力が変わる。

### 次アクション

`ModelConfig.use_mrte=True` に設定 → 既存 japanese-378h（vast の `checkpoints/checkpoint_14.pt`）から strict=False 部分ロードで継続学習（15ep、cosine、EMA無、TLA-SA off）→ dopri25 A/B（+参照長スイープ、MRTE は長参照で伸びる想定）→ 採否判定。**Stage D（複数参照 `get_reference_mel`）は第一版では未実装＝window/list 経路は pooled fallback**（長参照×MRTE を本格活用する場合の別フォローアップ）。

### 学習前の留意（open risk）

- 二重条件付け（pooled c + 系列 cross-attn）で内容・韻律リーク→CER 悪化の可能性（fake_ref CFG drop で緩和、効果不足なら pooled を段階的に外す）。
- 学習参照 z（random_slice の短スライス）が推論の長参照と長さ非対称。z が数フレームに退化すると MRTE の旨味が減る。
- cross_gate の立ち上がりを TensorBoard で監視（開かない兆候なら conv_o 小スケール init へ、ただし byte-identity は失う）。

### MRTE 実施結果（2026-07-08）— 主目標未達

japanese-378h（checkpoint_14）から MRTE cross-attn を zero-init で足して 15 epochs 継続学習（cosine / EMA無 / TLA-SA off、vast 2×RTX5090）。部分ロード検証はパス（missing 55=MRTE キーのみ・unexpected 0・cross_gate 全 zero スタート）。

**しかし cross_gate が 0.0006〜0.0009 までしか開かなかった**（zero-init から微増のみ）。これは open risk「立ち上がりが遅い」の的中。

**dopri25 A/B（n=15）:**

| model | CER | spk_cos | mel_std | peak |
|---|---|---|---|---|
| baseline（japanese-378h） | 0.0013 | **0.6502** | 2.590 | 0.823 |
| mrte | 0.0013 | 0.6432 | 2.572 | 0.816 |

spk_cos 差 = **−0.007（改善せず微減）**、CER 同等、話者別で全3話者 baseline≥mrte、参照長スイープでも MRTE 優位なし（full/win5/win2 すべて baseline≥mrte）。スクリプト `temps/phase3_eval/`（`eval_phase3_mrte.py`, `results_mrte.json`, `agg_mrte.py`）。

**原因**: japanese-378h からの継続学習ではモデルは既に pooled c で十分学習済みで、MRTE cross-attn を使うインセンティブが弱く（勾配が小さく）ゲートが開かない。生 Parameter zero-init ゲートは adaLN ゲート（Linear 出力）より立ち上がりが遅い。

---

## 13. Phase 3 総括（2026-07-08）— クローズ

| フェーズ | 施策 | 初期値 | spk_cos 差 | 結論 |
|---|---|---|---|---|
| Phase 2 | logit_normal + EMA | upstream→15ep | −0.020 | 不採用 |
| Phase 3-1 | TLA-SA（WavLM 教師・補助損失） | upstream→15ep | −0.009 | 不採用 |
| Phase 3-2 | MRTE（参照 cross-attention） | japanese-378h→15ep | −0.007 | 不採用 |

**結論: レシピ変更（Phase 2）も、後付けの構造変更（Phase 3 の TLA-SA / MRTE）も、この規模・条件では日本語ゼロショット話者類似性を改善できなかった。**

### 共通の失敗要因

3施策すべてに共通するのは「**既に pooled style vector で学習済みの重みに後付けした**」こと。

- TLA-SA: 補助損失の projection head が表現力十分で、デコーダ本体を弱くしか使わず整列を達成（下流に転移せず）。
- MRTE: モデルは pooled c で足りているため cross-attn を使うインセンティブが弱く、cross_gate が 0.0006 までしか開かなかった。

→ **pooled ボトルネックの緩和には「後付け継続学習」は原理的に向かない**（既存の到達点が局所最適で、新機構を使わない方向に落ち着く）ことが2回の実験で実証された。

### 今後の方向性（別途計画）

pooled ボトルネックを本当に解くには、以下のいずれか。いずれも大きめの投資で、別途計画として扱う:

1. **MRTE をスクラッチ学習**（upstream `checkpoint_0` から MRTE 込みで学習し、pooled 依存を作らせず最初から cross-attn を使わせる）。実装は既にあるので config 変更のみ。ただし日本語 378h を一から学習し直す。
2. **in-context infilling 全面移行**（F5/E2/ZipVoice 型。上限は最高だが既存 checkpoint を捨てるリビルド、378h で暗黙アラインメントが不安定というリスク）。
3. **データ拡張**（378h→数千時間規模。話者多様性がゼロショット類似性の上限を決めるため、レシピ・構造よりデータが効く可能性）。
4. **日本語アクセント高度化**（話者類似性ではなく日本語品質そのものの改善。記号方式は実装済みなので、音素ごと連続高低・核位置の埋め込み追加）。

実装済みの TLA-SA / MRTE コードは既定 False で残置（`use_tla_sa` / `use_mrte`、いずれも現行とビット一致）。スクラッチ学習や pooled 段階除去で再評価する際にそのまま使える。

---

## 14. 主要出典

- TLA-SA: https://arxiv.org/html/2511.09995 / REPA（系譜）: https://arxiv.org/abs/2410.06940
- pooled ボトルネックの反証 DiFlow-TTS: https://arxiv.org/html/2509.09631v2
- MRTE / Mega-TTS2: https://arxiv.org/html/2307.07218v3 / CAST-TTS: https://arxiv.org/html/2603.16280 / StableVC(DualAGC): https://arxiv.org/html/2412.04724
- Perceiver 多トークン IndexTTS2: https://arxiv.org/html/2506.21619v2 / XTTS: https://arxiv.org/html/2406.04904v1
- 小規模 in-context 実証 ZipVoice: https://arxiv.org/html/2506.13053v1 / F5-TTS: https://arxiv.org/html/2410.06885v1 / E2-TTS: https://arxiv.org/abs/2406.18009
- 事前学習埋め込み置換の否定的報告: https://arxiv.org/abs/2506.20190 / CAM++ を補強に使う成功例 CosyVoice3: https://arxiv.org/abs/2505.17589
- FACodec/GRL（NaturalSpeech3）: https://arxiv.org/abs/2403.03100
