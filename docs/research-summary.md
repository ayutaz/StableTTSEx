# StableTTSEx 研究サマリ（Phase 1 → 3 の全アーク）

作成日: 2026-07-08 / 対象: StableTTSEx（upstream v1.1、31.6M パラメータ、日本語特化）/ 位置づけ: 本ドキュメントは研究アークのエントリポイント（エグゼクティブサマリ）。各詳細は末尾の[詳細ドキュメントへのリンク](#詳細ドキュメントへのリンク)から辿る。

## 要旨

日本語継続事前学習（moe-speech 378h）で表現力・韻律を改善した日本語特化 StableTTS に対し、**推論・学習の両面から品質をさらに引き上げられるか**を3フェーズで検証した研究の1枚俯瞰である。結論として、**再学習不要の推論改善（Phase 1）は採用**され、webui の既定に組み込まれた（約10倍高速化を含む）。一方、**レシピ再学習（Phase 2）と構造変更再学習（Phase 3-1・3-2）はいずれも不採用**となり、Phase 3 はクローズ済み。後者2フェーズは「既存の到達点（pooled style vector で学習済みの重み）に新機構を後付けしても、モデルはそれを使わない方向に収束する」という**再現された否定的結果**をもたらした。これ自体が「安いレシピ改善・後付け継続学習では話者類似性の上限は破れない」という価値ある知見である。

主要な不変条件（既存チェックポイント互換の絶対条件）:
- 推論・baseline パラメータ数 = **31,644,545**
- n_vocab = len(symbols) = **401**
- v1.1 デフォルト: 44.1kHz / 128 mel(slaney) / encoder 3層・decoder 6層

## 採用/不採用の一覧表

| フェーズ | 施策 | 種別 | spk_cos 差（baseline 比） | 結論 |
|---|---|---|---|---|
| **Phase 1** | Sway Sampling / CFG rescale | 推論のみ（再学習不要） | — | **採用**（webui 既定に統合） |
| Phase 1 | SLG（Skip Layer Guidance） | 推論のみ | — | 不採用（全条件悪化・実装のみ残置） |
| **Phase 2** | logit-normal timestep + EMA | レシピ再学習 | **−0.020** | 不採用 |
| **Phase 3-1** | TLA-SA（補助話者整列損失） | 構造変更再学習 | **−0.009** | 不採用 |
| **Phase 3-2** | MRTE（参照 mel 系列への cross-attention） | 構造変更再学習 | **−0.007** | 不採用 |

主指標は ECAPA spk_cos（held-out 3話者・日本語文・**dopri25 で揃えた公平比較**）。baseline の spk_cos = **0.6502**。CER は Phase 2（0.0027）・MRTE（0.0013）で 0.001〜0.003 圏を維持し発音を悪化させていない。TLA-SA のみ 0.0013→0.0199 と微悪化した。

## Phase 1 で採用した推論改善（既存チェックポイント互換）

すべてデフォルト無効＝既存挙動とビット一致で追加した推論オプションのうち、以下2つを採用した。既存チェックポイントをそのまま使え、再学習は不要。

- **Sway Sampling**（`sway_coef`、euler 等の固定ステップソルバー専用）: `sway_coef=−1.0` で学習時の cosine スケジューラに一致する。**euler16 + sway−1.0 が dopri5-25 とほぼ同品質（mel-L1=0.078）でありながら約10倍高速（0.70s vs 7.08s）**。低ステップでも dopri5 品質を得られるのが最大の成果。
- **CFG rescale**（`cfg_rescale`、推奨 0.7）: 過剰 CFG による飽和を抑制し、高 cfg 時の話者類似性劣化も回復する安全な小改善。

温存（実装済み・デフォルト無効）: `cfg_interval`（interval CFG）、複数参照窓平均（`ref_window_seconds` + `ref_audio` の `list[str]` 対応）。
不採用: **SLG（Skip Layer Guidance）** は定量評価で全条件悪化（6層デコーダでは層スキップが粗すぎる）。実装のみ残置し既定無効。

**webui 既定サンプリング: solver=euler / step=16 / sway_coef=−1.0 / cfg_rescale=0.7**（既定チェックポイントは tsukuyomi_ft200.pt、既定ボコーダは bigvgan）。

## Phase 2・3 でわかったこと（否定的結果の価値）

Phase 2・3 の3施策はいずれも spk_cos を **わずかに劣化**させ、不採用となった。共通の失敗要因は明確である。

- **根本原因は pooled style vector ボトルネック**（3エージェントがコードで確認）: `MelStyleEncoder.forward` の temporal_avg_pool が参照 mel を単一 256 次元ベクトル `c` に潰し、estimator の adaLN affine（FiLM 変調・全時刻同一）にしか入らない。**参照 mel への cross-attention が存在しない**。同型の DiFlow-TTS（470h, global 埋込 + affine）が SIM-o 0.45 止まりである一方、参照を系列で使う ZipVoice(585h) 0.61 / CAST-TTS(1360h) 0.78 という反証がこれを裏づける。
- **Phase 2（logit-normal + EMA）**: upstream checkpoint_0 から 15ep。dopri25 公平比較で spk_cos が **3話者一貫して −0.020** 劣化。聴感 A/B でも「差を体感できない」と確認。**安いレシピ改善では品質が上がらない**ことを実証。
- **Phase 3-1（TLA-SA）**: デコーダ中間表現を凍結事前学習話者埋込（WavLM-base-plus-sv）に cosine 整列させる補助損失。spk_cos −0.009。原因は tla_sa_loss が ~0 に張り付いたこと＝層別 projection head が表現力十分で**デコーダ本体を弱くしか使わずに整列を達成**し、下流に転移しなかった。加えて WavLM は英語 SV で日本語話者転移が弱い。
- **Phase 3-2（MRTE）**: 参照 mel を系列のまま cross-attention 参照（zero-init ゲート付き）。japanese-378h checkpoint_14 から部分ロードで 15ep。spk_cos −0.007。原因は **cross_gate が 0.0006〜0.0009 までしか開かなかった**こと＝継続学習ではモデルは既に pooled `c` で十分学習済みで、cross-attention を使うインセンティブが弱い。

**核心**: 3施策に共通する失敗要因は「**既に pooled style vector で学習済みの重みに機構を後付けした**」こと。既存の到達点が局所最適であり、新機構を使わない方向へ収束する。→ **pooled ボトルネックの緩和には「後付け継続学習」は原理的に向かない**ことを2回の実験で実証した（価値ある否定的結果）。

## 現在の状態

- **config は推論安全（既存チェックポイントを strict ロード可能）**: `ModelConfig.use_mrte = False`、`TrainConfig.use_tla_sa = False`、`timestep_sampling = "cosine"`、`use_ema = False`。
- **実装済みだが既定 False で残置**（将来のスクラッチ学習で再利用可能・ビット一致）: `models/tla_sa.py`、`MelStyleEncoder.forward(return_sequence=True)`、`diffusion_transformer.py` の CrossAttention / cross_gate、estimator・flow_matching・model への `use_mrte`/`ref_seq`/`ref_mask` 透過、`precompute_spk_emb.py`、`api.py` の per-checkpoint `tts_model_config`。
- **テスト**: 98 passed / ruff clean。MRTE full config のパラメータ数 golden = **33,225,345**（+1,580,800）、zero-gate byte-identity（cfg=1.0/3.0）、state_dict superset、Phase 3 クローズ config の推論安全性（use_mrte False / use_tla_sa False / cosine / EMA無）を固定。
- **成果物**: 採用 = tsukuyomi_ft200.pt（webui 既定 FT）、japanese-378h checkpoint_14（HF: ayousanz/stable-tts-v1.1-japanese-378h）。不採用の研究成果物 = vast_run2(Phase2) / vast_run3(TLA-SA) / vast_run4(MRTE)。

## 今後の方向性（別途計画）

pooled ボトルネックを本当に解くには「後付け継続学習」ではなく、より大きめの投資が要る。候補は4案。

1. **MRTE をスクラッチ学習**: upstream checkpoint_0 から MRTE 込みで学習し、pooled 依存を作らせない。実装は流用可・config 変更のみ。
2. **in-context infilling 全面移行**: F5/E2/ZipVoice 型。上限は最高だが既存 checkpoint を捨てるリビルドで、378h では暗黙アラインメントが不安定。
3. **データ拡張**: 378h → 数千時間。話者多様性がゼロショット類似性の上限を決める。
4. **日本語アクセント高度化**: 話者類似性ではなく日本語品質の改善。記号方式は実装済みで、音素ごと連続高低・核位置の埋め込み追加が次段。

## 詳細ドキュメントへのリンク

- [architecture-improvement-research.md](architecture-improvement-research.md) — アーキテクチャ改善調査（現状把握・SD3.5 移植可否・Phase 1 の A/B。研究の起点）
- [phase2-plan.md](phase2-plan.md) — Phase 2 計画（logit-normal + EMA、不採用の記録）
- [phase3-plan.md](phase3-plan.md) — Phase 3 計画（TLA-SA / MRTE、両不採用の記録）
- [pretraining-plan.md](pretraining-plan.md) — 日本語継続事前学習の計画
- [pretraining-report.md](pretraining-report.md) — 日本語継続事前学習のレポート
