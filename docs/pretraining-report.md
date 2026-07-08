# moe-speech 378時間による日本語特化事前学習 実行レポート

実施日: 2026-07-06 / 実行環境: vast.ai 2× RTX 5090（台湾リージョン） / 関連文書: [計画書](pretraining-plan.md)

## 結論

KdaiP/StableTTS v1.1 の多言語チェックポイント（中英日 600h）を初期値に、日本語のみ 377.8 時間で15エポックの継続事前学習を実施した。聴感評価では主目的だった**表現力が明確に向上**し、長文の韻律・読みの正確さも改善。ゼロショットの話者類似性は依然弱く、計画どおり対象話者（つくよみちゃん）での fine-tune に移行した。

| 指標 | 値 |
|---|---|
| 学習データ（フィルタ後） | 377.8 時間 / 235,095 発話 |
| 学習量 | 15 epochs = 55,170 steps（約2.5時間、平均 6〜7 it/s） |
| 学習本体の GPU 費用 | 約 $2.5 |
| モデル | 31.6M パラメータ（v1.1 構成を変更せず、checkpoint_0.pt 互換） |

## データパイプライン

```
moe-speech-plus 原本   473話者 / 395,170発話 / 621.4h（44.1kHz スタジオ収録演技音声）
  → フィルタ後         235,095発話 / 377.8h（UTMOS≥1.5・2系統ASR一致度≥0.90・1〜15秒）
  → mel 変換           235,095個（g2p 失敗 0 件）
```

- ホールドアウト5話者（cf7e3a79, 93dda15e, 224a42d8, 0d70cf5c, 3371a8ac）を学習から除外し、ゼロショット評価に使用
- **フィルタ設計の判断**: UTMOS は自然発話基準の指標で演技音声（感情・叫び・囁き）を系統的に低く採点するため（全体の中央値 2.13）、計画当初案の閾値 2.5〜3.0 は不採用。品質担保は anime-whisper × parakeet の文字起こし一致度（正規化編集比 ≥0.90）を主軸にし、テキストの正確性を守りつつ目的である「表現力」のデータを温存した
- テキストは anime-whisper 側の文字起こしを採用

## 学習設定

- 初期値: 本家 v1.1 `checkpoint_0.pt`（optimizer なし配置 → 重みのみロードで epoch 0 から）
- batch 32 × 2GPU（実効64 = upstream 構成）、lr 1e-4、cosine スケジューラ、warmup 200
- g2p は pyopenjtalk-plus フル補正（`use_vanilla=False` + onnxruntime）で学習・推論を統一
- バケット境界 [32,300,...,1300]（従来上限1000では長尺 4.74% が黙って除外されるため拡張）

## 聴感評価（本家 v1.1 との A/B、2026-07-06）

| 項目 | 判定 | 所見 |
|---|---|---|
| 表現力 | ▲ 向上 | 感情文での演技の乗りが改善。主目的を達成 |
| 長文の韻律 | ▲ 向上 | 安定性が向上。ただしカタカナ語・アルファベットのアクセントは不安定（g2p の未知語アクセント推定の限界） |
| 読みの正確さ | △ 微改善 | 「何」系の読み等、非劣化を確認 |
| ゼロショット類似性 | ▼ 課題 | つくよみちゃん参照で「似ていると言われれば似ている」水準。1発話参照の構造的限界 → fine-tune で解決へ |

評価サンプル: `temps/eval_compare_e14/`（目的別フォルダ、`base_*`=本家 / `new_*`=事前学習後。参照はつくよみちゃん2種 + ホールドアウト3話者）

## 立ち上げ時に解決した問題

| 症状 | 原因 | 対処 |
|---|---|---|
| ダウンロードが無進捗でハング | HF の Xet 転送バックエンドのコネクションストール | `HF_HUB_DISABLE_XET=1` + 自動リトライ + 10分無更新で kill するウォッチドッグ |
| DDP 初期化で SIGSEGV | コンシューマ GPU ホストの NCCL P2P 問題 | `NCCL_P2P_DISABLE=1`（`NCCL_IB_DISABLE=1` も併用） |
| batch 64 で OOM | 長尺バケット（〜1291フレーム）の attention メモリ | batch 32 × 2GPU に変更 |
| 長尺データ 4.74% が黙って除外 | バケット境界の上限 1000 フレーム | 境界を 1300 まで拡張（train.py に反映済み） |
| 前処理が2並列固定 | `Pool(processes=2)` ハードコード | `PREPROCESS_WORKERS` 環境変数化 → 48並列で約20分 |

運用上の教訓: scp のポート指定は `-P`（大文字。ssh の `-p` と異なる）。学習終了後に DDP プロセスが `destroy_process_group` でハングすることがある（成果物には無害）。

## 成果物の場所

- チェックポイント: `checkpoints/vast_run1/checkpoint_0〜14.pt`（ローカル回収済み）
- 損失曲線: `runs_vast_run1/runs/` → `uv run tensorboard --logdir runs_vast_run1/runs`
- vast.ai 側: インスタンス 43982092 の `/data`（450GB ボリューム）に前処理済みデータ・環境一式

## つくよみちゃん fine-tune（M7、2026-07-06 完了）

事前学習後の checkpoint_14 を初期値に、つくよみちゃんコーパス100発話で fine-tune を実施した。

- 設定: batch 16 × 2GPU、lr 1e-4、warmup 10、401 epochs（約4,000 steps、実時間約30分）。loss 2.70 → 1.44
- エポック別聴感比較（epoch 100/150/200/400）: **epoch 400 は声のかすれ（過学習による音質劣化）が出る**。epoch 100 は良好、**epoch 200 が類似性と音質のバランス最良 → 採用**
- 採用モデル: `checkpoints/tsukuyomi_ft200.pt`（= ft epoch 200）。`webui.py` の既定チェックポイントはこのファイル
- 全 ft チェックポイント: `checkpoints/ft_tyc/checkpoint_{100,150,200,300,400}.pt`、評価サンプル: `temps/eval_ft_tyc/`
- 再現用: フィルタスクリプト `recipes/moe_speech_plus_filelist.py`、ホールドアウト話者 `filelists/holdout_speakers.txt`

## その後の研究フェーズ（2026-07-08 時点でクローズ）

本レポートの事前学習・fine-tune の後に、推論改善・レシピ再学習・構造変更再学習の**研究3フェーズ**を実施し、いずれも一巡してクローズした。全体の俯瞰は [research-summary.md](research-summary.md)、Phase 3 の詳細は [phase3-plan.md](phase3-plan.md) §11–13 を参照。

1. ~~つくよみちゃん fine-tune~~ → 完了、epoch 200 を採用
2. ~~HF 公開~~ → 完了（事前学習 / ft / baseline の3モデルを public 公開）
3. ~~Phase 1（推論のみ改善）~~ → 完了・**一部採用**。[architecture-improvement-research.md](architecture-improvement-research.md) にまとめ。Sway Sampling（低ステップで dopri5 品質・約10倍高速）と CFG rescale（推奨0.7）を実装し webui 既定に採用（既定無効でビット一致・既存チェックポイント互換）。SLG（Skip Layer Guidance）は定量評価で全条件悪化のため不採用・実装のみ残置。BigVGAN v2（MIT）ボコーダも追加
4. **長参照の運用検証** — 複数参照窓平均（`ref_window_seconds`）として実装し評価済み。ホールドアウト参照 ≤9秒では改善なし（30〜60秒級の長参照でしか効かない見込み。検証データ不足が限界）。既定無効のオプションとして温存
5. **カタカナ語・英字の正規化** — 英字→カタカナ読み変換をテキスト前処理に追加する改善課題（未着手）
6. ~~Phase 2（レシピ再学習：logit-normal timestep + EMA）~~ → 完了・**不採用**。upstream checkpoint_0 から15ep、dopri25 公平比較（n=15）で発音は非劣化（CER baseline 0.0013）だが、話者類似性が3話者一貫して劣化（spk_cos baseline 0.6502 → 0.628〜0.630、約 −0.020）。聴感 A/B でも差を体感できず、安価なレシピ改善では品質が上がらないことを実証
7. ~~Phase 3（構造変更再学習：TLA-SA / MRTE）~~ → 完了・**両方不採用でクローズ**。根本原因は pooled style vector ボトルネック（参照 mel を単一 256 次元ベクトルに潰し、estimator の adaLN affine にしか入らず参照系列への cross-attention が無い）。TLA-SA（補助話者整列損失、upstream→15ep）は spk_cos −0.009、MRTE（参照 mel 系列への cross-attention、japanese-378h→15ep）は −0.007 で、いずれも「既に pooled style vector で学習済みの重みに後付け継続学習した」ため新機構を使わない方向へ収束（MRTE の cross_gate は 0.0006〜0.0009 までしか開かず）。実装は既定 False で残置し将来のスクラッチ学習で再利用可能。詳細は [phase3-plan.md](phase3-plan.md) §11–13
8. **今後の方向性（別途計画）** — pooled ボトルネックを本当に解くには後付け継続学習でなく投資が要る：MRTE をスクラッチ学習 / in-context infilling 全面移行（F5・E2・ZipVoice 型）/ データ拡張（378h→数千時間）/ 日本語アクセント高度化。詳細は [research-summary.md](research-summary.md)
9. **インスタンスの後始末** — 成果物は全て回収済み。追加検証のため当面維持（2026-07-06 判断、$0.9/h ≈ $22/日）。検証完了後に破棄判断
