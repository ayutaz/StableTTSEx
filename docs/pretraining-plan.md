# StableTTS 日本語事前学習 計画書

作成日: 2026-07-06 / 対象リポジトリ: StableTTSEx (upstream v1.1 ベース)

## 1. 背景

本家 KdaiP/StableTTS v1.1 の事前学習チェックポイント（checkpoint_0.pt、31.6M パラメータ）は中国語・英語・日本語あわせて約600時間で学習されている。日本語はその一部でしかなく、2026-07-06 に実施した評価（moe-speech 3話者 + つくよみちゃんコーパス参照、計32サンプル生成）で以下を確認した。

| 評価項目 | 結果 |
|---|---|
| 発音・アクセント | 及第点（大きな誤読なし） |
| 表現力（演技・感情の幅） | 低い |
| ゼロショット話者類似性（つくよみちゃん参照） | 低い |
| 分布外話者参照時の韻律 | イントネーションに不自然な箇所あり |

StableTTS は話者ベクトル（MelStyleEncoder 出力）が duration predictor と decoder 全体に条件付けされる構造のため、参照話者が学習分布外だと音色だけでなく韻律も崩れる。表現力・類似性・韻律頑健性の3点はいずれも「日本語データの量と話者多様性」の不足に起因し、fine-tune（少量データ）では改善できない。

過去の類似試行（参考）:

- [ブログ記事 (2024-06-22)](https://ayousanz.hatenadiary.jp/entry/2024/06/22/212735): v1.0 英語チェックポイント + つくよみちゃん約100発話の fine-tune。ベースが日本語未学習・v1.0 には音質を下げる既知バグあり
- [stable-tts-moe-speech-1 / -2](https://huggingface.co/ayousanz/stable-tts-moe-speech-2) (2024-06-26): moe-speech での v1.0 時代の事前学習。アーキテクチャが異なり（-2 は enc8/dec8 の 48M 構成）現行 v1.1 コードとは非互換のため重みは再利用不可。「moe-speech で学習が回る」実績としてのみ参照

## 2. ゴール

**moe-speech-plus（日本語623時間・473話者・スタジオ品質演技音声）で checkpoint_0.pt から継続事前学習し、日本語特化チェックポイントを作る。**

成功基準（学習前に同一条件でベースラインを採取し比較する）:

1. **表現力**: 感情の乗った参照音声を与えたときの表現追従が本家より改善している（聴感 A/B）
2. **ゼロショット類似性**: つくよみちゃん参照の合成音声と本人収録音声の話者類似度（ECAPA-TDNN 等の話者埋め込み cos 類似度）がベースラインを上回る
3. **韻律頑健性**: 学習に使っていない話者（ホールドアウト）を参照してもイントネーションが破綻しない
4. **非劣化**: 発音・アクセントの正確さ（現状の強み）が落ちていない

明示的なトレードオフ（許容する劣化）: 中国語・英語の合成能力は失われる（日本語専用化）。

## 3. 決定事項

| 項目 | 決定 | 根拠 |
|---|---|---|
| 学習方式 | **A案: checkpoint_0.pt 初期化の継続事前学習**（ゼロからではない） | 同じ計算量でゼロから以上の到達点になりやすく、収束も速い。ゼロから（B案）は A の結果に初期値由来の問題が疑われた場合のみ再検討 |
| データ | **moe-speech-plus**（44.1kHz 保持） | v1.1 の 44.1kHz 構成に一致。UTMOS・2系統ASR文字起こし付きで品質フィルタが可能 |
| 不使用データ | moe-speech-20speakers-ljspeech | 22.05kHz に変換済みで帯域不足のため事前学習には不適 |
| g2p | **pyopenjtalk-plus フル補正（use_vanilla=False）+ onnxruntime 追加** | 学習・推論で統一するため分布ミスマッチなし。「何」系誤読修正とアクセント改善（検証: 誤読しやすい16文中4文で改善、劣化なし）。既存チェックポイント向けの use_vanilla=True 制約は自前学習により不要になる |
| 実行環境 | **vast.ai の 2× RTX 5090**（2026-07-06 API 調査で決定） | DLPerf/$ = 542 で全 GPU 中最良（4090 の1.7倍、A100/H100 の3倍超）。2枚構成は GPU 単価が 1枚時とほぼ同じ（~$0.35/GPU/h）でコスト同等のまま時間を半減できる。train.py は元々 2GPU DDP 前提の作りで変更最小。VRAM 32GB/枚でバッチも増やせる。API キーは `.env`（VAST_API_KEY、git 管理外） |
| ストレージ | **インスタンスディスク 500GB 以上 + チェックポイントの HF 同期**。ネットワークボリュームは vast.ai 全体でオファー0件のため使用不可（2026-07-06 時点）。マシンローカルボリュームはインスタンス作成直前に同一マシン上へ作成・アタッチ | データ原本は HF にあり再取得可能（データセンター回線なら 200GB が30分前後）。永続化が必要なのは学習途中のチェックポイントのみで、これは HF の非公開リポジトリへ定期同期する |
| モデル構成 | ModelConfig / MelConfig は**変更禁止**（デフォルト維持） | checkpoint_0.pt 初期化と既存ボコーダ（44.1kHz）互換のため |

## 4. マイルストーン

### M0: 環境準備 ✅（2026-07-06 完了）
- [x] `pyproject.toml` を `pyopenjtalk-plus[onnxruntime]` に変更し `uv sync`（onnxruntime 1.27.0）
- [x] `text/japanese.py` の `use_vanilla=True` を `False` に変更（コメントも更新）
- [x] `train.py` 冒頭の `CUDA_VISIBLE_DEVICES` ハードコードを `setdefault` に変更（外部からの指定を尊重）
- [x] 完了条件: 「何ですか→なん」「何の日→なんの」「何時のアクセント」の3文で plus 補正の有効化を確認済み（onnx 有効な検証環境の出力と完全一致）

### M1: データ取得（vast.ai インスタンス上で実施）✅（2026-07-06 完了）
- [x] インスタンス **43982092**（2×RTX 5090 / 144コア / 台湾 / 約$0.9/h）を作成。450GB ローカルボリュームを `/data` にアタッチ
- [x] リポジトリは private のため git bundle 経由で `/data/StableTTSEx` に配置。uv sync 済み（torch 2.8.0+cu128、GPU 2枚認識、g2p 出力はローカル Windows と完全一致）
- [x] moe-speech-plus ダウンロード完了（481ファイル・**143GB**）。HF の Xet 転送がハングする問題があり、`HF_HUB_DISABLE_XET=1` + 自動リトライ + 10分無更新でkillするウォッチドッグ構成で解決
- [x] 展開完了: **473話者 / 395,170発話 / 621.4時間**（zip は展開後削除）
- 学習初期値 checkpoint_0.pt とボコーダ2種も配置済み

### M2: フィルタリングと filelist 生成 ✅（2026-07-06 完了）
- [x] 統計に基づき閾値を **speechMOS ≥ 1.5 / ASR一致度 ≥ 0.90 / 1〜15秒** に決定
  - 計画時の「UTMOS 2.5〜3.0」は不採用。UTMOS は自然発話基準の指標で演技音声（感情表現・叫び・囁き）を系統的に低く採点するため（分布の中央値2.13）、高い閾値は目的の「表現力」をもつデータを削る。MOS は下位ジャンク除去（1.5未満 = 約12%）に留め、正確性は2系統 ASR（anime-whisper vs parakeet）の正規化一致度 0.90 で担保する
- [x] ホールドアウト5話者: cf7e3a79, 93dda15e, 224a42d8, 0d70cf5c, 3371a8ac（`filelists/holdout_speakers.txt`）
- [x] filelist.txt 生成: **235,095発話 / 377.8時間**（テキストは anime-whisper 側を採用）

### M3: 前処理（mel 化）✅（2026-07-06 完了）
- [x] `preprocess.py` を CPU 48並列で実行（`PREPROCESS_WORKERS` 環境変数を新設）。**235,095 mel 生成 = filelist と完全一致、g2p 失敗ゼロ**、約20分で完了

### M4: 学習立ち上げ実測 ✅（2026-07-06 完了）
- [x] batch 64 は長尺バケットで OOM → **batch 32 × 2GPU（実効64 = upstream 構成）** に決定
- [x] DDP 初期化の SIGSEGV はコンシューマ GPU ホストの NCCL P2P 問題 → **`NCCL_P2P_DISABLE=1`** で解決
- [x] バケット境界を [.., 1300] に拡張（上限1000のままだと長尺 4.74% が黙って除外されるため。train.py に反映済み）
- [x] 実測: 平均 6〜7 it/s、3,678 steps/epoch ≈ 10分/epoch、VRAM ピークは 32GB 内に収束

### M5: 本学習 ✅（2026-07-06 完了）
- [x] **15 epochs 完走（約2.5時間、学習本体の GPU 費用 ≈ $2.5）**。checkpoint_0.pt 初期化（optimizer なし → else 分岐で重みのみロード、epoch 0 から）
- [x] checkpoint_0〜14.pt + optimizer 全15組を確認、チェックポイントはローカル `checkpoints/vast_run1/` に回収済み
- [x] TensorBoard ログは `runs_vast_run1/runs/` に回収（`uv run tensorboard --logdir runs_vast_run1/runs`）
- 備考: 学習終了後に DDP プロセスが destroy_process_group でハングする（既知の挙動、成果物には無害）。監視スクリプトの scp は `-P`（大文字）が必要という教訓

### M6: 評価 ✅（2026-07-06 完了）
- [x] M4 と同一条件で再評価: 既存3評価セットの再生成、つくよみちゃん A/B（正解音声比較）、ホールドアウト話者ゼロショット
- [x] 定量: 話者類似度 cos-sim、UTMOS。エポック違いのチェックポイント数点を比較して最良を選定
- 完了: §2 の成功基準は表現力・韻律・非劣化を達成、ゼロショット類似性は事前学習だけでは不足（fine-tune で解決）。詳細は [実行レポート](pretraining-report.md)

### M7: 後続タスク ✅（2026-07-06〜07 完了）
- [x] つくよみちゃん fine-tune（100発話 → epoch 200 を採用: `checkpoints/tsukuyomi_ft200.pt`）
- [x] HF へのアップロード（public 公開: [事前学習](https://huggingface.co/ayousanz/stable-tts-v1.1-japanese-378h) / [ft](https://huggingface.co/ayousanz/stable-tts-v1.1-japanese-378h-tsukuyomi-ft) / [baseline](https://huggingface.co/ayousanz/stable-tts-v1.1-tsukuyomi-ft-baseline)。license: other / moe-speech-terms）
- [x] README / CLAUDE.md にモデルカード相当の情報（データ・フィルタ条件・g2p 設定）を記録

### 研究フェーズ（2026-07-06〜07-08、事前学習の後続）✅ クローズ済み
事前学習完了後、アーキテクチャ改善の調査を [architecture-improvement-research.md](architecture-improvement-research.md) にまとめ、3フェーズを実施・評価してクローズした。全体像は [research-summary.md](research-summary.md)（研究サマリのエントリポイント）、Phase 3 の総括は [phase3-plan.md §13](phase3-plan.md) を参照。

- **Phase 1（再学習不要の推論改善）= 一部採用**: Sway Sampling（euler 等の固定ステップソルバー専用、sway_coef=−1.0 で低ステップでも dopri5 品質・約10倍高速）と CFG rescale（過剰 CFG の飽和抑制、推奨 0.7）を webui 既定に採用。SLG（Skip Layer Guidance）は6層デコーダで悪化し不採用。ボコーダは BigVGAN v2（MIT）を追加。デフォルト無効時は既存挙動とビット一致で既存チェックポイント互換。
- **Phase 2（レシピ再学習 = logit-normal timestep + EMA）= 不採用**: upstream checkpoint_0 から 15ep。dopri25 公平比較で発音（CER）は悪化させないが、話者類似性（ECAPA spk_cos）が baseline 0.6502 に対し全3話者一貫して −0.020 劣化。聴感 A/B でも差を体感できず。
- **Phase 3（構造変更再学習）= TLA-SA / MRTE の両方とも不採用でクローズ**: pooled style vector ボトルネック（MelStyleEncoder が参照 mel を単一 256 次元ベクトルに潰し FiLM 変調にしか入れない）の解消を狙った。3-1 TLA-SA（補助話者整列損失）は spk_cos −0.009、3-2 MRTE（参照 mel 系列への cross-attention）は −0.007 で、いずれも改善せず。共通の失敗要因は「既に pooled style vector で学習済みの重みに後付けした」こと（新機構を使わない方向に収束）。→ pooled ボトルネックの緩和には後付け継続学習は原理的に向かないことを実証（価値ある否定的結果）。実装済みの TLA-SA / MRTE コードは既定 False で残置（ビット一致、将来のスクラッチ学習で再利用可能）。

## 5. 懸念事項とリスク

| # | 懸念 | 影響 | 対策 |
|---|---|---|---|
| 1 | ASR 文字起こしの誤り | 誤ったテキスト-音素ペアで学習が汚染される | 2系統 ASR の一致フィルタ（M2）。それでも残る分は量（数十万発話）で希釈される前提 |
| 2 | ライセンス（moe-speech は著作権法30条の4の情報解析目的限定・再配布禁止） | 公開方法を誤ると規約違反 | データ本体は再配布しない。学習済みモデルの公開は過去実績と同様 **gated + 用途申告制**。商用利用可否はライセンス原文を再確認してから判断 |
| 3 | NSFW 系音声の混入（Not-For-All-Audiences タグ） | 生成モデルの挙動・公開時の印象 | moe-speech 本体は不適切音声の除去済みとされるが、UTMOS フィルタに加えテキスト側の簡易フィルタ追加を M2 で検討 |
| 4 | vast.ai インスタンスの突然の停止（ホスト都合・障害） | 学習中断・インスタンスディスク上のデータ喪失 | 自動レジューム + チェックポイントを epoch ごとに HF へ同期。データ原本は HF から再取得可能。ネットワークボリュームはオファー0件で使えないため（2026-07-06 時点）この同期戦略で代替 |
| 5 | VRAM 不足（5090 は 32GB/枚、最長15秒 ≈ 1290 mel フレーム） | OOM で学習停止 | バケットサンプラーで同長がまとまるため長尺バッチがピーク。M4 で実測し batch_size を調整（64 起点）。**解消**: batch 32×2GPU（実効64）で VRAM ピークは 32GB 内に収束（M4） |
| 6 | MAS（CPU/numba）がボトルネックになる可能性 | GPU 利用率低下で学習が遅い | M4 の実測で判明する。深刻なら DataLoader worker 数調整等で緩和。**解消**: 平均 6〜7 it/s・約10分/epoch を実測しボトルネックにならず（M4） |
| 7 | 破滅的忘却（中英能力の喪失） | 多言語用途には使えなくなる | 意図的なトレードオフとして許容（§2）。多言語が必要な場面では本家チェックポイントを使い分け |
| 8 | g2p 切替（use_vanilla=False）と既存チェックポイントの混用 | 本家チェックポイントで推論すると学習時と読み分布が微妙にずれる | 新モデルとセットで運用する前提を README に明記。本家モデルの再評価が必要な場合のみ一時的に True に戻す |
| 9 | 過学習・話者リーク | 評価が過大に見える | ホールドアウト話者で評価（M2/M6）。エポック別チェックポイント比較で選定 |
| 10 | Windows 環境固有の問題（DDP gloo 分岐等） | 学習スクリプトの想定外動作 | 単一GPUなので DDP の複雑さはほぼ回避。M4 の試走で洗い出す |

## 6. 未決事項

事前学習（M0-M7）と後続研究フェーズ（Phase 1-3）が完了したため、当初の未決事項はすべて決着した。

（決着済み）
- **フィルタ閾値の最終値** → M2 で speechMOS ≥ 1.5 / ASR一致度 ≥ 0.90 / 1〜15秒 に確定（UTMOS 閾値案は演技音声を系統的に低採点するため不採用）
- **モデル公開の要否と範囲** → M7 で HF に3モデルを public 公開（[事前学習 378h](https://huggingface.co/ayousanz/stable-tts-v1.1-japanese-378h) / [tsukuyomi-ft](https://huggingface.co/ayousanz/stable-tts-v1.1-japanese-378h-tsukuyomi-ft) / [baseline](https://huggingface.co/ayousanz/stable-tts-v1.1-tsukuyomi-ft-baseline)、license: other / moe-speech-terms）
- **実行環境** → vast.ai の 2×RTX 5090 に確定（2026-07-06、API 調査に基づく。§3 参照）
- **ゼロショット話者類似性の改善手段** → Phase 3 で TLA-SA / MRTE を試し両方不採用（[phase3-plan.md §13](phase3-plan.md)）。後付け継続学習では pooled ボトルネックを解けないと実証

**残る検討事項（今後の方向性、別途計画として整理）**: pooled ボトルネックを本当に解くには大きめの投資が必要で、以下4案が候補（詳細は [phase3-plan.md §13](phase3-plan.md) / [research-summary.md](research-summary.md)）。
1. **MRTE をスクラッチ学習**（upstream checkpoint_0 から MRTE 込みで学習し pooled 依存を作らせない。実装は流用可、config 変更のみ）
2. **in-context infilling 全面移行**（F5/E2/ZipVoice 型。上限最高だが既存 checkpoint を捨てるリビルド、378h で暗黙アラインメント不安定）
3. **データ拡張**（378h→数千時間規模。話者多様性がゼロショット類似性の上限を決める）
4. **日本語アクセント高度化**（話者類似性でなく日本語品質そのものの改善。記号方式は実装済み、音素ごと連続高低・核位置の埋め込み追加）

## 7. 参考情報（今回の調査の実測値）

- 本家 v1.1 推論速度: RTX 4070 Ti SUPER で平均 RTF 2.43（step 25 / dopri5 / cfg 3。euler なら数倍高速）
- g2p 差分: use_vanilla True vs False は誤読しやすい16文中4文で差、すべて plus 側が正、全て「何」関連 + アクセント。文脈依存同形異音語（辛い・人気 等）は両者とも不正解のケースあり
- 評価サンプル置き場: `temps/eval_v11/`（moe-speech 参照）、`temps/eval_tsukuyomi/`（つくよみちゃん参照、正解音声同梱）
- moe-speech: 473話者 / 約395kファイル / 約623時間 / 44.1kHz 16bit mono / 2〜15秒 / スタジオ収録演技音声
