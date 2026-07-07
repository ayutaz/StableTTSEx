# StableTTS アーキテクチャ改善調査

作成日: 2026-07-06 / 対象: StableTTSEx（upstream v1.1、31.6M パラメータ）/ 関連文書: [事前学習レポート](pretraining-report.md)

## 0. 調査の目的と結論

日本語継続事前学習（moe-speech 378h）で表現力・韻律は改善したが、StableTTS のアーキテクチャ自体は 2024 年前半世代であり、最新 TTS との品質差が残る。本調査は「現在のアーキテクチャからどのような変更で精度を上げられるか」を、(a) Stable Diffusion 3.5 の改善の移植という仮説の検証、(b) 2024〜2026 の TTS 研究動向の両面から整理する。

**結論の要約:**

1. **「SD3.5 合わせ」仮説は部分的に有効だが、それだけでは差は埋まらない。** StableTTS が SD3 から借りているのは「flow-matching + DiT を組み合わせる」という発想のみ（README の References に明記）で、SD3 の中核である MM-DiT（テキスト/画像の2ストリーム joint attention）は採用していない。したがって SD3→3.5 の変更のうち移植できるのは **QK 正規化・logit-normal timestep サンプリング・Skip Layer Guidance の3点**で、いずれも「小〜中」の改善。SD3.5 Medium の MMDiT-X や dual attention は2ストリーム構造前提のため直接は移植不可。
2. **最新 TTS との品質差の主因は SD の世代差ではなく、(1) 参照音声の条件付け方式（pooled style vector か in-context か）と (2) データ規模（50k〜1M 時間）。** このうち (1) はアーキテクチャの問題であり、小規模（〜100M パラメータ・数百時間）でも再現可能な改善が複数存在する（P-Flow、MRTE、TLA-SA）。(2) は個人規模では追わない。
3. **日本語に限れば、明示的 g2p + アクセント特徴は弱点ではなく資産。** 2025〜26 の大規模ゼロショットモデルは raw text 入力でピッチアクセントを扱わず、日本語 WER は商用トップでも悪い（MiniMax-Speech 18.1%）。逆に小型で高評価の Kokoro は日本語処理を pyopenjtalk-plus + ピッチアクセントに収斂させた — 本フォークと同じツールチェーンである。**アクセント特徴の埋め込み追加が日本語品質で最も費用対効果の高い変更。**
4. 推奨順序: まず**再学習不要の推論改善**（Sway Sampling・interval CFG・複数参照平均）→ 次に**チェックポイント互換のレシピ変更**（logit-normal・EMA）で再事前学習 → 効果を確認してから**非互換の構造変更**（アクセント埋め込み・QK-Norm・参照条件付けの sequence 化）に進む。

---

## 1. 現状アーキテクチャの正確な把握（コード実態）

調査エージェントの報告とコードを突き合わせて確認した現状。外部レポートには「uniform t サンプリング」「adaLN 無印」等の誤りがあったため、以下はコードを正とする。

| コンポーネント | 実装 | 最新手法との差分 |
|---|---|---|
| DiT ブロック | `DiTConVBlock`（HierSpeech++ 由来、DiT + FFT 混合）。LayerNorm(affine なし) → MHA → Conv1d FFN | FFN が Conv1d+SiLU（SwiGLU 等の gated MLP でない）。QK 正規化なし |
| 条件付け | **adaLN-Zero は話者ベクトル c**、**timestep は FiLM 層**で注入という変則2系統（`models/estimator.py` の `DitWrapper`） | 標準 DiT / F5-TTS は timestep(+class) を adaLN-Zero に入れる。t と c を統合して adaLN に入れる整理が可能 |
| 位置符号化 | RoPE（ヘッド次元の 50% に部分適用、`models/diffusion_transformer.py:48`） | Stable Audio と同じ設計。**すでに現代的** |
| テキスト融合 | エンコーダ出力 mu を noise と **concat** して in_proj（`models/estimator.py:120`） | MM-DiT の joint attention ではない。F5 も concat 系なのでこれ自体は標準的 |
| 長スキップ | U-Net 風 long skip connection（U-ViT, arXiv:2209.12152） | v1.1 で追加済み。現代的 |
| timestep サンプリング | **CosyVoice 由来 cosine スケジューラ** `t = 1 - cos(u·π/2)`（`models/flow_matching.py:90`）— noise 側(t≈0)を重点サンプリング | SD3 論文の ablation では **logit-normal(0,1)（中間 t 重点）が一貫して最良**。現行は中間軽視の逆向きバイアス |
| FM 定式化 | OT-CFM（Matcha 系、σ_min=1e-4）、損失は素の MSE | F5/E2 の Rectified Flow とほぼ等価。変更不要 |
| アライメント | MAS（単位分散prior）+ 決定論的 duration predictor（VITS 系） | 最新勢は infilling / sparse anchor / token-count 制御に移行。ただし MAS は少データで安定という利点も残る |
| 参照エンコーダ | `MelStyleEncoder`（GPT-SoVITS 系）→ **単一 256 次元ベクトルに pooling** | **最新勢が総じて廃止した方式**。ゼロショット類似性が弱い構造的原因（§4） |
| CFG | fake_speaker + fake_content を同一マスクで drop（p=0.2）、推論時 cfg≈3 | 話者/内容の分離 CFG（multi-CFG）や interval CFG は未実装 |
| ボコーダ | Vocos（MIT）/ FireflyGAN（**CC-BY-NC-SA、非商用**） | BigVGAN v2 に 44.1kHz/128mel の MIT チェックポイントあり |
| 学習 | EMA なし、bf16 混合精度なし | EMA は定番の安価な改善 |

---

## 2. 仮説検証: SD3 → SD3.5 の変更と移植可否

SD3→3.5 のアーキテクチャ変更は実質2つ + 推論テク1つ（テキストエンコーダ・VAE・スケジューラは不変。出典: [HF SD3.5 blog](https://huggingface.co/blog/sd3-5)、[SD3.5 Medium model card](https://huggingface.co/stabilityai/stable-diffusion-3.5-medium)）。

| 変更 | 内容 | 種別 | StableTTS への移植 |
|---|---|---|---|
| QK 正規化 | attention の Q/K に RMSNorm（学習安定化。ViT-22B → SD3 論文由来） | 構造（非互換） | **可・推奨**。`MultiHeadAttention.attention` に十数行。31M では安定性問題は起きにくいため効果は保険的だが、次の非互換学習に同梱する価値あり |
| dual attention 層 | MM-DiT 前段ブロックでストリーム毎に attention を2重化 | 構造 | **不可（as-is）**。テキスト/画像2ストリームの MM-DiT 前提。StableTTS は単一ストリーム concat 融合 |
| MMDiT-X（Medium のみ） | 前段13層に self-attention 追加 + マルチ解像度学習 | 構造+学習 | **不適用**。動機が画像のマルチ解像度で、mel（周波数軸固定・時間1D）に対応物がない |
| Skip Layer Guidance | 特定層をスキップした出力を第3の guidance 項に使う（ステップ範囲限定） | **推論のみ** | **実験可**。再学習不要。ただしデコーダ6層と浅く、音声への効果は未検証（投機的） |
| （SD3 論文）logit-normal timestep | `logit(t)~N(0,1)` で中間 t を重点学習。ablation で `rf/lognorm(0.00,1.00)` が最良 | **レシピのみ（互換）** | **可・最優先**。現行 cosine スケジューラ（`flow_matching.py:90`）の置き換え。既存チェックポイントから継続学習で A/B 可能 |

**仮説への回答**: 「SD3.5 に合わせる」で得られるのは QK-Norm(小) + logit-normal(中) + SLG(投機的) 程度。やる価値はあるが、これは本命ではない。SD3 化を突き詰めるなら「テキストと mel の2ストリーム MM-DiT 化」が本当の SD3 化だが、大規模改造であり §3 の in-context 条件付けの方が TTS の実績がある。

---

## 3. 2024〜2026 TTS の潮流と品質差の要因

### パラダイム分布

| 系統 | 代表 | 話者条件付け |
|---|---|---|
| NAR flow-matching / infilling | F5-TTS, E2-TTS, Voicebox, P-Flow, MegaTTS3, ZipVoice | 参照**フレーム列**を同一 attention 内で in-context 参照 |
| AR codec-LM | CosyVoice 2/3, IndexTTS 2, Llasa, Fish-Speech, Chatterbox, Zonos | 参照**トークン列**を prefix にした継続生成 |
| masked 生成 | MaskGCT | prompt トークン prefix |
| 因子化 codec | NaturalSpeech 3（FACodec） | GRL で内容/韻律から**分離した** timbre ベクトル |
| StyleTTS2 系 GAN | Kokoro（82M）, Style-Bert-VITS2 | 固定話者（ゼロショットでない） |

### 品質差を生む5つの要因（影響度順）

1. **pooled style vector の廃止 → 参照系列への in-context 条件付け**（§4 で詳述）
2. **semantic token + LLM フロントエンド**（CosyVoice の FSQ 等）— 表現力・streaming・RL 調整可能性を買えるが 0.5B+ / 10万時間級が前提。**個人規模では追わない**
3. **アライメントの脱 MAS 化**（infilling / sparse anchor / token-count 制御）
4. **データ 50k〜1M 時間 + RL 後処理** — 再現不可能。追わない
5. **強い codec / 分離された話者表現**（WavVAE, FACodec, BiCodec）

### 小規模（〜100M・数百時間・2×コンシューマ GPU）で再現可能なもの

- **P-Flow**（NeurIPS 2023）: speech prompt を text encoder に cross-attention で入れる方式。**585 時間**で VALL-E 級の類似性を達成した、StableTTS に系譜が最も近い実証例。MAS/duration predictor は維持できる
- **MegaTTS 2 の MRTE**: 音素列を query、参照 mel を key/value とする cross-attention で細粒度の音色を注入。参照が長いほど類似性が上がる特性（SIM 0.905@10s → 0.932@300s）を獲得できる
- **TLA-SA**（arXiv:2511.09995）: flow-matching デコーダの中間表現を事前学習話者埋め込みに整列させる**補助損失**。LM-free flow-matching 向けに設計されており、StableTTS にそのまま bolt-on 可能（推論時は捨てられる projection head のみ追加）
- **FACodec 式の参照埋め込み分離**（bottleneck + 話者分類監督 + GRL）: pooled vector を維持したままエンタングルメントだけ解消する低リスク案
- 小型 flow-matching の実証: Flamed-TTS-Small 76M ≈ Base 品質、DiFlow-TTS は 470h で高品質 — 31M→50-100M への増量も選択肢

主要出典: [F5-TTS](https://arxiv.org/abs/2410.06885) / [E2-TTS](https://arxiv.org/abs/2406.18009) / [MegaTTS3](https://arxiv.org/abs/2502.18924) / [P-Flow](https://openreview.net/forum?id=zNA7u7wtIN) / [CosyVoice2](https://arxiv.org/abs/2412.10117) / [CosyVoice3](https://arxiv.org/abs/2505.17589) / [IndexTTS2](https://arxiv.org/abs/2506.21619) / [MaskGCT](https://arxiv.org/abs/2409.00750) / [NaturalSpeech3](https://arxiv.org/abs/2403.03100) / [Voicebox](https://arxiv.org/abs/2306.15687) / [VoiceStar](https://arxiv.org/abs/2505.19462) / [MegaTTS2](https://arxiv.org/abs/2307.07218)

---

## 4. ゼロショット類似性の根本原因（今回の事前学習で残った最大の課題）

事前学習レポートで「378 時間でも構造的に厳しい」と判定したゼロショット類似性の弱さは、文献上も pooled style vector の構造的限界として説明できる。

1. **情報ボトルネック**: 発話全体の音色・微細韻律・録音環境を 256 次元1本に押し込めない。MegaTTS 2 は単一ベクトル prompt を「不十分」と明言し、参照を長くするほど類似性が上がることを示した — 飽和する固定ベクトルには原理的に不可能な挙動
2. **同一ネットワーク内 attention vs 外部注入**: E2/F5 は参照フレームとターゲットを同じ self-attention スタックで処理するため、生成の各ステップが「必要な参照フレーム」を直接参照できる。E2-TTS は参照エンコーダと duration モデルを**撤去して** Voicebox/NS3 を SIM で上回った（0.708 vs 0.667/0.632）— ボトルネックが害だった直接証拠
3. **pooled 埋め込みの併用は劣化要因**: pooled 話者埋め込みを in-context に追加すると発話レベル情報が二重化・エンタングルして韻律連続性が劣化する報告（arXiv:2210.16045）。最新設計が pooled を「足す」のでなく「捨てる」のはこのため

**ただし infilling 全面移行には注意点**: 学習時の長さ分布外で急激に壊れる（F5 は 40〜50 秒生成で WER 52%）。採用時は IndexTTS2 の token-count 埋め込みか VoiceStar の PM-RoPE のような明示的長さ制御を併せて入れる。**中間解として、MAS/duration predictor を保つ P-Flow / MRTE 方式が本フォークには現実的。**

補足（運用で今すぐ効く話）: `collate_fn` は学習時に短いランダムスライスを参照にするため、推論で長い参照1本を入れるとエンコーダには分布外になる。**長参照1本より「複数の短スライスの埋め込み平均」が安全** — 進行中の長参照検証（`temps/eval_longref/`）はこの点を踏まえて解釈すること。

---

## 5. 日本語特化の改善（本フォーク固有の優位性）

- **アクセント特徴の埋め込み追加が最優先。** Style-Bert-VITS2 は pyopenjtalk のフルコンテキストラベルから (a) 音素ごとの高低 (0/1) を別埋め込みで加算、または (b) `[`（上昇）`]`（下降）記号を音素列に挿入する方式を採用。本フォークは既に `extract_fullcontext` を呼んでおり、アクセント核・アクセント句境界・モーラ情報は**取得済みで捨てている**状態。配管工事だけで済む。カタカナ語・英字のアクセント不安定（既知課題）にも、テキスト正規化と並ぶ直接対策になる
- **raw text 入力への転換はしない。** 大規模モデルの共通弱点が日本語ピッチアクセントであり（MiniMax-Speech の日本語 WER 18.1% vs ElevenLabs v3 11.0%、[MINT-Bench](https://arxiv.org/pdf/2604.17958)）、Kokoro（82M、TTS Arena 首位経験）は日本語トークナイザを pyopenjtalk-plus + UniDic + ピッチアクセントに収斂させた。明示的 g2p は本フォークの差別化要素
- **PL-BERT / PnG BERT 系の意味条件付け**は長文・分布外テキストの自然性に効く（StyleTTS 2 の中核）が、統合コストが高い。アクセント埋め込みの後の第二弾
- 実装形態の注意: アクセント埋め込みを「加算埋め込み（ゼロ初期化）」で入れると、既存チェックポイントを部分ロードして継続学習で立ち上げられる。記号挿入方式（n_vocab 変更）は互換性を完全に失う

---

## 6. 改善候補の総合マップ

| # | 施策 | 効果 | コスト | チェックポイント互換 | 対応する弱点 |
|---|---|---|---|---|---|
| 1 | Sway Sampling（`f(u)=u+s(cos(πu/2)−1+u)`, s=−1） | 中（低ステップ時の品質維持、速度2倍） | 低（推論のみ、数行） | **互換** | 推論速度 |
| 2 | interval CFG + CFG rescale | 中 | 低（推論のみ） | **互換** | 過剰 guidance の副作用 |
| 3 | 複数参照スライスの埋め込み平均 | 小〜中 | 低（推論のみ） | **互換** | ゼロショット類似性 |
| 4 | Skip Layer Guidance | 不明（音声で未検証） | 低（推論のみ） | **互換** | 品質全般（投機的） |
| 5 | logit-normal(0,1) timestep サンプリング | 中 | 低（1行置換 + 再学習） | **互換（レシピのみ）** | 品質全般 |
| 6 | EMA 重み | 小〜中 | 低 | **互換（レシピのみ）** | 品質全般 |
| 7 | TLA-SA 話者整列補助損失 | 中 | 中（話者埋め込みモデル導入） | ほぼ互換（学習時のみ head 追加） | ゼロショット類似性 |
| 8 | 日本語アクセント埋め込み | **高（日本語）** | 低〜中 | 加算方式なら部分ロード可 | アクセント・韻律 |
| 9 | QK-Norm + gated MLP + t/c の adaLN 統合 | 小〜中 | 中 | 非互換 | 学習安定性・品質 |
| 10 | MRTE（参照 mel への cross-attention） | 中〜高 | 中 | 非互換（新モジュール、ゼロ初期化で部分ロード可） | ゼロショット類似性 |
| 11 | P-Flow 式 speech prompt 化 | **高** | 高 | 非互換 | ゼロショット類似性 |
| 12 | 参照エンコーダの FACodec 式分離（GRL） | 中 | 中 | 非互換 | 類似性・韻律の混入 |
| 13 | E2/F5 式 infilling 全面移行 | 高（ただし長さ制御必須） | **高（別モデル）** | 非互換 | 類似性・表現力 |
| 14 | MM-DiT 化（真の SD3 化） | 不明（TTS 実績薄） | 高 | 非互換 | — |
| 15 | ボコーダ BigVGAN v2 44kHz/128mel 評価 | 中（表現力の強い音源で有利） | 低〜中（mel 設定一致確認要） | 互換（外部） | 音質・FireflyGAN の NC ライセンス回避 |
| 16 | 蒸留による 2〜4 step 化（MeanFlow 等） | 中（レイテンシのみ） | 高 | 非互換 | 速度（当面 Sway で十分） |

---

## 7. 推奨ロードマップ

**Phase 1 — 推論のみ・再学習なし（施策 1,2,3,4,15）**
既存の `japanese-378h(-tsukuyomi-ft)` チェックポイントのまま実装して A/B。Sway Sampling は `CFMDecoder.forward` の `t_span` 変換のみ。長参照検証（残課題）は「複数スライス平均」も条件に加える。ここで数 % でも改善が拾えれば公開モデルにもそのまま効く。

**Phase 2 — レシピのみの再事前学習（施策 5,6 ± 7）**
cosine → logit-normal 置換 + EMA で moe-speech 378h を再学習（前回実績: 15 epochs ≈ 2.5h / $2.5、環境は維持中の vast インスタンスに再現済み）。現行チェックポイントとの聴感 A/B + ECAPA cos 類似度で効果を定量化。TLA-SA はこのフェーズで実験的に追加してもよい。

**Phase 3 — 非互換の構造変更をまとめて1回で（施策 8,9,10）**
チェックポイントを壊す変更は同梱して1回の再事前学習に集約する: 日本語アクセント埋め込み（加算・ゼロ初期化）+ QK-Norm + MRTE。部分ロード（既存重みを読み、新規モジュールはゼロ初期化）で checkpoint_0 系の資産を引き継ぐ。ここが「日本語特化 StableTTS v2」の本体。

**Phase 4 — 研究テーマ（施策 11,13,14）**
P-Flow 式 prompt 化 or infilling 化は Phase 3 の結果を見てから。パラメータ 31M→50-100M への増量もこの段階で検討（Flamed-TTS-Small 76M の実証あり）。

## 9. Phase 1 実装状況（2026-07-07）

推論のみの4施策を実装済み（`api.py` / `models/model.py` / `models/flow_matching.py` / `models/estimator.py` / `webui.py`）。**全パラメータはデフォルトで既存挙動をビット単位で維持**する（`tsukuyomi_ft200.pt` で euler・dopri5 とも実装前後の mel が max abs diff = 0.0 を確認）。既存チェックポイントにそのまま適用でき、再学習不要。

| 施策 | API パラメータ（`StableTTSAPI.inference`） | デフォルト（=無効） | 備考 |
|---|---|---|---|
| Sway Sampling | `sway_coef` | `None` | **euler 等の固定ステップソルバー専用**（dopri5 では t_span が出力評価点にすぎず無効）。`s=-1` で学習時 cosine スケジューラと同一 warp。単調性のため `[-1.0, 1.75]` にクランプ |
| CFG rescale | `cfg_rescale` | `0.0` | Lin+ 2023 §3.4 を velocity 空間に適用。過剰 CFG の飽和抑制。推奨 0.7 |
| interval CFG | `cfg_interval` | `None` | `(t_min, t_max)`。区間外は uncond 前向きを省略（固定ステップソルバーで NFE 削減） |
| 複数参照平均 | `ref_window_seconds`（+ `ref_audio` に `list[str]` 可） | `None` | 参照を学習時スライス長相当の窓（既定 2.0s）に分割し style vector を平均。ファイル内平均→ファイル間平均の2段 |
| SLG（非推奨） | `slg_scale` / `slg_layers` / `slg_t_range` | `0.0` / `(2,)` / `(0.0, 0.5)` | `estimator.py` の層スキップ経由。CFG と独立に適用可（`cfg=1.0` 単独でも有効）。webui には未露出（API のみ）。**実験Dで全条件悪化＝6層では効かないと確認済み**。実装のみ残置 |

WebUI（`webui.py`）には Sway Coef・CFG Rescale・CFG t_min/t_max スライダーと参照音声の詳細設定（複数窓平均・追加参照ファイル）を追加済み（SLG を除く）。A/B サンプル生成スクリプトは `temps/phase1_ab/`（`gen_baseline.py` = デフォルト不変チェック、`gen_ab.py` = 各施策の健全性/聴感比較）。

**実装済み: 施策15（BigVGAN v2）。** NVIDIA/BigVGAN（MIT）の推論コードを `vocoders/bigvgan/` に vendor（huggingface_hub の `PyTorchModelHubMixin` と CUDA カーネル分岐を除去、import 相対化、`utils.py` を `init_weights`/`get_padding` の2関数に絞り matplotlib/scipy 依存を排除。ライセンスは `LICENSE` + `incl_licenses/` に MIT/Apache-2.0/BSD 由来を保持）。`vocoders/bigvgan/model.py` の `BigVGANWrapper` は config を直書きし、`load_state_dict`→`remove_weight_norm`→`eval` の順で構築、出力 `(B, T)`。`api.py` の `get_vocoder` は `'bigvgan'` 分岐で `BigVGANWrapper` をロード（`pyproject.toml` は変更なし = torch/torchaudio のみで動作）。チェックポイント `vocoders/pretrained/bigvgan_generator.pt`（466MB, MIT, `nvidia/bigvgan_v2_44khz_128band_512x`）は .gitignore 済で非追跡。**copy-synthesis 検証で mel 仕様の一致を実地確認**（held-out 実音声3本で mel-L1: bigvgan 0.176 vs ffgan 0.174、比 1.011、出力長一致・NaN なし。スクリプト `temps/bigvgan_eval/copy_synth.py`）。FireflyGAN の CC-BY-NC-SA（非商用）を MIT で置換できる。**同一 TTS mel の ffgan/bigvgan 聴き比べで問題なしと確認したため webui 既定を bigvgan に変更済み**（2026-07-07。`temps/bigvgan_eval/ab/` に聴き比べペア。ffgan/vocos に戻すには `webui.py` の `vocoder_model_path`/`vocoder_type` の2行を戻すだけ）。

### Phase 1 定量 A/B 評価結果（2026-07-07）

モデル `checkpoints/vast_run1/checkpoint_14.pt`（378h 事前学習・ゼロショット）、held-out 3話者（学習除外）× 日本語5文で測定。指標: CER（faster-whisper large-v3）／話者類似性（ECAPA-TDNN cos、参照埋め込みとの内積）／mel std・peak（飽和 proxy）／mel-L1（dopri5-25 出力を基準とした距離）／生成時間。スクリプト・数値は `temps/phase1_eval/`（`eval_phase1.py`、`results_full.json`、165条件）。**評価用依存（faster-whisper/jiwer/librosa/speechbrain）は `uv pip` で venv に入れただけで `pyproject.toml` には未追加**（`uv sync` で消える）。

**実験A: Sway Sampling は euler 低ステップを dopri5 品質に引き上げる（本命の成果）。** 各条件 n=10。

| 条件 | CER | spk_cos | mel_std | peak | dopri25 との mel-L1 | 生成秒 |
|---|---|---|---|---|---|---|
| dopri5 / 25step（品質基準） | 0.002 | 0.640 | 2.553 | 0.832 | — | 7.08 |
| euler16 / sway なし | 0.005 | 0.654 | 2.773 | 0.945 | 0.323 | 0.69 |
| euler16 / **sway −1.0** | 0.002 | 0.641 | 2.552 | 0.837 | **0.078** | 0.70 |
| euler8 / sway なし | 0.016 | 0.630 | 3.315 | 0.999（クリップ） | 0.914 | 0.40 |
| euler8 / sway −1.0 | 0.002 | 0.635 | 2.587 | 0.872 | 0.196 | 0.37 |

`euler16 + sway −1.0` は dopri5-25 と mel-L1=0.078（ほぼ一致）・mel std/peak/CER/話者類似性すべて同等で、**約10倍高速**（0.70s vs 7.08s）。sway なしの低ステップは mel が過飽和（peak≈1.0 でクリップ、mel_std 増）し dopri5 から乖離する。**推奨: 固定ステップ運用は euler / step 16 / sway_coef −1.0。** さらに速い euler8+sway−1.0 も実用域（L1 0.196）。

**実験B: CFG rescale は飽和を抑える安全な小改善。interval CFG は既定にしない。** 各条件 n=10、euler16。

| 条件 | CER | spk_cos | mel_std | peak |
|---|---|---|---|---|
| cfg 3.0（現行既定相当） | 0.005 | 0.654 | 2.773 | 0.945 |
| cfg 3.0 + **rescale 0.7** | 0.002 | 0.658 | 2.537 | 0.910 |
| cfg 3.0 + interval (0.4,1.0) | 0.008 | 0.623 | 2.421 | 0.546 |
| cfg 5.0（過剰 guidance） | 0.006 | 0.607 | 3.073 | 0.983 |
| cfg 5.0 + rescale 0.75 | 0.002 | 0.664 | 2.467 | 0.929 |

rescale 0.7 は mel_std と peak を下げつつ CER・話者類似性を維持〜微改善。cfg 5.0 単体は話者類似性を明確に劣化（0.607）させるが rescale 0.75 が回復（0.664）— **rescale は高 CFG を安全に使えるようにする**。interval CFG は音量を大きく下げ（peak 0.546）話者類似性も落ちるため既定化しない。

**実験C: 複数参照平均は今回の参照長（≤9秒）では効果なし（むしろ微減）。** spk_cos 全体平均: fullref 0.650 / win2.0s 0.633 / win1.5s 0.637。9秒参照でも改善せず。学習時スライス長に対しこの長さは強い分布外ではないため、窓平均は情報を薄めるだけになったと解釈できる。**期待した効果は 30〜60 秒級の長参照でしか出ない見込みで、その長さの検証データが無いのが限界。既定は無効のままとし、長参照運用時の選択肢として残す。**

**実験D: SLG（Skip Layer Guidance）は全条件で悪化 → 不採用。** cfg 3.0 baseline（slg0, spk_cos 0.660 / mel_std 2.80 / peak 0.94）に対し、全 SLG 条件で話者類似性が低下し mel_std/peak が上昇（過飽和方向）。

| 条件（cfg3+SLG） | CER | spk_cos | Δspk | mel_std | peak | mel-L1(対slg0) |
|---|---|---|---|---|---|---|
| slg0（baseline） | 0.003 | 0.660 | — | 2.80 | 0.94 | — |
| L2 / wide / s1.0 | 0.003 | 0.626 | −0.034 | 2.88 | 0.97 | 0.231 |
| L2 / wide / s2.0 | 0.007 | 0.575 | −0.086 | 2.98 | 0.99 | 0.461 |
| L2 / narrow / s2.0 | 0.003 | 0.629 | −0.032 | 2.88 | 0.98 | 0.234 |
| L3 / wide / s2.0 | 0.330 | 0.261 | −0.399 | 5.42 | 1.00 | 2.807 |

最も穏やかな設定（L2/narrow/s1.0 相当）でも改善はなく話者類似性が下がる。L3（デコーダの pop 側層）スキップは破滅的に崩壊（CER 0.33・mel_std 5.4）。計画時の懸念どおり、**6層デコーダでは1層スキップが粗すぎて独立した guidance 方向にならず、CFG と冗長な過飽和にしかならない**。cfg=1.0 単独 SLG も baseline を回復できない。SLG は実装を残すが既定無効・非推奨とする。

**結論**: Phase 1 の確実な成果は **Sway Sampling（低ステップで dopri5 品質・約10倍高速）** と **CFG rescale（飽和抑制の安全な小改善）** の2点で、webui 既定に採用済み（`solver=euler / step=16 / sway_coef=−1.0 / cfg_rescale=0.7`、2026-07-07 聴感確認で承認）。複数参照平均・interval CFG は既定無効のまま長参照/特殊用途のオプションとして温存。**SLG は6層デコーダでは効かないことを定量確認したため非推奨**（施策4クローズ）。ボコーダは BigVGAN v2（MIT）を追加済みで、FireflyGAN の非商用ライセンスを回避可能（施策15クローズ）。

## 8. 調査の限界

- 効果見積りの多くは画像（SD3）または英中 TTS の結果からの転用で、日本語 44.1kHz mel での検証値はない。Phase 1〜2 で自前の A/B を取ってから Phase 3 に進む構成にしているのはこのため
- QK-Norm・logit-normal の TTS での直接検証は文献が薄い（画像/LLM では確立）。SwiGLU 系 FFN は音声 DiT での採用が割れている（F5 は素の GELU）
- 事前学習話者埋め込み（ECAPA 等）への単純な参照エンコーダ置換は、TTS 類似性への転移が否定的な報告もあり（arXiv:2506.20190）、プロトタイプ検証なしに本採用しない
- Skip Layer Guidance の音声への効果は完全に未検証（画像では構図・解剖学的整合の改善）
