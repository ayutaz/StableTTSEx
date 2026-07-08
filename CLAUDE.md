# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## プロジェクト概要

StableTTSEx は [KdaiP/StableTTS](https://github.com/KdaiP/StableTTS) のフォーク（upstream v1.1 ベース）。flow-matching と DiT を組み合わせた軽量 TTS モデル（約31Mパラメータ）で、単一チェックポイントで中国語・英語・日本語に対応する。話者IDは使わず、reference encoder が参照音声から話者性を抽出するゼロショット方式。フォーク固有の変更は、日本語をデフォルトにする設定・`generate-audio-list.py`・uv による依存管理・日本語 g2p の pyopenjtalk-plus 化など小規模。

テストは `tests/`（pytest、`[dependency-groups]` の `dev`）にある。**GPU・実チェックポイント・ネットワーク不要の CPU 決定論テスト**で、`uv run pytest` で実行する。方針: モデル出力は torch 版で微小に揺れるため値ゴールデンではなく shape・有限性・同一 seed 内一致・関係性（feature ON/OFF で出力が変わる差分）・no-op 等価を検証する。ただし `len(symbols)==401`・g2p の音素列ゴールデン・baseline パラメータ数（31,644,545）・MRTE full-config（`use_mrte=True`）のパラメータ数（33,225,345）は「壊れると静かに既存チェックポイント非互換になる」不変条件として意図的に厳密固定している。研究アークの残置機構は `test_reference_encoder.py`（return_sequence）・`test_model.py`（MRTE の param golden と zero-gate ビット一致・state_dict superset・配線ガード）・`test_config.py`（Phase 3 クローズ設定 = `use_mrte`/`use_tla_sa` False・cosine・EMA無 を固定）で検証する。`conftest.py` は torch import 前に `CUDA_VISIBLE_DEVICES=''` で CPU 固定し、tiny config（hidden 偶数・hidden%heads==0・n_dec_layers 偶数）の StableTTS/CFMDecoder factory を提供する。`train.py`/`preprocess.py` はモジュールレベル副作用があるため import せず、部品（config/models/utils/datas/text と api の g2p_mapping）をテストする。設定は CLI 引数ではなく、`config.py` や各スクリプト冒頭の dataclass を直接編集する方式。

リンタ／フォーマッタは ruff（`pyproject.toml` の `[tool.ruff]`）。`line-length=120`、ルールは `E/F/W/I/UP/B`（`E501` は formatter に委譲するため lint では無効、`UP031` は upstream g2p の `'%s' % x` 慣用のため許容）。**vendor 3rd-party（`text/cn2an`・`text/custom_pypinyin_dict`・`vocoders/{bigvgan,ffgan,vocos}`・`monotonic_align`）は `extend-exclude` で対象外**にしており、自分たちが著作しないコードは整形・lint しない。ruff は `[dependency-groups]` の `dev` グループにあり `uv sync` でデフォルト導入される。

依存は uv（`pyproject.toml` + `uv.lock`）で管理する。Python は 3.13 固定（`.python-version`。上限 <3.14 は torch 2.8+cu128 と pyopenjtalk-plus に cp314 の Windows wheel が無いため）、numpy は `>=2.1,<2.5`（上限は numba 0.66 の numpy 対応上限。numba が numpy 2.5 に対応したら緩和可）。torch/torchaudio は 2.8 系 + cu128 を `[tool.uv.index]` / `[tool.uv.sources]` で自動解決する（別途インストール不要。torchaudio 2.9+ は load/save が TorchCodec 委譲になりコード非互換なので上げない）。

## よく使うコマンド

```bash
# 依存関係（torch cu128 含めて全て uv が解決する。初回は 2.5GB 超のダウンロードあり）
uv sync --extra webui             # 学習+推論+WebUI
uv sync                           # WebUI 不要なら（gradio/matplotlib を省く）
# recipes/ のスクリプトを使う場合は --extra recipes を追加（openpyxl, pandas）

# データ準備: esd.list（Style-Bert-VITS2 形式: file|speaker|lang|text）→ filelist.txt
uv run python generate-audio-list.py

# 前処理: filelist.txt（audiopath|text 形式）→ mel特徴量(.pt) + filelist.json
# 実行前に preprocess.py の DataConfig（入出力パス・language）を編集する。多言語データは言語ごとに別々に処理する
uv run python preprocess.py

# 学習（DDP。冒頭で CUDA_VISIBLE_DEVICES='0,1' がハードコードされている点に注意）
# config.py の TrainConfig を編集してから実行。checkpoints/ に既存チェックポイントがあれば自動レジューム
uv run python train.py

# 学習ログ
uv run tensorboard --logdir runs

# コード品質（ruff。vendor コードは pyproject の extend-exclude で対象外）
uv run ruff check              # lint（--fix で安全な自動修正を適用）
uv run ruff format             # フォーマット（--check で差分の有無だけ確認）

# テスト（pytest。CPU のみ・GPU/実チェックポイント/ネットワーク不要。約10秒）
uv run pytest                       # 全テスト
uv run pytest tests/test_ema.py -v  # 個別ファイル / -k で個別テスト

# 推論
uv run python webui.py # Gradio WebUI（share=True で起動、デフォルト言語は日本語）
# または inference.ipynb / api.py の StableTTSAPI を使用
```

事前学習済みモデルの配置（HuggingFace KdaiP/StableTTS1.1 から取得）:
- TTS 本体 `checkpoint_0.pt` → `./checkpoints/`
- ボコーダ → `./vocoders/pretrained/`（`vocos.pt` / fishaudio の `firefly-gan-base-generator.ckpt` / BigVGAN v2 の `bigvgan_generator.pt`。webui.py のデフォルトは bigvgan。BigVGAN は `nvidia/bigvgan_v2_44khz_128band_512x` から取得）

## アーキテクチャ

### 音声合成パイプライン全体

```
テキスト → g2p（言語別音素化）→ 音素ID列（intersperse で blank=0 を挿入）
  → StableTTS（テキスト→mel）→ ボコーダ（mel→波形, 44.1kHz）
```

- **推論の入口は `api.py` の `StableTTSAPI`**: TTS モデル・ボコーダ・g2p をまとめてロードし、`inference(text, ref_audio, language, step, temperature, length_scale, solver, cfg, ...)` を提供する。webui.py はこの API の薄いラッパー。
- **Phase 1 推論改善オプション**（すべてデフォルト無効＝既存挙動とビット一致、既存チェックポイント互換。詳細と A/B は `docs/architecture-improvement-research.md` §9）: `sway_coef`（Sway Sampling。**euler 等の固定ステップソルバー専用**、`-1.0` で学習時 cosine スケジューラと一致し低ステップでも dopri5 品質・約10倍高速）、`cfg_rescale`（過剰 CFG の飽和抑制、推奨 0.7）、`cfg_interval`（guidance 適用区間）、`ref_window_seconds` と `ref_audio` の `list[str]` 対応（複数参照窓平均。`get_style_vector`）、`slg_scale` ほか（Skip Layer Guidance。**評価の結果6層デコーダでは悪化し非推奨**、実装のみ残置）。webui の既定は推奨値（solver=euler / step=16 / sway_coef=−1.0 / cfg_rescale=0.7）。
- **Phase 2 / Phase 3 は再学習を伴う研究で、いずれもクローズ済み・不採用**（研究アーク全体の俯瞰は `docs/research-summary.md`、詳細は `docs/phase2-plan.md` / `docs/phase3-plan.md`）。Phase 2（レシピ = logit-normal timestep + EMA）は spk_cos が baseline 比 −0.020 で不採用。Phase 3（構造変更 = TLA-SA 補助話者整列損失 / MRTE 参照 mel への cross-attention）は両方とも既存の pooled style vector 到達点に後付け継続学習した結果 spk_cos −0.009 / −0.007 で不採用。**共通の失敗要因は「既に pooled style で学習済みの重みに新機構を後付けした」こと**で、pooled ボトルネックの緩和には後付け継続学習が原理的に向かないことを実証した（価値ある否定的結果。今後の方向性は別途計画）。
- **Phase 2/3 の実装は既定 False で残置**（いずれも既定でビット一致・baseline パラメータ数 31,644,545 不変。`use_mrte=True` のときのみ 33,225,345）: `models/tla_sa.py`（`TLASAHead`。推論経路・state_dict 不変の独立 DDP モジュール）、`reference_encoder.py` の `MelStyleEncoder.forward(return_sequence=True)`（pool 前系列を返す、param 0）、`diffusion_transformer.py` の `CrossAttention` クラスと `DiTConVBlock` の `use_cross_attn`/`cross_gate`（zero-init ゲート）、`estimator.py`/`flow_matching.py`/`model.py` の `use_mrte`・`ref_seq`/`ref_mask` 透過、`precompute_spk_emb.py`（オフライン SV 埋込の事前計算）、`api.py` の `StableTTSAPI(tts_model_config=...)`（per-checkpoint ModelConfig）。
- **g2p フロントエンド** (`text/`): `chinese_to_cnm3`（`text/mandarin.py`）、`english_to_ipa2`（`text/english.py`）、`japanese_to_ipa2`（`text/japanese.py`、[pyopenjtalk-plus](https://github.com/tsukumijima/pyopenjtalk-plus) ベース。`extract_fullcontext` は `use_vanilla=False` で呼び、plus の読み補正（Sudachi 同形異音語補正・「何」の ONNX 推定）を有効化している。自前の日本語事前学習と推論をこの設定で統一する方針（`docs/pretraining-plan.md`）。本家 checkpoint_0.pt は素の OpenJTalk の読みで学習されているため、本家モデルの厳密な再現評価時のみ `True` に戻す）。`text/symbols.py` の記号表が語彙を定義し、`len(symbols)` がモデルの `n_vocab` になる。`g2p_mapping` 辞書は `api.py` と `preprocess.py` の2箇所にあり、言語追加時は両方に反映が必要。
- **StableTTS 本体** (`models/model.py`): 4コンポーネント構成
  - `TextEncoder`（DiT ブロック）: 音素列 → mu_x（mel の事前分布）
  - `MelStyleEncoder`（`reference_encoder.py`）: 参照音声の mel → 話者スタイルベクトル c（gin_channels 次元）。全コンポーネントに条件付けされる
  - `DurationPredictor`: 音素ごとの継続長を予測。学習時は MAS（`monotonic_align/`、numba JIT 実装）で求めたアラインメントを教師とする
  - `CFMDecoder`（`flow_matching.py` + `estimator.py`）: flow matching で mel を精緻化。U-Net 風 long skip connection 付き DiT。ODE ソルバーは torchdiffeq（euler/dopri5 など選択可）で、推論 step 数（10〜25程度）が品質と速度のトレードオフ
- **CFG（Classifier-Free Guidance）**: 学習時に一定確率で話者ベクトルを学習可能な `fake_speaker` に置き換え、推論時に cfg 係数（webui デフォルト3）で誘導する
- **損失**: dur_loss + diff_loss（flow matching）+ prior_loss の単純和（`train.py`）
- **ボコーダ** (`vocoders/`): `vocos/` は学習可能なサブプロジェクト（専用の config/preprocess/train を持つ）、`ffgan/` は FireflyGAN の推論専用ラッパー（CC-BY-NC-SA・非商用）、`bigvgan/` は BigVGAN v2 の推論専用ラッパー（MIT・NVIDIA/BigVGAN を vendor。CUDA カーネルは除去し pure-PyTorch）。`api.py` の `get_vocoder()` で `'vocos'` / `'ffgan'` / `'bigvgan'` を切り替える。3者とも mel 仕様（44.1kHz/128mel/slaney）が一致し互換

### データフロー（学習）

`preprocess.py` が音声を mel（.pt ファイル）に事前変換し、JSON Lines 形式の filelist（`mel_path`, `phone`, `audio_path`, `text`, `mel_length`）を出力する。学習時（`datas/dataset.py`）は mel と音素のみ読み込み、`mel_length` を `DistributedBucketSampler`（`datas/sampler.py`）のバケット割当に使う。参照音声は別ファイルではなく、`collate_fn` がターゲット mel からランダムスライスした断片（z）を reference encoder に渡す。

### 設定の共有

`config.py` の `MelConfig` / `ModelConfig` / `VocosConfig` は前処理・学習・推論の全てで共有される。チェックポイントはこれらの値に依存するため、変更すると既存チェックポイントと非互換になる。v1.1 のデフォルトは 44.1kHz / 128 mel（slaney スケール）、エンコーダ3層・デコーダ6層。設定は CLI ではなく dataclass を直接編集する方式で、Phase 3 の `ModelConfig.use_mrte`（state_dict にキーが増えるアーキ設定）・`TrainConfig.use_tla_sa` / `timestep_sampling` / `use_ema`、および学習高速化の Tier1-2 フラグ（`use_amp`/`grad_clip`/`use_fused_optimizer`/`num_workers`/`prefetch_factor`/`use_compile`/`use_gpu_mas`）もここで切り替える。**推論安全（既存チェックポイントを strict ロード可能）な既定は `use_mrte=False`・`use_tla_sa=False`・`timestep_sampling="cosine"`・`use_ema=False`**。Tier1-2 は数値精度のみ変えチェックポイント形式・n_vocab・推論経路は不変。

## 注意点

- Windows で開発されており、`train.py` は Windows では DDP バックエンドに gloo を使う分岐がある
- Windows では `uv sync` 前に webui.py 等の Python プロセスを終了すること（torch の DLL がロックされ上書きに失敗する）
- 日本語 g2p は pyopenjtalk-plus（辞書 wheel 同梱、ネットワーク DL なし）。「何」の読み推定に onnxruntime が必要なため extra（`pyopenjtalk-plus[onnxruntime]`）で導入している。辞書がカスタム naist-jdic のため、旧 pyopenjtalk-prebuilt とはアクセント句境界が一部異なる（音素セットは互換、n_vocab 不変）
- jieba の `pkg_resources` DeprecationWarning は無害（pkg_resources 不在時のフォールバックも jieba 側にある）
- `webui.py` は冒頭で `TMPDIR=./temps` を設定し、TTS チェックポイントパス（`./checkpoints/tsukuyomi_ft200.pt`）とボコーダ種別（`bigvgan`）がハードコードされている。別のチェックポイントを使う場合はここを書き換える。UI のサンプリング既定は Phase 1 推奨（solver=euler / step=16 / sway_coef=−1.0 / cfg_rescale=0.7）
- `preprocess.py` の `DataConfig.language` はフォークで `'japanese'` がデフォルトになっている
- `recipes/` にはオープンソースデータセット（LibriTTS、AiSHELL3 など）から filelist を生成するスクリプトがある
- v1.0 時代のチェックポイント（旧 8層/8層構成や `vocos_pytorch/` の vocoder.pt）は v1.1 のコードとは非互換
