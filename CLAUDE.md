# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## プロジェクト概要

StableTTSEx は [KdaiP/StableTTS](https://github.com/KdaiP/StableTTS) のフォーク（upstream v1.1 ベース）。flow-matching と DiT を組み合わせた軽量 TTS モデル（約31Mパラメータ）で、単一チェックポイントで中国語・英語・日本語に対応する。話者IDは使わず、reference encoder が参照音声から話者性を抽出するゼロショット方式。フォーク固有の変更は、日本語をデフォルトにする設定・`generate-audio-list.py`・uv による依存管理・日本語 g2p の pyopenjtalk-plus 化など小規模。

テストやリンタの設定は存在しない。設定は CLI 引数ではなく、`config.py` や各スクリプト冒頭の dataclass を直接編集する方式。

依存は uv（`pyproject.toml` + `uv.lock`）で管理する。Python は 3.11 固定（`.python-version`。環境再現性のための固定で、引き上げるなら torch cu128/numba/numpy の対応確認と venv 再構築を伴う）、numpy は 2 未満固定（旧 pyopenjtalk-prebuilt 時代の名残。緩和は依存一式の再検証とセットで別コミットにて）。torch/torchaudio は 2.8 系 + cu128 を `[tool.uv.index]` / `[tool.uv.sources]` で自動解決する（別途インストール不要。torchaudio 2.9+ は load/save が TorchCodec 委譲になりコード非互換なので上げない）。

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

# 推論
uv run python webui.py # Gradio WebUI（share=True で起動、デフォルト言語は日本語）
# または inference.ipynb / api.py の StableTTSAPI を使用
```

事前学習済みモデルの配置（HuggingFace KdaiP/StableTTS1.1 から取得）:
- TTS 本体 `checkpoint_0.pt` → `./checkpoints/`
- ボコーダ → `./vocoders/pretrained/`（`vocos.pt` または fishaudio の `firefly-gan-base-generator.ckpt`。webui.py のデフォルトは ffgan）

## アーキテクチャ

### 音声合成パイプライン全体

```
テキスト → g2p（言語別音素化）→ 音素ID列（intersperse で blank=0 を挿入）
  → StableTTS（テキスト→mel）→ ボコーダ（mel→波形, 44.1kHz）
```

- **推論の入口は `api.py` の `StableTTSAPI`**: TTS モデル・ボコーダ・g2p をまとめてロードし、`inference(text, ref_audio, language, step, temperature, length_scale, solver, cfg)` を提供する。webui.py はこの API の薄いラッパー。
- **g2p フロントエンド** (`text/`): `chinese_to_cnm3`（`text/mandarin.py`）、`english_to_ipa2`（`text/english.py`）、`japanese_to_ipa2`（`text/japanese.py`、[pyopenjtalk-plus](https://github.com/tsukumijima/pyopenjtalk-plus) ベース。`extract_fullcontext` は `use_vanilla=True` で呼び、plus 独自の読み後処理を無効化して事前学習チェックポイント学習時の素の OpenJTalk 挙動に揃えている）。`text/symbols.py` の記号表が語彙を定義し、`len(symbols)` がモデルの `n_vocab` になる。`g2p_mapping` 辞書は `api.py` と `preprocess.py` の2箇所にあり、言語追加時は両方に反映が必要。
- **StableTTS 本体** (`models/model.py`): 4コンポーネント構成
  - `TextEncoder`（DiT ブロック）: 音素列 → mu_x（mel の事前分布）
  - `MelStyleEncoder`（`reference_encoder.py`）: 参照音声の mel → 話者スタイルベクトル c（gin_channels 次元）。全コンポーネントに条件付けされる
  - `DurationPredictor`: 音素ごとの継続長を予測。学習時は MAS（`monotonic_align/`、numba JIT 実装）で求めたアラインメントを教師とする
  - `CFMDecoder`（`flow_matching.py` + `estimator.py`）: flow matching で mel を精緻化。U-Net 風 long skip connection 付き DiT。ODE ソルバーは torchdiffeq（euler/dopri5 など選択可）で、推論 step 数（10〜25程度）が品質と速度のトレードオフ
- **CFG（Classifier-Free Guidance）**: 学習時に一定確率で話者ベクトルを学習可能な `fake_speaker` に置き換え、推論時に cfg 係数（webui デフォルト3）で誘導する
- **損失**: dur_loss + diff_loss（flow matching）+ prior_loss の単純和（`train.py`）
- **ボコーダ** (`vocoders/`): `vocos/` は学習可能なサブプロジェクト（専用の config/preprocess/train を持つ）、`ffgan/` は FireflyGAN の推論専用ラッパー（本リポジトリでは学習・設定変更不可）。`api.py` の `get_vocoder()` で `'vocos'` / `'ffgan'` を切り替える

### データフロー（学習）

`preprocess.py` が音声を mel（.pt ファイル）に事前変換し、JSON Lines 形式の filelist（`mel_path`, `phone`, `audio_path`, `text`, `mel_length`）を出力する。学習時（`datas/dataset.py`）は mel と音素のみ読み込み、`mel_length` を `DistributedBucketSampler`（`datas/sampler.py`）のバケット割当に使う。参照音声は別ファイルではなく、`collate_fn` がターゲット mel からランダムスライスした断片（z）を reference encoder に渡す。

### 設定の共有

`config.py` の `MelConfig` / `ModelConfig` / `VocosConfig` は前処理・学習・推論の全てで共有される。チェックポイントはこれらの値に依存するため、変更すると既存チェックポイントと非互換になる。v1.1 のデフォルトは 44.1kHz / 128 mel（slaney スケール）、エンコーダ3層・デコーダ6層。

## 注意点

- Windows で開発されており、`train.py` は Windows では DDP バックエンドに gloo を使う分岐がある
- Windows では `uv sync` 前に webui.py 等の Python プロセスを終了すること（torch の DLL がロックされ上書きに失敗する）
- 日本語 g2p は pyopenjtalk-plus（辞書 wheel 同梱、ネットワーク DL なし）。import 時の「ONNX Runtime is not installed」警告は無害（`use_vanilla=True` で読み推定機能を使わないため）。辞書がカスタム naist-jdic のため、旧 pyopenjtalk-prebuilt とはアクセント句境界が一部異なる（音素セットは互換、n_vocab 不変）
- jieba の `pkg_resources` DeprecationWarning は無害（pkg_resources 不在時のフォールバックも jieba 側にある）
- `webui.py` は冒頭で `TMPDIR=./temps` を設定し、TTS チェックポイントパス（`./checkpoints/checkpoint_0.pt`）とボコーダ種別がハードコードされている。別のチェックポイントを使う場合はここを書き換える
- `preprocess.py` の `DataConfig.language` はフォークで `'japanese'` がデフォルトになっている
- `recipes/` にはオープンソースデータセット（LibriTTS、AiSHELL3 など）から filelist を生成するスクリプトがある
- v1.0 時代のチェックポイント（旧 8層/8層構成や `vocos_pytorch/` の vocoder.pt）は v1.1 のコードとは非互換
