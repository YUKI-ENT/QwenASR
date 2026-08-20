# QwenASRBench

Qwen3-ASR 0.6B / 1.7B を、日本語（特に診察音声）で比較する独立したCLIベンチマークです。Transformers backendだけを使い、認識文、処理時間、RTF、PyTorchおよび `nvidia-smi` のVRAM値をJSON/CSVへ保存します。

このプロジェクトは [YUKI-ENT/SpeechSummarizer](https://github.com/YUKI-ENT/SpeechSummarizer) とコード・仮想環境・依存パッケージを共有しません。既存環境を変更せず、不要ならこのディレクトリだけを削除できます。CUDA ToolkitやNVIDIA driverを追加・変更する手順も含みません。

## 対象環境

- Ubuntu 24.04 / Python 3.12
- NVIDIA RTX 5080 16GB（CUDA対応PyTorch）
- ffmpeg / ffprobe（MP3、M4A、および一部形式の長さ取得に使用）
- モデル初回取得時のみインターネット接続

Forced Aligner、timestamp、vLLM、Web UI、FastAPI、話者分離は現段階では使いません。

## セットアップ

SpeechSummarizerとは別のディレクトリ、別のvenvで実行してください。

```bash
cd /path/to/QwenASRBench
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

まず [PyTorch公式のインストール選択画面](https://pytorch.org/get-started/locally/) で、Linux / Pip / Python と環境に合うCUDA wheelを確認して導入します。この構成で固定したPyTorch 2.7.0にはCUDA 12.8 wheelがあり、RTX 5080向けの基準構成にできます。システムへCUDA Toolkitを追加する必要はありません。

```bash
pip install torch==2.7.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

`requirements.txt` は `torch==2.7.0` / `torchaudio==2.7.0`、公式 `qwen-asr==0.0.6` と、その公式依存である `transformers==4.57.6`、`accelerate==1.12.0` を固定しています。PyTorchを先にCUDA 12.8 indexから入れることで、通常のPyPI wheelへ置き換わるのを防ぎます。将来PyTorch版を上げる場合も、torchとtorchaudioは同じ版にしてください。

Ubuntuでffmpegが未導入の場合に限り、OS標準パッケージを導入します（CUDA関連パッケージは導入しません）。

```bash
sudo apt update
sudo apt install ffmpeg
```

CUDA確認:

```bash
nvidia-smi
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

## 単一ファイルの実行

```bash
python app.py --model 0.6b test_audio/sample.wav
python app.py --model 1.7b test_audio/sample.wav
python app.py --model 0.6b --unload-after test_audio/sample.m4a
```

入力は WAV / MP3 / M4A / FLAC です。MP3とM4Aは一時的なmono 16 kHz WAVへ変換し、終了時に削除します。結果は既定で `results/YYYYMMDD_HHMMSS_...json` に保存されます。モデルは既定で結果保存前に明示的に解放されます。

患者情報を含む音声は同梱していません。手元の匿名化済み音声を `test_audio/sample.wav` として配置してから実行してください。

主な表示値:

- `before load` / `after load` / `inference peak` / `after inference` / `after unload`
- PyTorch allocated / reserved memory
- `nvidia-smi` が取得できる場合は現在プロセスの使用量（推論中は短周期ポーリングによる参考ピーク値）

`nvidia-smi` のポーリングピークはサンプリング値のため、極短時間の最大値を取り逃す場合があります。PyTorch peakは `torch.cuda.reset_peak_memory_stats()` と `max_memory_allocated()` で計測します。

## 設定

`config.json` を編集します。主な初期値:

```json
{
  "device": "cuda:0",
  "dtype": "bfloat16",
  "model": "Qwen/Qwen3-ASR-0.6B",
  "language": "Japanese",
  "max_new_tokens": 256,
  "max_inference_batch_size": 1,
  "model_cache_dir": "models",
  "offline": false,
  "unload_after": true
}
```

`--model` を省略すると `model` を使います。FlashAttention 2は必須ではなく、標準PyTorch attentionで動作します。

## モデルの事前ダウンロードとオフライン運用

オンライン環境で専用venvを有効にし、モデルをプロジェクト内へ取得します。

```bash
pip install "huggingface_hub[cli]"
huggingface-cli download Qwen/Qwen3-ASR-0.6B --local-dir models/Qwen3-ASR-0.6B
huggingface-cli download Qwen/Qwen3-ASR-1.7B --local-dir models/Qwen3-ASR-1.7B
```

`config.json` の `local_model_paths` を設定します。

```json
"local_model_paths": {
  "0.6b": "models/Qwen3-ASR-0.6B",
  "1.7b": "models/Qwen3-ASR-1.7B"
}
```

その後は次のどちらかで外部アクセスを禁止できます。

```bash
python app.py --offline --model 0.6b test_audio/sample.wav
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python app.py --model 0.6b test_audio/sample.wav
```

`--offline` は `local_files_only=True` もモデルロードへ渡します。

## 複数ファイルベンチマーク

フォルダー直下の対応音声を、0.6B、1.7Bの順に処理します。各モデルは一度だけロードし、全ファイル処理後に解放します。

```bash
python benchmark.py test_audio/
python benchmark.py --models 0.6b test_audio/
python benchmark.py --offline --continue-on-error test_audio/
```

`results/` に汎用的な列構成のJSONとUTF-8 BOM付きCSVを保存します。将来 Faster-Whisper / Kotoba Whisper を加えられるよう、`engine`、`backend`、`model` を別フィールドにしています。

医療用途ではCERだけで判断せず、薬剤名、疾患名、左右、数字、用量、体温、日付、短い相づち、小児・高齢者音声、雑音、発話重複を同じ音源で目視比較してください。特に「左耳ではなく右耳」など、意味上の重大度が高い誤りは別途記録するのが安全です。本ツールは研究・評価用で、医療判断に認識結果を直接使用するものではありません。

## エラーと終了コード

想定可能な失敗は `Error:` で短く表示し、終了コード2を返します。CUDA未検出、ローカルモデル不在、未対応形式、ffmpeg不在、ロード失敗、推論失敗、GPU OOMを区別します。1.7BでOOMの場合は0.6Bを試す案内を表示します。

## 開発時の確認

重いモデルをロードしない単体テスト:

```bash
python -m unittest discover -s tests -v
python -m compileall app.py benchmark.py audio_utils.py gpu_monitor.py qwen_asr_engine.py cli_common.py
```

実機での最初の到達確認:

```bash
python app.py --model 0.6b test_audio/sample.wav
python app.py --model 1.7b test_audio/sample.wav
```

認識精度とVRAMの比較が良好なら、次段階でHTTP APIを独立プロセスとして追加し、SpeechSummarizerから呼び出す構成へ拡張できます。
