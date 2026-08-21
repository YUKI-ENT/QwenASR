# QwenASR

Qwen3-ASR 0.6B / 1.7Bの独立したCLIベンチマークとlocalhost HTTP APIです。Transformers backendだけを使い、CLIでは認識文、処理時間、RTF、PyTorchおよび `nvidia-smi` のVRAM値をJSON/CSVへ保存します。APIはSpeechSummarizerから利用できますが、プロセス、仮想環境、モデル、GPU障害の影響範囲はSpeechSummarizerから分離されています。

このプロジェクトは [YUKI-ENT/SpeechSummarizer](https://github.com/YUKI-ENT/SpeechSummarizer) とコード・仮想環境・依存パッケージを共有しません。既存環境を変更せず、不要ならこのディレクトリだけを削除できます。CUDA ToolkitやNVIDIA driverを追加・変更する手順も含みません。

## 対象環境

- Ubuntu 24.04 / Python 3.12
- NVIDIA RTX 5080 16GB（CUDA対応PyTorch）
- ffmpeg / ffprobe（MP3、M4A、および一部形式の長さ取得に使用）
- モデル初回取得時のみインターネット接続

Forced Aligner、timestamp、vLLM、Web UI、話者分離は現段階では使いません。

## セットアップ

SpeechSummarizerとは別のディレクトリ、別のvenvで実行してください。

```bash
cd /path/to/QwenASR
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

## localhost HTTP API

APIは起動時に1モデルだけをロードし、停止まで常駐させます。複数リクエストは上限付きFIFOキューで待機し、GPU推論は必ず1件ずつ実行されます。VAD、音声分割、品質判定、Whisper fallback、認識結果の保存は行いません。

API依存のFastAPI、Uvicorn、python-multipartは `requirements.txt` に含まれるため、通常のセットアップで導入されます。既存環境を更新する場合は、仮想環境内で次を実行します。

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

`config.json` のトップレベルに次を追加します。`model_alias` が空または未指定なら、トップレベルの `model` が使われます。APIは `unload_after` を無視しますが、既存CLIは従来どおりこの値を使います。

```json
"api": {
  "host": "127.0.0.1",
  "port": 8010,
  "model_alias": "1.7b",
  "max_queue_size": 20,
  "request_timeout_sec": 30,
  "max_audio_sec": 30,
  "max_upload_mib": 10,
  "max_context_chars": 2000
}
```

起動と停止:

```bash
source .venv/bin/activate
python server.py --config config.json
# 停止は Ctrl+C
```

アプリケーションバージョンは `version.py` の `APP_VERSION`で、`YYYYMMDD` 形式（例: `20260821`）で一元管理します。`python server.py --version` で確認でき、`/health` と `/ready` の `app_version` にも同じ値が返ります。API互換性用の `schema_version` とは別の値です。

`--host` と `--port` でlisten先を上書きできますが、hostは `127.0.0.1`、`localhost`、`::1` のみ許可されます。Uvicorn workerは1つに固定されます。

動作確認:

```bash
curl -s http://127.0.0.1:8010/health | python -m json.tool
curl -s http://127.0.0.1:8010/ready | python -m json.tool
curl -s -X POST http://127.0.0.1:8010/transcribe \
  -F 'audio=@test_audio/sample.wav;type=audio/wav' \
  -F 'request_id=manual-test-001' \
  -F 'language=Japanese' \
  -F 'context=耳鼻咽喉科の診察会話' | python -m json.tool
```

`POST /transcribe` はWAVの `multipart/form-data` だけを受け付け、認識文は `text` で返します。レスポンスには `schema_version: 1`、音声長、queue/推論/全体時間、RTFが含まれます。Qwenが提供しないconfidenceやWhisper風の疑似確率は返しません。`/ready` の `queue_depth` は待機中の件数で、実行中の1件は含みません。

無音時などにQwenが `context` 全文を認識文として繰り返す場合があるため、APIは応答前にcontextと完全一致する文言をすべて除去します。除去後に発話が残らなければ `text` は空文字になります。類似した発話の誤削除を避けるため、あいまい一致では除去しません。これはcontext echo対策であり、一般的な無音検出の代替ではありません。

このAPIには認証やTLSがありません。患者音声を扱う可能性があるため、リバースプロキシ等を使ってLANやインターネットへ公開しないでください。uploadはサーバー生成名の一時ファイルとして扱われ、成功・失敗・タイムアウトのいずれでも推論が終わり次第削除されます。

RTX 5080 16GB / bfloat16での既存実測では、0.6Bはロード後約2.1 GiB・推論時約2.2 GiB、1.7Bはロード後約4.7 GiB・推論時約4.9 GiBのプロセスVRAM使用量でした。音声長や環境で変動するため、余裕を確保してください。

## Windows向けEXEの作成

Windows用のEXEは、対象のWindows PC（または同等のWindows環境）でビルドします。Linux上のPyInstallerからWindows用EXEをクロスビルドすることはできません。PythonやモデルをEXEへ埋め込むのではなく、CUDA DLLを含む依存ファイル一式を同じフォルダーに置く `onedir` 形式です。PyTorchが大きいため配布フォルダーも数GBになりますが、`onefile` より起動とDLL読込が安定します。

前提:

- 64-bit Windows 11
- 64-bit Python 3.12
- 対応するNVIDIAドライバー
- PowerShell

PowerShellでプロジェクトへ移動し、新しいvenvからビルドします。

```powershell
py -3.12 -m venv .venv-win
.\.venv-win\Scripts\Activate.ps1
Set-ExecutionPolicy -Scope Process Bypass
.\packaging\build_windows.ps1 -InstallDependencies
```

既に依存パッケージを導入済みなら `-InstallDependencies` は省略できます。CLIだけなら `-Target cli`、APIだけなら `-Target server` を指定できます。

出力先:

```text
dist\QwenASR\QwenASR.exe
dist\QwenASR-Server\QwenASR-Server.exe
```

各フォルダーには `config.json`、空の `models` / `results` も作成されます。フォルダー内の一部だけでなく、フォルダー全体を配布してください。モデルはサイズが大きいため自動では同梱しません。オフライン運用では、事前取得したモデルをたとえば `models\Qwen3-ASR-0.6B` へコピーし、同梱された `config.json` を次のように変更します。

```json
"offline": true,
"local_model_paths": {
  "0.6b": "models/Qwen3-ASR-0.6B",
  "1.7b": ""
}
```

実行例:

```powershell
cd dist\QwenASR
.\QwenASR.exe --model 0.6b C:\audio\sample.wav

cd ..\QwenASR-Server
.\QwenASR-Server.exe
```

EXE起動時の既定の `config.json`、`models`、`results` は、カレントディレクトリではなくEXEのあるフォルダーを基準に解決されます。MP3/M4AをCLIで扱う場合は、Windows版の `ffmpeg.exe` と `ffprobe.exe` をEXEと同じフォルダー（または `bin` サブフォルダー）に置くか、PATHへ追加してください。APIはWAVだけを受け付けるため通常は不要です。

初回確認はモデルを同梱してビルドし直すのではなく、`dist` のフォルダーへモデルをコピーして行います。Windows Defender等が未署名の自作EXEを警告する場合があります。第三者へ配布する場合はコード署名と、PyTorch・FFmpeg・モデルを含む各ライセンスの確認が必要です。

### APIのトラブルシューティング

- モデル未配置 / offline時のキャッシュ不足: `local_model_paths` の相対パスが `config.json` 基準で正しいか、モデルが全部ダウンロード済みかを確認します。初回取得が必要なら一度 `offline: false` で取得します。
- CUDA未検出: `nvidia-smi` と `python -c "import torch; print(torch.cuda.is_available())"` を確認し、CUDA版PyTorchが導入されているか確認します。
- GPU OOM: 他のGPUプロセスを確認し、1.7Bから0.6Bへ変更します。APIはHTTP 503 / `gpu_out_of_memory` を返します。
- queue full: 待機キュー上限でHTTP 429 / `queue_full` になります。呼び出し側で間隔を空けて再試行するか、GPUの処理速度とメモリに余裕がある場合だけ `max_queue_size` を調整します。
- timeout: キュー待ちを含む `request_timeout_sec` でHTTP 504 / `request_timeout` になります。すでに始まったGPU推論は安全のため強制中断せず、完了後に結果を破棄してworkerを継続します。

## モデルの事前ダウンロードとオフライン運用

### Hugging Faceからブラウザーで手動ダウンロードする場合

インターネットへ接続できるPCで、使用するモデルのHugging Faceページを開き、`Files and versions` に表示されるファイルをダウンロードします。

- [Qwen/Qwen3-ASR-0.6B](https://huggingface.co/Qwen/Qwen3-ASR-0.6B/tree/main)
- [Qwen/Qwen3-ASR-1.7B](https://huggingface.co/Qwen/Qwen3-ASR-1.7B/tree/main)

各ファイルのページで `Download` を押してください。`.safetensors` はサイズが大きいため、保存完了後にファイルサイズがHugging Faceの表示と大きく違わないことを確認します。Git LFS/Xetのポインタだけを保存した数百byte程度のファイルではモデルをロードできません。使用するモデルだけをダウンロードすればよく、0.6Bと1.7Bの両方を必ず配置する必要はありません。

次のように、モデルごとに1つのフォルダーを作り、Hugging Face上のファイル名を変えずに同じ階層へ保存します。

```text
QwenASR/
├─ config.json
└─ models/
   ├─ Qwen3-ASR-0.6B/
   │  ├─ config.json
   │  ├─ generation_config.json
   │  ├─ preprocessor_config.json
   │  ├─ tokenizer_config.json
   │  ├─ chat_template.json
   │  ├─ merges.txt
   │  ├─ vocab.json
   │  └─ model.safetensors
   └─ Qwen3-ASR-1.7B/
      ├─ config.json
      ├─ generation_config.json
      ├─ preprocessor_config.json
      ├─ tokenizer_config.json
      ├─ chat_template.json
      ├─ merges.txt
      ├─ vocab.json
      ├─ model.safetensors.index.json
      ├─ model-00001-of-00002.safetensors
      └─ model-00002-of-00002.safetensors
```

モデルリポジトリの `README.md` と `.gitattributes` は実行には不要です。それ以外はモデルの更新で構成が変わる可能性があるため、`Files and versions` に追加ファイルがあれば一緒に保存してください。ダウンロード後、`models` フォルダーごとUSBメモリ等でオフラインPCのQwenASRフォルダーへコピーします。EXE版では `QwenASR.exe` または `QwenASR-Server.exe` と同じフォルダーにある `models` へ配置します。

### `config.json` の設定

QwenASR側の `config.json`（モデルフォルダー内の同名ファイルではありません）を次のように設定します。

```json
{
  "model": "Qwen/Qwen3-ASR-0.6B",
  "model_cache_dir": "models",
  "offline": true,
  "api": {
    "model_alias": "1.7b"
  },
  "models": {
    "0.6b": "Qwen/Qwen3-ASR-0.6B",
    "1.7b": "Qwen/Qwen3-ASR-1.7B"
  },
  "local_model_paths": {
    "0.6b": "models/Qwen3-ASR-0.6B",
    "1.7b": "models/Qwen3-ASR-1.7B"
  }
}
```

`config.json.sample` をコピーして使う場合は、既存のほかの設定を残し、`offline` と `local_model_paths` の値だけを変更します。

- `local_model_paths` のキーは `--model 0.6b` / `--model 1.7b` および `api.model_alias` に対応します。値が空でなければ、オンライン用の `models` よりローカルパスが優先されます。
- 相対パスはQwenASR側の `config.json` があるフォルダーを基準に解決されます。WindowsでもJSON内では上例のように `/` を使えます。
- 絶対パスも指定できます。Windowsの `D:\ASRModels\Qwen3-ASR-0.6B` を記載する場合は、JSONでは `D:\\ASRModels\\Qwen3-ASR-0.6B` のように `\` を2つ重ねます。
- 片方のモデルだけ配置する場合は、未配置側を空文字（例: `"1.7b": ""`）のままにし、実行時に配置済みのaliasを選択します。
- CLIで `--model` を省略するとトップレベルの `model` が使われます。オフラインで省略実行したい場合は、`model` 自体を `"models/Qwen3-ASR-0.6B"` のようなローカルパスに変更してください。
- APIは `api.model_alias` のモデルを起動時にロードします。上例は1.7Bなので、0.6Bだけ配置した場合は `"model_alias": "0.6b"` に変更します。

`offline: true` にするとHugging Faceへのアクセスを禁止し、モデルロードに `local_files_only=True` を渡します。CLIでは次のように起動します。

```powershell
python app.py --model 0.6b test_audio/sample.wav
python app.py --model 1.7b test_audio/sample.wav
python server.py --config config.json
```

### CLIで一括ダウンロードする場合

ブラウザーで1ファイルずつ保存する代わりに、オンライン環境で専用venvを有効にし、次のコマンドでモデル一式を取得できます。

```bash
pip install "huggingface_hub[cli]"
huggingface-cli download Qwen/Qwen3-ASR-0.6B --local-dir models/Qwen3-ASR-0.6B
huggingface-cli download Qwen/Qwen3-ASR-1.7B --local-dir models/Qwen3-ASR-1.7B
```

取得後は、前項と同じように `config.json` の `local_model_paths` と `offline` を設定します。

```json
"local_model_paths": {
  "0.6b": "models/Qwen3-ASR-0.6B",
  "1.7b": "models/Qwen3-ASR-1.7B"
}
```

`config.json` を変更せず一時的にオフラインモードを指定する場合は、次のどちらかを使用できます。

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
python -m compileall app.py benchmark.py server.py api_service.py audio_utils.py gpu_monitor.py qwen_asr_engine.py cli_common.py
```

実機での最初の到達確認:

```bash
python app.py --model 0.6b test_audio/sample.wav
python app.py --model 1.7b test_audio/sample.wav
```

実機API確認ではサーバーを起動し、同じWAVを連続2回 `POST /transcribe` して、モデルが再ロードされないこともログとVRAMで確認します。
