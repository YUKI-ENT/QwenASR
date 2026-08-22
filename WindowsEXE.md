# QwenASR-Server.exe Windows運用ガイド

## 1. 対象と用途

この文書は、Windows版 `QwenASR-Server.exe` をSpeechSummarizerからローカルHTTPサーバーとして起動・停止するためのガイドです。`QwenASR.exe`（単発実行CLI）は対象外です。

`QwenASR-Server.exe` は起動時にQwen3-ASRモデルを1つだけロードし、終了までGPUメモリ上に常駐させます。SpeechSummarizerから受け取ったWAVをFIFOキューへ入れ、推論は常に1件ずつ実行します。

サーバーは次の処理を行いません。

- VAD、発話区間の分割
- 音声品質の判定
- Whisperへのフォールバック
- 認識結果の恒久保存
- リクエストごとのモデル切替
- 設定ファイルの自動再読込

これらは必要に応じてSpeechSummarizer側で管理します。

## 2. 配置

PyInstallerの `onedir` 形式なので、EXEだけを取り出さず、`QwenASR-Server` フォルダー全体を配置してください。

```text
QwenASR-Server\
├─ QwenASR-Server.exe
├─ config.json
├─ _internal\
└─ models\
   ├─ Qwen3-ASR-0.6B\
   └─ Qwen3-ASR-1.7B\
```

既定の `config.json` は、カレントディレクトリではなく `QwenASR-Server.exe` と同じフォルダーから読み込まれます。モデルパスなどの相対パスは、読み込んだ `config.json` があるフォルダーを基準に解決されます。

APIはWAVだけを受け付けるため、通常はFFmpegを配置する必要はありません。

## 3. コマンドラインオプション

```powershell
.\QwenASR-Server.exe [--config PATH] [--model ALIAS] [--host HOST] [--port PORT]
```

| オプション | 内容 | 既定値 |
|---|---|---|
| `-h`, `--help` | ヘルプを表示して終了 | - |
| `--version` | アプリケーションバージョンを表示して終了 | - |
| `--config PATH` | 使用する設定JSONを指定 | EXEと同じフォルダーの `config.json` |
| `--model ALIAS` | 起動時にロードするモデルaliasを今回の起動だけ上書き | `api.model_alias`、未設定ならトップレベルの `model` |
| `--host HOST` | `config.json` の `api.host`を今回の起動だけ上書き | 設定値、未設定なら `127.0.0.1` |
| `--port PORT` | `config.json` の `api.port`を今回の起動だけ上書き | 設定値、未設定なら `8010` |

起動例:

```powershell
cd D:\SSWin\QwenASR\dist\QwenASR-Server
.\QwenASR-Server.exe
```

設定ファイルとポートを明示する例:

```powershell
.\QwenASR-Server.exe --config "D:\SpeechSummarizer\config\qwen-asr.json" --model 1.7b --port 8010
```

`--host` には `localhost` または有効なIPv4/IPv6アドレスを指定できます。初期状態の `127.0.0.1` は同じPCからだけ接続できます。LANから接続する場合は、サーバーPCに実際に割り当てられたLAN側IP（例: `192.168.253.10`）か、全IPv4インターフェースを表す `0.0.0.0` を指定します。指定した個別IPがPCに割り当てられていなければ起動できません。

LANへ公開する場合は、Windows Firewallの受信規則をPrivateまたはDomainプロファイルに限定し、接続元をSpeechSummarizer端末のIPだけに絞ってください。このAPIには認証やTLSがないため、インターネットへは公開しないでください。

## 4. config.json

### 4.1 推奨設定例

ローカルに両モデルを配置し、1.7Bを使用する例です。

```json
{
  "device": "cuda:0",
  "dtype": "bfloat16",
  "model": "Qwen/Qwen3-ASR-0.6B",
  "language": "Japanese",
  "max_new_tokens": 256,
  "max_inference_batch_size": 1,
  "model_cache_dir": "models",
  "offline": true,
  "unload_after": true,
  "nvidia_smi_poll_interval_sec": 0.05,
  "api": {
    "host": "127.0.0.1",
    "port": 8010,
    "model_alias": "1.7b",
    "max_queue_size": 20,
    "request_timeout_sec": 30,
    "max_audio_sec": 30,
    "max_upload_mib": 10,
    "max_context_chars": 2000,
    "max_request_id_chars": 128,
    "max_language_chars": 64
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

JSONにはコメントや末尾カンマを記述できません。Windowsの絶対パスを `\` で書く場合は、`"D:\\ASRModels\\Qwen3-ASR-0.6B"` のように二重化してください。相対パスでは `/` を使うと簡潔です。

### 4.2 モデル・推論設定

| 設定 | 内容 | 制約・補足 |
|---|---|---|
| `device` | 推論デバイス | 通常は `cuda:0`。CPUなら `cpu` |
| `dtype` | 推論精度 | `bfloat16` / `bf16` / `float16` / `fp16` / `float32` / `fp32` |
| `model` | `api.model_alias`が空の場合のモデル | 空でない文字列が必須 |
| `language` | リクエストで省略した場合の言語 | 通常は `Japanese`。文字列または `null` |
| `max_new_tokens` | 生成する最大トークン数 | 1以上の整数 |
| `max_inference_batch_size` | モデル内部の最大推論バッチサイズ | 1以上の整数。HTTP推論自体は常に直列 |
| `model_cache_dir` | Hugging Faceキャッシュ | 相対パスは設定ファイル基準 |
| `offline` | 外部ダウンロードを禁止するか | ローカル運用では `true` 推奨 |
| `unload_after` | 単発CLI用の設定 | サーバーでは無視され、モデルは常駐 |
| `nvidia_smi_poll_interval_sec` | GPU使用量監視の間隔（秒） | 0より大きい数値 |
| `models` | aliasからHugging FaceモデルIDへの対応 | JSON object |
| `local_model_paths` | aliasからローカルモデルへの対応 | JSON object。空文字なら `models` を使用 |

CPUで `float16` は使用できません。CPU利用時は `dtype` を `float32` にしてください。

### 4.3 API設定

| 設定 | 内容 | 既定値・範囲 |
|---|---|---|
| `api.host` | Listen先 | `127.0.0.1`。`localhost` または有効なIPv4/IPv6アドレス |
| `api.port` | Listenポート | `8010`、1～65535 |
| `api.model_alias` | 起動時にロードするモデルalias | 空または未指定ならトップレベルの `model` |
| `api.max_queue_size` | 推論待ちキューの上限 | `20`、1～10000。実行中の1件は含まない |
| `api.request_timeout_sec` | キュー待ちを含む要求タイムアウト | `30`、0より大きい数値 |
| `api.max_audio_sec` | WAV音声長の上限（秒） | `30`、0より大きい数値 |
| `api.max_upload_mib` | WAVアップロード容量の上限（MiB） | `10`、0より大きい数値 |
| `api.max_context_chars` | `context` の最大文字数 | `2000`、1～1000000 |
| `api.max_request_id_chars` | `request_id` の最大文字数 | `128`、1～4096 |
| `api.max_language_chars` | `language` の最大文字数 | `64`、1～1024 |

## 5. モデルの選択

モデルは次の優先順位で決まります。

1. `--model ALIAS` が指定されていれば、`api.model_alias` を今回の起動だけ上書きする
2. 選択されたaliasの `local_model_paths` を確認する
3. `local_model_paths` の値が空でなければ、そのローカルモデルを使用する
4. ローカルパスが空なら、同じaliasの `models` にあるモデルIDを使用する
5. `--model` と `api.model_alias` がどちらも空または未指定なら、トップレベルの `model` を使用する

コマンドラインで選択する場合:

```powershell
.\QwenASR-Server.exe --config config.json --model 0.6b
.\QwenASR-Server.exe --config config.json --model 1.7b
```

0.6Bへ切り替える場合:

```json
"api": {
  "model_alias": "0.6b"
}
```

1.7Bへ切り替える場合:

```json
"api": {
  "model_alias": "1.7b"
}
```

モデルaliasは音声認識リクエストでは指定できません。稼働中のモデルを切り替える場合は、サーバーを停止し、別の `--model` で再起動します。`--model` は設定ファイルを書き換えません。

参考として、RTX 5080 16GB / bfloat16でのプロセスVRAM実測値は、0.6Bがロード後約2.1 GiB、1.7Bがロード後約4.7 GiBです。環境と入力によって変動するため余裕を確保してください。

## 6. SpeechSummarizerからの起動管理

### 6.1 推奨起動フロー

1. `QwenASR-Server.exe`、`config.json`、選択モデルが存在することを確認する
2. サーバープロセスを1つだけ起動する
3. 標準出力と標準エラーを非同期で読み取り、SpeechSummarizerの診断ログへ保存する
4. `GET /ready` を一定間隔で呼び出す
5. HTTP 200かつ `status: "ready"` を確認してから音声認識を開始する
6. 起動タイムアウトまたはプロセス終了時は、ログと終了コードを利用者へ提示する

モデルロード中は、ログが `Waiting for application startup.` のまましばらく待機します。次のログが出て `/ready` がHTTP 200を返せば起動完了です。

```text
モデルのロード完了 model=1.7b device=cuda:0
Application startup complete.
```

起動直後はHTTP接続自体が失敗することがあります。これはモデルロード完了までの正常な状態なので、短い間隔で再試行してください。固定秒数の待機だけで起動完了と判断せず、必ず `/ready` を確認します。

### 6.2 プロセス起動時の注意

- `UseShellExecute=false` で直接EXEを起動する
- EXEと設定ファイルは絶対パスで管理する
- `--config` を指定しない場合でも、既定設定はEXE基準なのでWorkingDirectoryには依存しない
- 標準出力・標準エラーをリダイレクトする場合は両方を非同期に読み取り、バッファ詰まりを防ぐ
- 同じポートにサーバーを重複起動しない
- SpeechSummarizerが起動したプロセスIDを保持し、無関係な同名プロセスを終了しない

起動引数の例:

```text
--config "D:\SpeechSummarizer\config\qwen-asr.json" --model 1.7b --host 127.0.0.1 --port 8010
```

### 6.3 稼働確認

`GET /health` はプロセスの基本情報を返します。通常の利用可否判定には、モデル情報も返す `GET /ready` を使用します。

```http
GET http://127.0.0.1:8010/ready
```

成功例:

```json
{
  "schema_version": 1,
  "app_version": "20260821",
  "status": "ready",
  "engine": "qwen3-asr",
  "backend": "transformers",
  "model": "1.7b",
  "model_id": "1.7b",
  "device": "cuda:0",
  "queue_depth": 0,
  "queue_capacity": 20
}
```

SpeechSummarizerは、期待する `model`、`device`、`schema_version` も確認できます。`app_version` はビルドのバージョンであり、API互換性を表す `schema_version` とは別です。

## 7. 音声認識API

```http
POST http://127.0.0.1:8010/transcribe
Content-Type: multipart/form-data
```

| フィールド | 必須 | 内容 |
|---|---:|---|
| `audio` | はい | WAVファイル。ファイル名の拡張子も `.wav` にする |
| `request_id` | いいえ | 追跡用ID。患者情報を含めない。省略時はUUIDを生成 |
| `language` | いいえ | 例: `Japanese`。省略時は `config.json` の `language` |
| `context` | いいえ | 認識ヒント。医療用語など。文字数上限あり |

成功応答では、認識結果が `text` に入ります。

```json
{
  "schema_version": 1,
  "request_id": "segment-00042",
  "text": "今日は右の耳が痛いです。",
  "language": "Japanese",
  "engine": "qwen3-asr",
  "backend": "transformers",
  "model": "1.7b",
  "model_id": "1.7b",
  "audio": { "duration_sec": 4.25 },
  "timing": {
    "queue_sec": 0.01,
    "inference_sec": 0.72,
    "total_sec": 0.74,
    "rtf": 0.1694
  },
  "provider_metrics": {}
}
```

APIはアップロードされた音声と認識結果を恒久保存しません。`context` と完全一致する文字列が認識結果にエコーされた場合は応答前に除去され、発話が残らなければ `text` は空文字になります。

主なエラー:

| HTTP | `error.code` | SpeechSummarizer側の扱い |
|---:|---|---|
| 400 | `invalid_request` | 入力値を修正。自動再試行しない |
| 413 | `audio_too_large` | 音声を短く分割。自動再送しない |
| 415 | `unsupported_audio` | WAVへ変換。自動再送しない |
| 422 | `invalid_audio` | WAVの破損や長さを確認。自動再送しない |
| 429 | `queue_full` | バックオフして再試行可能 |
| 503 | `not_ready` | `/ready` を待って再試行可能 |
| 503 | `gpu_out_of_memory` | モデル変更またはGPU状態の確認が必要 |
| 504 | `request_timeout` | 状況を確認して再試行可能 |
| 500 | `inference_failed` | サーバーログを確認 |

タイムアウト後も、開始済みのGPU推論は安全のため強制停止されません。結果は破棄されますが、推論終了までは次のリクエストを処理できないことがあります。

## 8. 設定変更・モデル変更・再起動

`config.json` はプロセス起動時に一度だけ読み込まれます。実行中にファイルを書き換えても反映されず、ホットリロード用のAPIもありません。モデル、ポート、タイムアウトを含め、すべての設定変更にはサーバー再起動が必要です。

SpeechSummarizerからの推奨再起動手順:

1. Qwen ASRへの新規リクエスト送信を停止する
2. 実行中・待機中のリクエストが終わるのを待つ
3. サーバープロセスを終了する
4. 新しいJSONを一時ファイルへ書き、同一ボリューム上で `config.json` と置換する
5. 同じEXEを起動する
6. `/ready` がHTTP 200になるまで待つ
7. `/ready` の `model` が選択値と一致することを確認する
8. 新規リクエストの受付を再開する

設定ファイルを直接書きかけの状態にすると、再起動時にJSON解析エラーになる可能性があります。一時ファイルへ完全なJSONを書いてから置換してください。新設定で起動できなかった場合に戻せるよう、直前の正常な設定を保持することを推奨します。

現在のAPIには `/shutdown` endpointがありません。コンソールからの `Ctrl+C` では、キューworker停止後にモデルを解放して終了します。SpeechSummarizerから非表示の子プロセスとして起動した場合は、実行中・待機中の要求がなくなったことを確認してから、保持しているプロセスIDのプロセスツリーを終了し、終了完了を待ちます。プロセス名による一括終了は、別に起動したサーバーまで停止する危険があるため使用しないでください。

強制終了時もWindowsがプロセスのGPUメモリを回収しますが、処理中の要求や一時ファイルを正常終了できません。終了猶予を設け、応答しない場合だけ強制終了する運用にしてください。将来、常に正常終了が必要になった場合は、SpeechSummarizerだけが呼べる安全なshutdown方式を別途実装する必要があります。

サーバー停止後は、プロセス終了とポート解放を確認してから再起動します。GPUメモリやポートが解放されるまで短時間かかる場合は、上限時間を設けて再試行してください。

## 9. 終了コードと障害判定

| 状態 | 終了コードの目安 |
|---|---:|
| `--version`、`--help`、通常停止 | 0 |
| 直接のキーボード割り込み | 130になる場合がある |
| 引数、設定、起動の失敗 | 0以外（通常2） |

SpeechSummarizerは終了コードだけでなく、標準エラー、プロセスの予期しない終了、`/ready` の結果を組み合わせて障害を判定してください。

代表的な起動失敗原因:

- `offline: true` なのに選択モデルがローカルに揃っていない
- `api.model_alias` と `local_model_paths` / `models` のキーが一致していない
- CUDA対応GPUまたはNVIDIAドライバーを利用できない
- 1.7BモデルのロードでGPUメモリが不足した
- 指定ポートが別プロセスに使用されている
- `config.json` のJSON形式、型、数値範囲が不正

障害調査では、まずサーバーの標準出力・標準エラーと、使用した `config.json`、終了コードを確認してください。

## 10. GitHub Release用の分割インストーラー

`packaging\build_windows_installer.ps1` は、`dist\QwenASR-Server` だけを対象にInno Setupの分割インストーラーを作成します。`dist\QwenASR` は含まれません。

前提:

- `packaging\build_windows.ps1 -Target server` によるサーバー版ビルドが完了している
- Inno Setup 6がインストールされている
- 配布するモデルが `dist\QwenASR-Server\models` に配置されている

商用利用の場合は、Inno Setup 6.5以降の[商用ライセンス条件](https://jrsoftware.org/ishelp/topic_purchase.htm)を確認し、必要なライセンスを用意してください。モデル一式を圧縮するため、ビルドには環境によって数十分かかります。

作成コマンド:

```powershell
.\packaging\build_windows_installer.ps1
```

Inno Setupを標準以外の場所へインストールした場合:

```powershell
.\packaging\build_windows_installer.ps1 `
  -ISCCPath "D:\Tools\Inno Setup 6\ISCC.exe"
```

`release` フォルダーへ次の形式で出力されます。バージョンは `version.py` の `APP_VERSION` から取得します。

```text
QwenASR-Server-20260822-win-x64.exe
QwenASR-Server-20260822-win-x64-1.bin
QwenASR-Server-20260822-win-x64-2.bin
...
QwenASR-Server-20260822-win-x64-SHA256.txt
```

各 `.bin` は1,900,000,000バイト以下で、GitHub Releaseの1ファイル2GiB未満という制限に収まります。ビルドスクリプトも完成後にサイズを検証し、すべての分割ファイルに対するSHA-256一覧を生成します。

Releaseには同じバージョンのEXE、すべての `.bin`、SHA-256ファイルをアップロードします。利用者は全ファイルを同じフォルダーへ保存し、EXEを実行します。EXEが後続の `.bin` を読み込み、指定フォルダーへ `QwenASR-Server` の内容を展開します。利用者側にInno Setupや7-Zipは不要です。

配布物の `config.json` は、ローカルで変更された `dist` 内のファイルではなく、リポジトリの `config.json.sample` から生成されます。モデルキャッシュを含める場合は、ライセンスと配布条件を事前に確認してください。
