# Qwen3-ASR HTTP API 実装仕様書

## 1. この文書の目的

このリポジトリのQwen3-ASR推論機能を、別プロセスで稼働するローカルHTTP APIへ拡張する。
呼び出し元は別WorkspaceのSpeechSummarizerである。

この作業では、既存のCLIベンチマークを壊さず、モデル・Python依存関係・GPU障害を
SpeechSummarizer本体から分離することを最優先とする。

実装担当のCodexは、本仕様の必須要件を満たすところまで実装・テスト・README更新を行う。
不明点が軽微で安全な場合は合理的な既定値で進め、重要なAPI互換性を変更する場合だけ確認する。

## 2. Workspaceと変更範囲

- 作業対象: `/home/yuki/QwenASR`
- 呼び出し元の検証用worktree: `/home/yuki/SpeechSummarizer-qwen`
- 稼働中の本番: `/home/yuki/SpeechSummarizer`
- `/home/yuki/SpeechSummarizer` は変更しないこと。
- この段階では `/home/yuki/SpeechSummarizer-qwen` も変更しないこと。
- QwenASRはSpeechSummarizerと仮想環境やPythonパッケージを共有しない。
- モデルファイル、患者音声、認識結果をGit管理対象へ追加しない。

`/home/yuki/QwenASR` は現時点ではGitリポジトリではない。Git初期化やremote設定は、ユーザーから
明示的に依頼されない限り実行しないこと。

## 3. 現在の実装

既存の主なファイル:

- `app.py`: 単一音声ファイル用CLI
- `benchmark.py`: 複数音声・モデル比較CLI
- `qwen_asr_engine.py`: `Qwen3ASRModel` のラッパー
- `audio_utils.py`: 音声検証、長さ取得、必要時の形式変換
- `cli_common.py`: 設定、モデル解決、エンジン生成
- `config.json.sample`: 設定例
- `tests/test_core.py`: モデルを実ロードしない既存テスト

既存CLIの動作と出力形式は維持すること。API追加のためにCLIをHTTPクライアントへ変更しない。

## 4. 全体構成と責務

想定構成:

```text
SpeechSummarizer
  - ブラウザ音声受信
  - RMSベースVAD
  - WAV保存
  - 品質判定
  - JSONL、UI、LLM、Whisper fallback
            |
            | HTTP multipart/form-data (localhost)
            v
QwenASR API
  - モデルを起動時に1回ロードして常駐
  - リクエストをキューイングして直列推論
  - 音声認識結果と客観的な実行メタデータを返却
```

QwenASR APIが担当しないもの:

- VAD、発話区間分割
- `good/maybe/bad` の品質判定
- Whisper固有の `avg_logprob`、`no_speech_prob`、`compression_ratio` の代替値生成
- SpeechSummarizerのJSONLや患者セッションへの書き込み
- Whisperへのフォールバック
- 音声・認識文の恒久保存
- Web UI
- LLM補正、要約

## 5. v1の非目標

以下は今回実装しない。

- vLLM backend
- ストリーミングASR
- WebSocket API
- Forced Aligner、word timestamp、SRT
- 複数モデルの同時常駐
- リクエスト単位のモデル切替
- Docker、systemdサービス登録
- 外部公開、TLS、ユーザー認証
- バッチ推論API
- GPUを使う本物のモデルをCI/unit testでロードすること

将来追加できる構造にはしてよいが、v1を複雑にしないこと。

## 6. サーバー起動とモデルライフサイクル

### 6.1 必須動作

- FastAPI + Uvicornで実装する。
- 既定listen先は `127.0.0.1:8010` とし、外部インターフェースへ公開しない。
- モデルはサーバー起動時に1回だけロードする。
- 各リクエスト後に `unload()` しない。
- サーバー終了時に `engine.unload()` を呼ぶ。
- モデルロードに失敗した場合は、理由をログへ出してプロセスを正常稼働扱いにしない。
- Uvicorn workerは必ず1つとする。複数workerでモデルが重複ロードされる構成を禁止する。
- 推論は同時に1件だけ実行する。ブロッキング推論でasyncio event loopを停止させない。
- 推奨実装は `asyncio.Queue` + 単一worker + `asyncio.to_thread()` である。
- キューには上限を設ける。
- 待機を含むリクエストタイムアウトを設ける。

### 6.2 起動コマンド

最低限、次のどちらかに相当する起動方法をREADMEへ記載する。

```bash
source .venv/bin/activate
python server.py --config config.json
```

必要なら `QwenASR_API.sh` のような薄い起動スクリプトを追加してよい。スクリプトは必ず自身の
配置ディレクトリを基準にし、呼び出し時のカレントディレクトリへ依存しないこと。

## 7. 設定

既存設定との互換性を維持し、`api` セクションを追加する。

`config.json.sample` の想定例:

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
}
```

要件:

- API起動時は `api.model_alias` を `resolve_model()` に渡してモデルを固定する。
- `api.model_alias` が空または未指定なら、既存のトップレベル `model` を使う。
- 相対モデルパスとキャッシュパスは、これまでどおり設定ファイル位置を基準にする。
- `offline` の既存動作を維持する。
- APIサーバーではトップレベル `unload_after` を無視してモデルを常駐させる。
- CLIでは従来どおり `unload_after` を利用する。
- CLI引数でhost、portを上書きできるようにしてよい。
- 設定値の型・範囲を起動時に検証し、不正なら分かりやすいエラーで終了する。

## 8. HTTP API契約

APIのJSONには `schema_version: 1` を含める。将来、互換性を壊す変更はschema versionを上げる。

### 8.1 `GET /health`

用途: livenessと基本情報の確認。

成功: HTTP 200

```json
{
  "schema_version": 1,
  "status": "ok",
  "engine": "qwen3-asr",
  "backend": "transformers"
}
```

このendpointではGPU推論を実行しない。

### 8.2 `GET /ready`

用途: モデルがロード済みで、リクエストを受け付けられることの確認。

成功: HTTP 200

```json
{
  "schema_version": 1,
  "status": "ready",
  "engine": "qwen3-asr",
  "backend": "transformers",
  "model": "1.7b",
  "model_id": "Qwen/Qwen3-ASR-1.7B",
  "device": "cuda:0",
  "queue_depth": 0,
  "queue_capacity": 20
}
```

未準備の場合はHTTP 503と共通エラー形式を返す。ローカル絶対パスは `model_id` として公開しない。
ローカルモデルを使う場合の `model_id` は設定上のaliasまたは安全な表示名とする。

### 8.3 `POST /transcribe`

Content-Type: `multipart/form-data`

フォームフィールド:

| 名前 | 型 | 必須 | 内容 |
|---|---|---:|---|
| `audio` | file | yes | WAV音声。SpeechSummarizerからはmono PCM WAVが渡る |
| `request_id` | string | no | 呼び出し元の追跡ID。省略時はサーバーでUUID生成 |
| `language` | string | no | 例: `Japanese`。省略時は設定値 |
| `context` | string | no | 医療用語などの認識コンテキスト。空文字可 |

v1ではWAVだけを正式サポートする。SpeechSummarizerの現行WAVは48 kHzになり得るため、16 kHz固定を
要求しない。Qwen公式ラッパーに渡す前に既存の音声検証を行い、必要な正規化は公式ラッパーまたは
既存ユーティリティへ任せる。入力音声を書き換えない。

成功: HTTP 200

```json
{
  "schema_version": 1,
  "request_id": "ws1-seg-42",
  "text": "今日は右の耳が痛いです。",
  "language": "Japanese",
  "engine": "qwen3-asr",
  "backend": "transformers",
  "model": "1.7b",
  "model_id": "Qwen/Qwen3-ASR-1.7B",
  "audio": {
    "duration_sec": 4.25
  },
  "timing": {
    "queue_sec": 0.01,
    "inference_sec": 0.72,
    "total_sec": 0.74,
    "rtf": 0.1694
  },
  "provider_metrics": {}
}
```

契約上の注意:

- 認識文のキーはCLIの `transcript` ではなくAPIでは `text` とする。
- `text` は常にstring。無音等で認識文がない場合は空文字を正常結果として返してよい。
- `language` は不明なら `null`。
- 秒数とRTFは数値。取得不能な任意値は `null` とする。
- Qwenが提供しないconfidenceやWhisper風の疑似確率を生成しない。
- `provider_metrics` はv1では空objectでよい。将来の後方互換拡張用とする。
- リクエストでモデル名、device、dtypeを指定させない。
- `context` を `Qwen3ASRModel.transcribe(..., context=...)` へ渡せるよう、
  `QwenASREngine.transcribe()` を後方互換な任意引数で拡張する。
- languageも可能なら呼び出し単位で上書き可能にする。ただしengine自身の既定値を書き換えて
  並行リクエストへ漏らさない。

## 9. エラー契約

全APIエラーは可能な限り次の形式へ統一する。

```json
{
  "schema_version": 1,
  "request_id": "ws1-seg-42",
  "error": {
    "code": "invalid_audio",
    "message": "WAV音声を読み込めません。",
    "retryable": false
  }
}
```

最低限の分類:

| HTTP | code | 例 | retryable |
|---:|---|---|---:|
| 400 | `invalid_request` | 不正なrequest_id、language、context | false |
| 413 | `audio_too_large` | upload上限超過 | false |
| 415 | `unsupported_audio` | WAV以外 | false |
| 422 | `invalid_audio` | 壊れたWAV、0秒、長さ上限超過 | false |
| 429 | `queue_full` | 推論キュー満杯 | true |
| 503 | `not_ready` | モデル未準備 | true |
| 503 | `gpu_out_of_memory` | GPU OOM | true |
| 504 | `request_timeout` | キュー待ちを含むタイムアウト | true |
| 500 | `inference_failed` | その他推論失敗 | 状況に応じる |

要件:

- クライアントへPython traceback、ローカル一時パス、モデルキャッシュパスを返さない。
- サーバーログには診断可能な例外情報を残してよい。
- upload上限は可能な限り全データをメモリへ読み切る前に検出する。
- 音声長上限を超えた入力は自動分割せず拒否する。分割はSpeechSummarizerの責務である。
- タイムアウト後も既に開始したGPU処理を安全に強制終了できない場合は、処理を完了させて結果だけ
  破棄してよい。その場合もworkerが壊れず次のジョブを処理できること。

## 10. 一時ファイルとプライバシー

- uploadは安全な一時ディレクトリへ保存し、成功・失敗・キャンセルを問わず必ず削除する。
- クライアント提供ファイル名を保存パスとしてそのまま使わない。
- ディレクトリトラバーサルを許さない。
- APIは音声や認識文を `results/` へ自動保存しない。
- 通常ログへ認識全文を出さない。
- 通常ログへ患者IDや元ファイル名を出さない。
- ログへ記録してよいものはrequest_id、音声長、モデルalias、キュー時間、推論時間、HTTP結果、
  エラー分類などとする。
- `request_id` も患者情報を含まないopaqueなIDを想定する。
- 外部ネットワークアクセスを避けるため、本番相当試験は `offline: true` で行う。

## 11. キューとキャンセル

- FIFOキューとする。
- 1ジョブにつきFuture等でHTTPリクエストと結果を対応づける。
- queue depthは待機中の件数とし、実行中ジョブを含めるかどうかをコードとテストで一貫させる。
- HTTPクライアント切断・キャンセル後に、まだ推論開始前ならジョブを安全に無効化する。
- 推論開始後ならモデル状態を壊す強制中断はしない。
- 1件の失敗やキャンセルでworker taskが終了しないよう、ジョブ単位で例外を処理する。
- shutdown時は新規受付を停止し、workerを停止してモデルを解放する。

## 12. 依存パッケージ

`requirements.txt` にAPI用依存を追加する。

- `fastapi`
- `uvicorn`
- `python-multipart`

既存のtorch、torchaudio、qwen-asr、transformers等の固定を維持する。依存追加後もREADME記載の
「PyTorchをCUDA 12.8 indexから先に入れる」手順を維持する。

## 13. テスト要件

### 13.1 自動テスト

実モデルやGPUを必要としない `tests/test_api.py` を追加する。FastAPIのテストクライアントまたは
httpxを使用し、エンジンをfake/mockへ置き換える。

最低限テストする項目:

1. `/health` のschemaとHTTP 200
2. `/ready` の準備済み応答
3. 正常なWAVを `/transcribe` へ送り、API契約どおり返る
4. request_id省略時にIDが生成される
5. languageとcontextがengineへ渡る
6. 空の認識文を正常に返せる
7. WAV以外を拒否する
8. 壊れたWAVを拒否する
9. 長さ上限を拒否する
10. uploadサイズ上限を拒否する
11. queue fullを429で返す
12. timeoutを504で返し、その後もworkerが利用可能
13. engine例外を共通エラー形式へ変換する
14. OOMを503へ分類する
15. 一時ファイルが成功時・失敗時とも残らない
16. 複数リクエストでengine/modelのロードが1回だけ
17. 推論が並列実行されない
18. contextやrequest_idの長さ制限
19. 既存 `tests/test_core.py` が引き続き成功する

可能な限り時間依存で不安定なsleepを使わず、Eventや制御可能なfakeで並行性を検証する。

実行例:

```bash
python -m unittest discover -s tests -v
python -m compileall app.py benchmark.py server.py audio_utils.py gpu_monitor.py qwen_asr_engine.py cli_common.py
```

### 13.2 手動実機テスト

個人情報を含まないテストWAVで確認する。

```bash
curl -s http://127.0.0.1:8010/health | python -m json.tool
curl -s http://127.0.0.1:8010/ready | python -m json.tool
curl -s -X POST http://127.0.0.1:8010/transcribe \
  -F 'audio=@test_audio/sample.wav;type=audio/wav' \
  -F 'request_id=manual-test-001' \
  -F 'language=Japanese' \
  -F 'context=耳鼻咽喉科の診察会話' | python -m json.tool
```

確認事項:

- 起動時にだけモデルロードが行われる
- 連続2回の認識でモデルを再ロードしない
- リクエスト後もVRAMがモデル常駐相当で安定する
- 0.6Bまたは1.7Bの設定モデルで認識できる
- API停止後にGPUリソースが解放される
- `offline: true` で外部アクセスなしに起動・認識できる
- 既存 `python app.py ...` と `python benchmark.py ...` が動作する

## 14. README更新要件

READMEへ以下を追記する。

- APIの目的とSpeechSummarizerから独立していること
- API依存のインストール方法
- 設定例
- 起動・停止方法
- health/ready/transcribeのcurl例
- モデル常駐に必要なVRAMの実測目安
- localhost専用であり、認証なしで外部公開してはいけないこと
- トラブルシューティング
  - モデル未配置
  - offline時のキャッシュ不足
  - CUDA未検出
  - OOM
  - queue full
  - timeout

既存READMEにあるCLIベンチマークの説明は削除しない。

## 15. 実装上の推奨ファイル構成

厳密な指定ではないが、責務を分ける場合は次を推奨する。

```text
server.py              # CLI引数、FastAPI lifespan、Uvicorn起動
api_service.py         # queue worker、リクエスト処理、engine adapter
api_models.py          # response/error model（必要なら）
tests/test_api.py
```

過剰なフレームワーク化は不要。循環importを避け、API部分を実モデルなしでテストできる設計にする。

## 16. 完了条件

次のすべてを満たしたらQwenASR側のv1完了とする。

- `GET /health`、`GET /ready`、`POST /transcribe` が本仕様どおり動作する
- モデルは1回ロードされ、推論は直列化される
- Qwenへlanguageとcontextを渡せる
- API応答にWhisper互換の架空confidenceを含めない
- upload、一時ファイル、エラー、timeout、queue fullを安全に扱う
- 通常ログへ音声・認識全文・患者情報を残さない
- unit testが実モデルなしで成功する
- 既存CLIテストとCLI利用方法が維持される
- 実機で少なくとも1モデルの連続2回認識に成功する
- READMEが更新される
- SpeechSummarizer側を変更していない

## 17. 実装完了時の報告内容

実装担当のCodexは、完了時に次を簡潔に報告する。

- 追加・変更したファイル
- APIの起動コマンド
- 自動テスト結果
- 実モデル試験を行ったか、その結果
- 未解決事項または意図的に先送りした項目
- SpeechSummarizer側が利用すべき最終API契約との差異（差異がある場合のみ）

