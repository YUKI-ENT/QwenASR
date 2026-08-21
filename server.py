#!/usr/bin/env python3
"""Local-only FastAPI server for one resident Qwen3-ASR model."""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import tempfile
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from api_service import (
    APISettings,
    ASRService,
    NotReadyError,
    QueueFullError,
    RequestTimeoutError,
    TranscriptionJob,
)
from audio_utils import AudioError, get_audio_duration
from cli_common import create_engine, load_config, resolve_model
from qwen_asr_engine import ASRError


LOGGER = logging.getLogger("qwen_asr.api")
SCHEMA_VERSION = 1


def _error(
    status: int, request_id: str | None, code: str, message: str, retryable: bool
) -> JSONResponse:
    return JSONResponse(
        status_code=status,
        content={
            "schema_version": SCHEMA_VERSION,
            "request_id": request_id,
            "error": {"code": code, "message": message, "retryable": retryable},
        },
    )


def _contains_control(value: str) -> bool:
    return any(ord(character) < 32 or ord(character) == 127 for character in value)


def _is_oom(exc: BaseException) -> bool:
    current: BaseException | None = exc
    while current is not None:
        if "out of memory" in str(current).lower() or "gpuメモリ不足" in str(current).lower():
            return True
        current = current.__cause__
    return False


def _remove_context_echo(transcript: str, context: str | None) -> str:
    """Remove exact context echoes without guessing whether other text is speech."""
    text = str(transcript).strip()
    prompt = context.strip() if context is not None else ""
    if not text or not prompt or prompt not in text:
        return text

    # Replace every exact occurrence with a boundary so surrounding utterances do
    # not become accidentally concatenated. Preserve meaningful line separation.
    filtered = text.replace(prompt, "\n")
    return "\n".join(line.strip() for line in filtered.splitlines() if line.strip())


def _safe_model_names(config: dict[str, Any], settings: APISettings, resolved: str) -> tuple[str, str]:
    alias = settings.model_alias
    if Path(resolved).is_absolute():
        safe_name = alias or Path(resolved).name or "local-model"
        return safe_name, safe_name
    return alias or resolved, resolved


def _validate_engine_config(config: dict[str, Any]) -> None:
    for name in ("device", "dtype", "model", "model_cache_dir"):
        if not isinstance(config.get(name), str) or not str(config[name]).strip():
            raise ValueError(f"{name}は空でない文字列で指定してください。")
    if str(config["dtype"]).lower() not in {
        "bfloat16", "bf16", "float16", "fp16", "float32", "fp32"
    }:
        raise ValueError("dtypeはbfloat16/float16/float32のいずれかを指定してください。")
    language = config.get("language")
    if language is not None and not isinstance(language, str):
        raise ValueError("languageは文字列またはnullで指定してください。")
    for name in ("max_new_tokens", "max_inference_batch_size"):
        value = config.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name}は1以上の整数で指定してください。")
    poll = config.get("nvidia_smi_poll_interval_sec")
    if isinstance(poll, bool) or not isinstance(poll, (int, float)) or float(poll) <= 0:
        raise ValueError("nvidia_smi_poll_interval_secは0より大きい数値で指定してください。")
    if not isinstance(config.get("offline"), bool):
        raise ValueError("offlineはbooleanで指定してください。")
    for name in ("models", "local_model_paths"):
        if not isinstance(config.get(name), dict):
            raise ValueError(f"{name}はJSON objectで指定してください。")


def create_app(
    config: dict[str, Any],
    base_dir: Path,
    *,
    engine: Any | None = None,
    settings: APISettings | None = None,
) -> FastAPI:
    _validate_engine_config(config)
    settings = settings or APISettings.from_config(config)
    resolved_model = resolve_model(config, settings.model_alias, base_dir)
    engine = engine or create_engine(config, resolved_model, base_dir)
    model_name, model_id = _safe_model_names(config, settings, resolved_model)
    service = ASRService(engine, settings)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        try:
            await service.start()
            LOGGER.info("モデルのロード完了 model=%s device=%s", model_name, engine.device)
        except Exception:
            LOGGER.exception("モデルのロードに失敗しました model=%s", model_name)
            raise
        try:
            yield
        finally:
            await service.stop()
            LOGGER.info("モデルを解放しました model=%s", model_name)

    app = FastAPI(title="Qwen3-ASR Local API", version="1", lifespan=lifespan)
    app.state.service = service
    app.state.api_settings = settings

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(request: Request, _: RequestValidationError) -> JSONResponse:
        return _error(400, None, "invalid_request", "リクエスト形式が不正です。", False)

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "status": "ok",
            "engine": "qwen3-asr",
            "backend": "transformers",
        }

    @app.get("/ready", response_model=None)
    async def ready() -> dict[str, Any] | JSONResponse:
        if not service.ready:
            return _error(503, None, "not_ready", "モデルの準備ができていません。", True)
        return {
            "schema_version": SCHEMA_VERSION,
            "status": "ready",
            "engine": "qwen3-asr",
            "backend": "transformers",
            "model": model_name,
            "model_id": model_id,
            "device": str(engine.device),
            "queue_depth": service.queue_depth,
            "queue_capacity": settings.max_queue_size,
        }

    @app.post("/transcribe", response_model=None)
    async def transcribe(
        audio: UploadFile = File(...),
        request_id: str | None = Form(None),
        language: str | None = Form(None),
        context: str | None = Form(None),
    ) -> dict[str, Any] | JSONResponse:
        actual_request_id = request_id if request_id is not None else str(uuid.uuid4())
        if (
            not actual_request_id
            or actual_request_id != actual_request_id.strip()
            or len(actual_request_id) > settings.max_request_id_chars
            or _contains_control(actual_request_id)
        ):
            return _error(400, actual_request_id or None, "invalid_request", "request_idが不正です。", False)
        if language is not None and (
            not language.strip()
            or len(language) > settings.max_language_chars
            or _contains_control(language)
        ):
            return _error(400, actual_request_id, "invalid_request", "languageが不正です。", False)
        if context is not None and (
            len(context) > settings.max_context_chars or "\x00" in context
        ):
            return _error(400, actual_request_id, "invalid_request", "contextが不正です。", False)

        suffix = Path(audio.filename or "").suffix.lower()
        if suffix and suffix != ".wav":
            return _error(415, actual_request_id, "unsupported_audio", "WAV形式の音声だけを利用できます。", False)
        if audio.content_type and audio.content_type.lower() not in {
            "audio/wav", "audio/x-wav", "audio/wave", "audio/vnd.wave", "application/octet-stream"
        }:
            return _error(415, actual_request_id, "unsupported_audio", "WAV形式の音声だけを利用できます。", False)

        temp_dir = tempfile.TemporaryDirectory(prefix="qwen_asr_api_")
        handed_to_service = False
        temp_path = Path(temp_dir.name) / "upload.wav"
        size = 0
        header = b""
        try:
            with temp_path.open("wb") as destination:
                while chunk := await audio.read(64 * 1024):
                    size += len(chunk)
                    if size > settings.max_upload_bytes:
                        temp_dir.cleanup()
                        return _error(413, actual_request_id, "audio_too_large", "音声ファイルがupload上限を超えています。", False)
                    if len(header) < 12:
                        header += chunk[: 12 - len(header)]
                    destination.write(chunk)
            if len(header) < 12 or header[:4] not in {b"RIFF", b"RF64"} or header[8:12] != b"WAVE":
                temp_dir.cleanup()
                return _error(415, actual_request_id, "unsupported_audio", "WAV形式の音声だけを利用できます。", False)
            try:
                duration = await asyncio.to_thread(get_audio_duration, temp_path)
            except Exception:
                temp_dir.cleanup()
                return _error(422, actual_request_id, "invalid_audio", "WAV音声を読み込めません。", False)
            if duration <= 0 or duration > settings.max_audio_sec:
                temp_dir.cleanup()
                message = "音声長が不正です。" if duration <= 0 else "音声長が上限を超えています。"
                return _error(422, actual_request_id, "invalid_audio", message, False)

            job = TranscriptionJob(
                audio_path=temp_path,
                request_id=actual_request_id,
                language=language.strip() if language is not None else config.get("language"),
                context=context,
                duration_sec=duration,
                cleanup=temp_dir.cleanup,
            )
            handed_to_service = True
            try:
                completed = await service.submit(job)
            except NotReadyError:
                return _error(503, actual_request_id, "not_ready", "モデルの準備ができていません。", True)
            except QueueFullError:
                return _error(429, actual_request_id, "queue_full", "推論キューが混雑しています。", True)
            except RequestTimeoutError:
                return _error(504, actual_request_id, "request_timeout", "音声認識がタイムアウトしました。", True)
            except Exception as exc:
                if _is_oom(exc):
                    LOGGER.exception("音声認識でGPU OOM request_id=%s", actual_request_id)
                    return _error(503, actual_request_id, "gpu_out_of_memory", "GPUメモリが不足しています。", True)
                LOGGER.exception("音声認識に失敗 request_id=%s", actual_request_id)
                return _error(500, actual_request_id, "inference_failed", "音声認識に失敗しました。", False)

            result = completed.result
            response_text = _remove_context_echo(result.transcript, job.context)
            response_language = result.detected_language or job.language
            inference_sec = completed.inference_sec
            LOGGER.info(
                "音声認識完了 request_id=%s duration_sec=%.3f model=%s queue_sec=%.3f inference_sec=%.3f status=200",
                actual_request_id, duration, model_name, completed.queue_sec, inference_sec,
            )
            return {
                "schema_version": SCHEMA_VERSION,
                "request_id": actual_request_id,
                "text": response_text,
                "language": response_language,
                "engine": "qwen3-asr",
                "backend": "transformers",
                "model": model_name,
                "model_id": model_id,
                "audio": {"duration_sec": round(duration, 4)},
                "timing": {
                    "queue_sec": round(completed.queue_sec, 4),
                    "inference_sec": round(inference_sec, 4),
                    "total_sec": round(completed.total_sec, 4),
                    "rtf": round(inference_sec / duration, 6) if duration > 0 else None,
                },
                "provider_metrics": {},
            }
        except (OSError, AudioError):
            temp_dir.cleanup()
            return _error(422, actual_request_id, "invalid_audio", "WAV音声を読み込めません。", False)
        finally:
            await audio.close()
            if not handed_to_service:
                temp_dir.cleanup()

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Qwen3-ASR localhost HTTP API")
    parser.add_argument("--config", default="config.json", help="設定JSON")
    parser.add_argument("--host", help="localhost listen address")
    parser.add_argument("--port", type=int, help="listen port")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        config, base_dir = load_config(args.config)
        api_config = dict(config.get("api", {}) or {})
        if args.host is not None:
            api_config["host"] = args.host
        if args.port is not None:
            api_config["port"] = args.port
        config["api"] = api_config
        settings = APISettings.from_config(config)
        app = create_app(config, base_dir, settings=settings)
    except (ValueError, ASRError, AudioError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    uvicorn_config = uvicorn.Config(
        app, host=settings.host, port=settings.port, workers=1
    )
    uvicorn_server = uvicorn.Server(uvicorn_config)
    try:
        uvicorn_server.run()
    except KeyboardInterrupt:
        return 130
    return 0 if uvicorn_server.started else 2


if __name__ == "__main__":
    raise SystemExit(main())
