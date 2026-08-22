"""Queueing and lifecycle management for the QwenASR HTTP API."""

from __future__ import annotations

import asyncio
import ipaddress
import logging
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


LOGGER = logging.getLogger("qwen_asr.api")


class ServiceError(RuntimeError):
    """Base class for errors translated to the public API contract."""


class NotReadyError(ServiceError):
    pass


class QueueFullError(ServiceError):
    pass


class RequestTimeoutError(ServiceError):
    pass


@dataclass(frozen=True)
class APISettings:
    host: str = "127.0.0.1"
    port: int = 8010
    model_alias: str | None = None
    max_queue_size: int = 20
    request_timeout_sec: float = 30.0
    max_audio_sec: float = 30.0
    max_upload_mib: float = 10.0
    max_context_chars: int = 2000
    max_request_id_chars: int = 128
    max_language_chars: int = 64

    @property
    def max_upload_bytes(self) -> int:
        return int(self.max_upload_mib * 1024 * 1024)

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "APISettings":
        raw = config.get("api", {})
        if raw is None:
            raw = {}
        if not isinstance(raw, dict):
            raise ValueError("api設定はJSON objectである必要があります。")

        def integer(name: str, default: int, minimum: int, maximum: int) -> int:
            value = raw.get(name, default)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"api.{name}は整数で指定してください。")
            if not minimum <= value <= maximum:
                raise ValueError(
                    f"api.{name}は{minimum}から{maximum}の範囲で指定してください。"
                )
            return value

        def number(name: str, default: float, minimum: float) -> float:
            value = raw.get(name, default)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"api.{name}は数値で指定してください。")
            result = float(value)
            if not math.isfinite(result) or result <= minimum:
                raise ValueError(f"api.{name}は{minimum}より大きい数値で指定してください。")
            return result

        host = raw.get("host", "127.0.0.1")
        if not isinstance(host, str):
            raise ValueError("api.hostはIPアドレスまたはlocalhostで指定してください。")
        if host.lower() == "localhost":
            host = "localhost"
        else:
            try:
                ipaddress.ip_address(host)
            except ValueError as exc:
                raise ValueError(
                    "api.hostはIPアドレスまたはlocalhostで指定してください。"
                ) from exc
        alias_value = raw.get("model_alias")
        if alias_value is not None and not isinstance(alias_value, str):
            raise ValueError("api.model_aliasは文字列で指定してください。")
        alias = alias_value.strip() if isinstance(alias_value, str) else None
        if alias is not None and (len(alias) > 128 or any(ord(c) < 32 for c in alias)):
            raise ValueError("api.model_aliasが不正です。")

        return cls(
            host=host,
            port=integer("port", 8010, 1, 65535),
            model_alias=alias or None,
            max_queue_size=integer("max_queue_size", 20, 1, 10000),
            request_timeout_sec=number("request_timeout_sec", 30.0, 0.0),
            max_audio_sec=number("max_audio_sec", 30.0, 0.0),
            max_upload_mib=number("max_upload_mib", 10.0, 0.0),
            max_context_chars=integer("max_context_chars", 2000, 1, 1_000_000),
            max_request_id_chars=integer("max_request_id_chars", 128, 1, 4096),
            max_language_chars=integer("max_language_chars", 64, 1, 1024),
        )


@dataclass
class TranscriptionJob:
    audio_path: Path
    request_id: str
    language: str | None
    context: str | None
    duration_sec: float
    cleanup: Callable[[], None]
    future: asyncio.Future[Any] | None = None
    enqueued_at: float = 0.0


@dataclass(frozen=True)
class ServiceResult:
    result: Any
    queue_sec: float
    inference_sec: float
    total_sec: float


class ASRService:
    """Own one model and serialize all calls through one FIFO worker."""

    def __init__(self, engine: Any, settings: APISettings):
        self.engine = engine
        self.settings = settings
        self.queue: asyncio.Queue[TranscriptionJob | None] = asyncio.Queue(
            maxsize=settings.max_queue_size
        )
        self.worker_task: asyncio.Task[None] | None = None
        self.ready = False
        self.accepting = False

    @property
    def queue_depth(self) -> int:
        """Number of jobs waiting; the currently running job is excluded."""
        return self.queue.qsize()

    async def start(self) -> None:
        if self.ready:
            return
        await asyncio.to_thread(self.engine.load)
        self.accepting = True
        self.ready = True
        self.worker_task = asyncio.create_task(self._worker(), name="qwen-asr-worker")

    async def stop(self) -> None:
        self.accepting = False
        self.ready = False
        if self.worker_task is not None:
            await self.queue.put(None)
            await self.worker_task
            self.worker_task = None
        await asyncio.to_thread(self.engine.unload)

    async def submit(self, job: TranscriptionJob) -> ServiceResult:
        if not self.ready or not self.accepting:
            job.cleanup()
            raise NotReadyError
        loop = asyncio.get_running_loop()
        job.future = loop.create_future()
        job.enqueued_at = time.perf_counter()
        try:
            self.queue.put_nowait(job)
        except asyncio.QueueFull as exc:
            job.cleanup()
            raise QueueFullError from exc

        try:
            return await asyncio.wait_for(
                asyncio.shield(job.future), timeout=self.settings.request_timeout_sec
            )
        except TimeoutError as exc:
            job.future.cancel()
            raise RequestTimeoutError from exc
        except asyncio.CancelledError:
            job.future.cancel()
            raise

    async def _worker(self) -> None:
        while True:
            job = await self.queue.get()
            if job is None:
                self.queue.task_done()
                return
            try:
                if job.future is None or job.future.cancelled():
                    continue
                started = time.perf_counter()
                try:
                    result = await asyncio.to_thread(
                        self.engine.transcribe,
                        job.audio_path,
                        language=job.language,
                        context=job.context,
                    )
                    finished = time.perf_counter()
                    if not job.future.cancelled():
                        job.future.set_result(
                            ServiceResult(
                                result=result,
                                queue_sec=max(0.0, started - job.enqueued_at),
                                inference_sec=max(0.0, finished - started),
                                total_sec=max(0.0, finished - job.enqueued_at),
                            )
                        )
                except BaseException as exc:
                    if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                        raise
                    if job.future is not None and not job.future.cancelled():
                        job.future.set_exception(exc)
            finally:
                try:
                    job.cleanup()
                except Exception:
                    LOGGER.exception("一時ファイルの削除に失敗しました request_id=%s", job.request_id)
                self.queue.task_done()
