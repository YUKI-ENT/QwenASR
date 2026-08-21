from __future__ import annotations

import asyncio
import io
import threading
import time
import unittest
import wave
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx

from api_service import APISettings
from server import create_app


def wav_bytes(duration_sec: float = 0.1, rate: int = 16000) -> bytes:
    output = io.BytesIO()
    with wave.open(output, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(rate)
        wav_file.writeframes(b"\x00\x00" * int(duration_sec * rate))
    return output.getvalue()


class FakeEngine:
    device = "cpu"

    def __init__(self) -> None:
        self.load_count = 0
        self.unload_count = 0
        self.calls: list[dict[str, Any]] = []
        self.paths: list[Path] = []
        self.text = "今日は右の耳が痛いです。"
        self.error: Exception | None = None
        self.delay = 0.0
        self.active = 0
        self.max_active = 0
        self.lock = threading.Lock()
        self.started = threading.Event()
        self.release: threading.Event | None = None

    def load(self) -> None:
        self.load_count += 1

    def unload(self) -> None:
        self.unload_count += 1

    def transcribe(self, audio_path: Path, **kwargs: Any) -> Any:
        with self.lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        self.started.set()
        self.paths.append(Path(audio_path))
        self.calls.append(kwargs)
        try:
            if self.release is not None:
                self.release.wait(timeout=2)
            if self.delay:
                time.sleep(self.delay)
            if self.error is not None:
                raise self.error
            return SimpleNamespace(
                transcript=self.text,
                detected_language=kwargs.get("language"),
            )
        finally:
            with self.lock:
                self.active -= 1


def base_config() -> dict[str, Any]:
    return {
        "device": "cpu",
        "dtype": "float32",
        "model": "Qwen/Qwen3-ASR-0.6B",
        "language": "Japanese",
        "max_new_tokens": 256,
        "max_inference_batch_size": 1,
        "model_cache_dir": "models",
        "offline": True,
        "nvidia_smi_poll_interval_sec": 0.05,
        "models": {"0.6b": "Qwen/Qwen3-ASR-0.6B"},
        "local_model_paths": {"0.6b": ""},
    }


class APITests(unittest.IsolatedAsyncioTestCase):
    async def use_app(
        self, engine: FakeEngine | None = None, settings: APISettings | None = None
    ) -> tuple[Any, FakeEngine, httpx.AsyncClient, Any]:
        fake = engine or FakeEngine()
        chosen = settings or APISettings(model_alias="0.6b")
        app = create_app(base_config(), Path.cwd(), engine=fake, settings=chosen)
        lifespan = app.router.lifespan_context(app)
        await lifespan.__aenter__()
        client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        )
        return app, fake, client, lifespan

    async def close_app(self, client: httpx.AsyncClient, lifespan: Any) -> None:
        await client.aclose()
        await lifespan.__aexit__(None, None, None)

    async def test_health_ready_and_model_loaded_once(self) -> None:
        app, engine, client, lifespan = await self.use_app()
        try:
            health = await client.get("/health")
            self.assertEqual(health.status_code, 200)
            self.assertEqual(
                health.json(),
                {
                    "schema_version": 1,
                    "app_version": "20260821",
                    "status": "ok",
                    "engine": "qwen3-asr",
                    "backend": "transformers",
                },
            )
            ready = await client.get("/ready")
            self.assertEqual(ready.status_code, 200)
            self.assertEqual(ready.json()["model"], "0.6b")
            self.assertEqual(ready.json()["app_version"], "20260821")
            self.assertEqual(ready.json()["model_id"], "Qwen/Qwen3-ASR-0.6B")
            self.assertEqual(ready.json()["queue_depth"], 0)
            self.assertEqual(engine.load_count, 1)
        finally:
            await self.close_app(client, lifespan)
        self.assertEqual(engine.unload_count, 1)

    async def test_transcribe_contract_and_argument_forwarding(self) -> None:
        _, engine, client, lifespan = await self.use_app()
        try:
            response = await client.post(
                "/transcribe",
                files={"audio": ("sample.wav", wav_bytes(), "audio/wav")},
                data={"request_id": "req-1", "language": "Japanese", "context": "耳鼻咽喉科"},
            )
            self.assertEqual(response.status_code, 200, response.text)
            body = response.json()
            self.assertEqual(body["schema_version"], 1)
            self.assertEqual(body["request_id"], "req-1")
            self.assertEqual(body["text"], engine.text)
            self.assertEqual(body["language"], "Japanese")
            self.assertEqual(body["provider_metrics"], {})
            self.assertAlmostEqual(body["audio"]["duration_sec"], 0.1, places=3)
            for key in ("queue_sec", "inference_sec", "total_sec", "rtf"):
                self.assertIsInstance(body["timing"][key], (int, float))
            self.assertEqual(engine.calls[0]["language"], "Japanese")
            self.assertEqual(engine.calls[0]["context"], "耳鼻咽喉科")
        finally:
            await self.close_app(client, lifespan)

    async def test_generated_request_id_empty_text_and_cleanup(self) -> None:
        engine = FakeEngine()
        engine.text = ""
        _, engine, client, lifespan = await self.use_app(engine)
        try:
            response = await client.post(
                "/transcribe", files={"audio": ("sample.wav", wav_bytes(), "audio/wav")}
            )
            self.assertEqual(response.status_code, 200)
            self.assertTrue(response.json()["request_id"])
            self.assertEqual(response.json()["text"], "")
            self.assertTrue(engine.paths)
            self.assertFalse(engine.paths[0].exists())
        finally:
            await self.close_app(client, lifespan)

    async def test_exact_context_echo_is_removed_but_other_speech_remains(self) -> None:
        context = "日本の医療現場の会話。聞こえたとおりに書き起こす。推測で補完しない。"
        engine = FakeEngine()
        engine.text = f"はいこんにちは。\n{context}\n{context}\nちょっとごめんなさい。"
        _, _, client, lifespan = await self.use_app(engine)
        try:
            response = await client.post(
                "/transcribe",
                files={"audio": ("sample.wav", wav_bytes(), "audio/wav")},
                data={"context": context},
            )
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()["text"], "はいこんにちは。\nちょっとごめんなさい。")
        finally:
            await self.close_app(client, lifespan)

    async def test_context_only_hallucination_becomes_empty_text(self) -> None:
        context = "聞こえたとおりに書き起こす。"
        engine = FakeEngine()
        engine.text = f"{context}\n{context}\n{context}"
        _, _, client, lifespan = await self.use_app(engine)
        try:
            response = await client.post(
                "/transcribe",
                files={"audio": ("sample.wav", wav_bytes(), "audio/wav")},
                data={"context": context},
            )
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()["text"], "")
        finally:
            await self.close_app(client, lifespan)

    async def test_similar_but_not_identical_speech_is_not_removed(self) -> None:
        context = "推測で補完しない。"
        engine = FakeEngine()
        engine.text = "推測では補完しない。"
        _, _, client, lifespan = await self.use_app(engine)
        try:
            response = await client.post(
                "/transcribe",
                files={"audio": ("sample.wav", wav_bytes(), "audio/wav")},
                data={"context": context},
            )
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()["text"], engine.text)
        finally:
            await self.close_app(client, lifespan)

    async def test_rejects_unsupported_corrupt_too_long_and_too_large(self) -> None:
        settings = APISettings(model_alias="0.6b", max_audio_sec=0.05, max_upload_mib=0.01)
        _, _, client, lifespan = await self.use_app(settings=settings)
        try:
            unsupported = await client.post(
                "/transcribe", files={"audio": ("sample.mp3", b"ID3data", "audio/mpeg")}
            )
            self.assertEqual((unsupported.status_code, unsupported.json()["error"]["code"]), (415, "unsupported_audio"))
            corrupt = await client.post(
                "/transcribe", files={"audio": ("sample.wav", b"RIFFxxxxWAVEbad", "audio/wav")}
            )
            self.assertEqual((corrupt.status_code, corrupt.json()["error"]["code"]), (422, "invalid_audio"))
            too_long = await client.post(
                "/transcribe", files={"audio": ("sample.wav", wav_bytes(0.1), "audio/wav")}
            )
            self.assertEqual((too_long.status_code, too_long.json()["error"]["code"]), (422, "invalid_audio"))
            too_large = await client.post(
                "/transcribe", files={"audio": ("sample.wav", wav_bytes(1.0), "audio/wav")}
            )
            self.assertEqual((too_large.status_code, too_large.json()["error"]["code"]), (413, "audio_too_large"))
        finally:
            await self.close_app(client, lifespan)

    async def test_request_validation_and_common_error_shape(self) -> None:
        settings = APISettings(model_alias="0.6b", max_context_chars=3, max_request_id_chars=4)
        _, _, client, lifespan = await self.use_app(settings=settings)
        try:
            for data in ({"request_id": "12345"}, {"context": "1234"}, {"language": "bad\nvalue"}):
                response = await client.post(
                    "/transcribe", files={"audio": ("sample.wav", wav_bytes(), "audio/wav")}, data=data
                )
                self.assertEqual(response.status_code, 400)
                self.assertEqual(response.json()["error"]["code"], "invalid_request")
                self.assertFalse(response.json()["error"]["retryable"])
            missing = await client.post("/transcribe")
            self.assertEqual(missing.status_code, 400)
            self.assertEqual(missing.json()["schema_version"], 1)
        finally:
            await self.close_app(client, lifespan)

    async def test_engine_failure_and_oom_are_classified_and_cleaned(self) -> None:
        for error, status, code in (
            (RuntimeError("ordinary failure /private/path"), 500, "inference_failed"),
            (RuntimeError("CUDA out of memory"), 503, "gpu_out_of_memory"),
        ):
            engine = FakeEngine()
            engine.error = error
            _, engine, client, lifespan = await self.use_app(engine)
            try:
                response = await client.post(
                    "/transcribe", files={"audio": ("sample.wav", wav_bytes(), "audio/wav")}
                )
                self.assertEqual((response.status_code, response.json()["error"]["code"]), (status, code))
                self.assertNotIn("private", response.text)
                self.assertFalse(engine.paths[0].exists())
            finally:
                await self.close_app(client, lifespan)

    async def test_timeout_then_worker_remains_available(self) -> None:
        engine = FakeEngine()
        engine.release = threading.Event()
        settings = APISettings(model_alias="0.6b", request_timeout_sec=0.05)
        _, engine, client, lifespan = await self.use_app(engine, settings)
        try:
            first = await client.post(
                "/transcribe", files={"audio": ("one.wav", wav_bytes(), "audio/wav")}
            )
            self.assertEqual((first.status_code, first.json()["error"]["code"]), (504, "request_timeout"))
            engine.release.set()
            await asyncio.to_thread(lambda: time.sleep(0.02))
            engine.release = None
            second = await client.post(
                "/transcribe", files={"audio": ("two.wav", wav_bytes(), "audio/wav")}
            )
            self.assertEqual(second.status_code, 200, second.text)
        finally:
            engine.release.set() if engine.release is not None else None
            await self.close_app(client, lifespan)

    async def test_queue_full_and_serial_inference(self) -> None:
        engine = FakeEngine()
        engine.release = threading.Event()
        settings = APISettings(model_alias="0.6b", max_queue_size=1, request_timeout_sec=2)
        app, engine, client, lifespan = await self.use_app(engine, settings)
        try:
            request = lambda name: client.post(
                "/transcribe", files={"audio": (name, wav_bytes(), "audio/wav")}
            )
            first_task = asyncio.create_task(request("one.wav"))
            await asyncio.to_thread(engine.started.wait, 1)
            second_task = asyncio.create_task(request("two.wav"))
            for _ in range(100):
                if app.state.service.queue_depth == 1:
                    break
                await asyncio.sleep(0.005)
            third = await request("three.wav")
            self.assertEqual((third.status_code, third.json()["error"]["code"]), (429, "queue_full"))
            engine.release.set()
            first, second = await asyncio.gather(first_task, second_task)
            self.assertEqual(first.status_code, 200)
            self.assertEqual(second.status_code, 200)
            self.assertEqual(engine.max_active, 1)
            self.assertEqual(engine.load_count, 1)
        finally:
            engine.release.set()
            await self.close_app(client, lifespan)

    async def test_two_delayed_requests_never_run_in_parallel(self) -> None:
        engine = FakeEngine()
        engine.delay = 0.03
        _, engine, client, lifespan = await self.use_app(engine)
        try:
            responses = await asyncio.gather(*[
                client.post("/transcribe", files={"audio": (f"{i}.wav", wav_bytes(), "audio/wav")})
                for i in range(2)
            ])
            self.assertEqual([response.status_code for response in responses], [200, 200])
            self.assertEqual(engine.max_active, 1)
            self.assertEqual(engine.load_count, 1)
        finally:
            await self.close_app(client, lifespan)


class SettingsTests(unittest.TestCase):
    def test_settings_validation(self) -> None:
        with self.assertRaisesRegex(ValueError, "localhost"):
            APISettings.from_config({"api": {"host": "0.0.0.0"}})
        with self.assertRaisesRegex(ValueError, "max_queue_size"):
            APISettings.from_config({"api": {"max_queue_size": 0}})
        with self.assertRaisesRegex(ValueError, "port"):
            APISettings.from_config({"api": {"port": "8010"}})


if __name__ == "__main__":
    unittest.main()
