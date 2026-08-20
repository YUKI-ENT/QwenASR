from __future__ import annotations

import json
import sys
import tempfile
import types
import unittest
import wave
from pathlib import Path
from unittest.mock import patch

from audio_utils import AudioError, get_audio_duration, list_audio_files
from cli_common import load_config, resolve_model
from qwen_asr_engine import QwenASREngine


class AudioUtilsTests(unittest.TestCase):
    def test_wav_duration_and_listing(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            wav_path = Path(temp_dir) / "sample.wav"
            with wave.open(str(wav_path), "wb") as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(16000)
                wav_file.writeframes(b"\x00\x00" * 32000)
            self.assertAlmostEqual(get_audio_duration(wav_path), 2.0, places=3)
            self.assertEqual(list_audio_files(temp_dir), [wav_path])

    def test_rejects_unsupported_extension(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "sample.txt"
            path.write_text("not audio", encoding="utf-8")
            with self.assertRaises(AudioError):
                get_audio_duration(path)


class ConfigTests(unittest.TestCase):
    def test_local_model_path_is_relative_to_config(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config_path = root / "config.json"
            config_path.write_text(
                json.dumps({"local_model_paths": {"0.6b": "models/local"}}),
                encoding="utf-8",
            )
            config, base = load_config(config_path)
            self.assertEqual(
                resolve_model(config, "0.6b", base),
                str((root / "models/local").resolve()),
            )

    def test_direct_relative_model_path_is_relative_to_config(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            model_dir = root / "models" / "direct"
            model_dir.mkdir(parents=True)
            config_path = root / "config.json"
            config_path.write_text(
                json.dumps({"model": "models/direct"}), encoding="utf-8"
            )
            config, base = load_config(config_path)
            self.assertEqual(resolve_model(config, None, base), str(model_dir.resolve()))


class EngineContractTests(unittest.TestCase):
    def test_official_wrapper_contract_without_real_model(self) -> None:
        fake_torch = types.ModuleType("torch")
        fake_torch.bfloat16 = object()
        fake_torch.float16 = object()
        fake_torch.float32 = object()
        fake_torch.cuda = types.SimpleNamespace(is_available=lambda: False)

        class FakeModel:
            @classmethod
            def from_pretrained(cls, model_path: str, **kwargs: object) -> "FakeModel":
                self = cls()
                self.model_path = model_path
                self.kwargs = kwargs
                return self

            def transcribe(self, **kwargs: object) -> list[object]:
                self.transcribe_kwargs = kwargs
                return [types.SimpleNamespace(text="テストです。", language="Japanese")]

        fake_qwen = types.ModuleType("qwen_asr")
        fake_qwen.Qwen3ASRModel = FakeModel

        with tempfile.TemporaryDirectory() as temp_dir:
            wav_path = Path(temp_dir) / "sample.wav"
            with wave.open(str(wav_path), "wb") as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(16000)
                wav_file.writeframes(b"\x00\x00" * 16000)
            with patch.dict(sys.modules, {"torch": fake_torch, "qwen_asr": fake_qwen}):
                engine = QwenASREngine(
                    "Qwen/Qwen3-ASR-0.6B", device="cpu", dtype="bfloat16"
                )
                engine.load()
                result = engine.transcribe(
                    wav_path, language="English", context="medical terms"
                )
                self.assertEqual(result.transcript, "テストです。")
                self.assertEqual(result.detected_language, "Japanese")
                self.assertGreaterEqual(result.rtf, 0.0)
                self.assertEqual(engine.model.transcribe_kwargs["language"], "English")
                self.assertEqual(engine.model.transcribe_kwargs["context"], "medical terms")
                self.assertEqual(engine.language, "Japanese")
                engine.unload()


if __name__ == "__main__":
    unittest.main()
