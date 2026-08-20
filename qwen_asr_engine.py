"""Qwen3-ASR Transformers backend wrapper."""

from __future__ import annotations

import gc
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from audio_utils import get_audio_duration, prepared_audio
from gpu_monitor import GPUMonitor, GPUSnapshot


class ASRError(RuntimeError):
    """Human-readable ASR failure."""


@dataclass
class TranscriptionResult:
    model: str
    audio_file: str
    audio_duration_sec: float
    elapsed_sec: float
    rtf: float
    transcript: str
    detected_language: str | None
    gpu_name: str | None
    before_load: GPUSnapshot
    after_load: GPUSnapshot
    inference_peak: GPUSnapshot
    after_inference: GPUSnapshot
    after_unload: GPUSnapshot | None = None

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["backend"] = "transformers"
        value["engine"] = "qwen3-asr"
        value["gpu"] = {
            "name": value.pop("gpu_name"),
            "before_load": value.pop("before_load"),
            "after_load": value.pop("after_load"),
            "inference_peak": value.pop("inference_peak"),
            "after_inference": value.pop("after_inference"),
            "after_unload": value.pop("after_unload"),
        }
        return value


def resolve_dtype(torch_module: Any, name: str) -> Any:
    aliases = {
        "bfloat16": "bfloat16",
        "bf16": "bfloat16",
        "float16": "float16",
        "fp16": "float16",
        "float32": "float32",
        "fp32": "float32",
    }
    attribute = aliases.get(name.lower())
    if attribute is None:
        raise ASRError(f"未対応のdtypeです: {name}")
    return getattr(torch_module, attribute)


class QwenASREngine:
    def __init__(
        self,
        model_path: str,
        device: str = "cuda:0",
        dtype: str = "bfloat16",
        language: str | None = "Japanese",
        max_new_tokens: int = 256,
        max_inference_batch_size: int = 1,
        cache_dir: str | Path | None = None,
        offline: bool = False,
        poll_interval_sec: float = 0.05,
    ):
        self.model_path = model_path
        self.device = device
        self.dtype_name = dtype
        self.language = language
        self.max_new_tokens = int(max_new_tokens)
        self.max_inference_batch_size = int(max_inference_batch_size)
        self.cache_dir = str(Path(cache_dir).expanduser().resolve()) if cache_dir else None
        self.offline = bool(offline)
        self.monitor = GPUMonitor(device, poll_interval_sec)
        self.model: Any = None
        self.before_load: GPUSnapshot | None = None
        self.after_load: GPUSnapshot | None = None

    def load(self) -> None:
        if self.model is not None:
            return
        self.monitor.initialize()
        if self.device == "cpu" and self.dtype_name.lower() in {"float16", "fp16"}:
            raise ASRError("CPUではfloat16を使用できません。dtypeをfloat32にしてください。")
        self.before_load = self.monitor.snapshot()
        if self.cache_dir:
            # qwen-asr forwards cache_dir to the model but not to its processor.
            # Setting HF_HOME before importing qwen_asr keeps both in this project.
            os.environ.setdefault("HF_HOME", self.cache_dir)
        if self.offline:
            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
        try:
            import torch
            from qwen_asr import Qwen3ASRModel
        except ImportError as exc:
            raise ASRError(
                "Qwen3-ASRの依存パッケージが未導入です。READMEのセットアップを実行してください。"
            ) from exc

        local_path = Path(self.model_path).expanduser()
        looks_like_path = local_path.is_absolute() or self.model_path.startswith((".", "~"))
        if looks_like_path and not local_path.exists():
            raise ASRError(f"ローカルモデルが見つかりません: {local_path}")

        kwargs: dict[str, Any] = {
            "dtype": resolve_dtype(torch, self.dtype_name),
            "device_map": self.device,
            "max_inference_batch_size": self.max_inference_batch_size,
            "max_new_tokens": self.max_new_tokens,
            "local_files_only": self.offline or local_path.is_dir(),
        }
        if self.cache_dir:
            Path(self.cache_dir).mkdir(parents=True, exist_ok=True)
            kwargs["cache_dir"] = self.cache_dir
        try:
            self.model = Qwen3ASRModel.from_pretrained(self.model_path, **kwargs)
            self.monitor.synchronize()
            self.after_load = self.monitor.snapshot()
        except torch.OutOfMemoryError as exc:
            self._cleanup_model()
            raise ASRError(self._oom_message()) from exc
        except Exception as exc:
            self._cleanup_model()
            if self.offline:
                detail = "オフラインモードです。ローカルキャッシュ/モデルパスを確認してください。"
            else:
                detail = "モデル名、ネットワーク接続、Hugging Faceキャッシュを確認してください。"
            raise ASRError(f"モデルのロードに失敗しました: {self.model_path}\n{detail}\n{exc}") from exc

    def _oom_message(self) -> str:
        short_name = "1.7B" if "1.7B" in self.model_path.upper() else self.model_path
        return f"Qwen3-ASR {short_name}でGPUメモリ不足。0.6Bモデルを試してください。"

    def transcribe(self, audio_file: str | Path) -> TranscriptionResult:
        if self.model is None:
            raise ASRError("モデルがロードされていません。")
        original_path = Path(audio_file).expanduser().resolve()
        duration = get_audio_duration(original_path)
        self.monitor.reset_peak()
        self.monitor.start_peak_polling()
        started = time.perf_counter()
        try:
            with prepared_audio(original_path) as ready_path:
                output = self.model.transcribe(
                    audio=str(ready_path),
                    language=self.language,
                    return_time_stamps=False,
                )
            self.monitor.synchronize()
            elapsed = time.perf_counter() - started
        except Exception as exc:
            try:
                import torch
                is_oom = isinstance(exc, torch.OutOfMemoryError)
            except ImportError:
                is_oom = False
            if is_oom or "out of memory" in str(exc).lower():
                raise ASRError(self._oom_message()) from exc
            raise ASRError(f"音声認識に失敗しました: {original_path}\n{exc}") from exc
        finally:
            self.monitor.stop_peak_polling()

        if not output:
            raise ASRError("Qwen3-ASRから認識結果が返りませんでした。")
        item = output[0]
        transcript = str(getattr(item, "text", item)).strip()
        detected_language = getattr(item, "language", None)
        return TranscriptionResult(
            model=self.model_path,
            audio_file=str(original_path),
            audio_duration_sec=round(duration, 4),
            elapsed_sec=round(elapsed, 4),
            rtf=round(elapsed / duration, 6) if duration > 0 else 0.0,
            transcript=transcript,
            detected_language=str(detected_language) if detected_language else None,
            gpu_name=self.monitor.gpu_name(),
            before_load=self.before_load or GPUSnapshot(),
            after_load=self.after_load or GPUSnapshot(),
            inference_peak=self.monitor.peak(),
            after_inference=self.monitor.snapshot(),
        )

    def _cleanup_model(self) -> None:
        self.model = None
        gc.collect()
        try:
            import torch
            if self.device.startswith("cuda") and torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        except ImportError:
            pass

    def unload(self) -> GPUSnapshot:
        self._cleanup_model()
        return self.monitor.snapshot()
