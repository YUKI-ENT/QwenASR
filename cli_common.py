"""Shared CLI configuration, rendering, and result persistence."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from qwen_asr_engine import QwenASREngine, TranscriptionResult


DEFAULT_CONFIG: dict[str, Any] = {
    "device": "cuda:0",
    "dtype": "bfloat16",
    "model": "Qwen/Qwen3-ASR-0.6B",
    "language": "Japanese",
    "max_new_tokens": 256,
    "max_inference_batch_size": 1,
    "model_cache_dir": "models",
    "offline": False,
    "unload_after": True,
    "nvidia_smi_poll_interval_sec": 0.05,
    "models": {
        "0.6b": "Qwen/Qwen3-ASR-0.6B",
        "1.7b": "Qwen/Qwen3-ASR-1.7B",
    },
    "local_model_paths": {"0.6b": "", "1.7b": ""},
}


def load_config(path: str | Path) -> tuple[dict[str, Any], Path]:
    config_path = Path(path).expanduser().resolve()
    config = dict(DEFAULT_CONFIG)
    if config_path.exists():
        try:
            loaded = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"設定ファイルを読み込めません: {config_path}\n{exc}") from exc
        if not isinstance(loaded, dict):
            raise ValueError(f"設定ファイルのルートはJSON objectである必要があります: {config_path}")
        config.update(loaded)
    return config, config_path.parent


def resolve_model(config: dict[str, Any], alias: str | None, base_dir: Path) -> str:
    if alias:
        local_value = str(config.get("local_model_paths", {}).get(alias, "")).strip()
        if local_value:
            local_path = Path(local_value).expanduser()
            if not local_path.is_absolute():
                local_path = base_dir / local_path
            return str(local_path.resolve())
        try:
            return str(config["models"][alias])
        except (KeyError, TypeError) as exc:
            raise ValueError(f"モデルaliasが設定されていません: {alias}") from exc
    configured = str(config["model"])
    candidate = Path(configured).expanduser()
    if not candidate.is_absolute():
        candidate = base_dir / candidate
    if candidate.exists() or configured.startswith((".", "~")):
        return str(candidate.resolve())
    return configured


def create_engine(
    config: dict[str, Any], model: str, base_dir: Path, offline: bool | None = None
) -> QwenASREngine:
    cache = Path(str(config.get("model_cache_dir", "models"))).expanduser()
    if not cache.is_absolute():
        cache = base_dir / cache
    return QwenASREngine(
        model_path=model,
        device=str(config.get("device", "cuda:0")),
        dtype=str(config.get("dtype", "bfloat16")),
        language=config.get("language", "Japanese"),
        max_new_tokens=int(config.get("max_new_tokens", 256)),
        max_inference_batch_size=int(config.get("max_inference_batch_size", 1)),
        cache_dir=cache,
        offline=bool(config.get("offline", False) if offline is None else offline),
        poll_interval_sec=float(config.get("nvidia_smi_poll_interval_sec", 0.05)),
    )


def make_result_document(result: TranscriptionResult) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "timestamp": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        **result.to_dict(),
    }


def save_result(result: TranscriptionResult, results_dir: str | Path) -> Path:
    destination = Path(results_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    model_slug = Path(result.model).name.lower().replace("_", "-")
    output = destination / f"{stamp}_{model_slug}.json"
    output.write_text(
        json.dumps(make_result_document(result), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return output


def _fmt(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.1f} MiB"


def print_result(result: TranscriptionResult) -> None:
    print(f"Model: {result.model}")
    print(f"Audio duration: {result.audio_duration_sec:.2f} sec")
    print(f"ASR time: {result.elapsed_sec:.2f} sec")
    print(f"RTF: {result.rtf:.4f}")
    if result.gpu_name:
        print(f"GPU: {result.gpu_name}")
    print("\nVRAM (PyTorch allocated / reserved / nvidia-smi process):")
    for label, snapshot in (
        ("before load", result.before_load),
        ("after load", result.after_load),
        ("inference peak", result.inference_peak),
        ("after inference", result.after_inference),
        ("after unload", result.after_unload),
    ):
        if snapshot is not None:
            print(
                f"  {label}: {_fmt(snapshot.torch_allocated_mib)} / "
                f"{_fmt(snapshot.torch_reserved_mib)} / {_fmt(snapshot.process_mib)}"
            )
    print("\nTranscript:")
    print(result.transcript)
