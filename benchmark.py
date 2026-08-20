#!/usr/bin/env python3
"""Benchmark both Qwen3-ASR model sizes over a directory."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from audio_utils import AudioError, list_audio_files
from cli_common import create_engine, load_config, make_result_document, resolve_model
from qwen_asr_engine import ASRError


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Qwen3-ASR 0.6B/1.7B 一括比較")
    parser.add_argument("audio_dir", help="音声ファイルを含むフォルダー")
    parser.add_argument("--models", nargs="+", choices=("0.6b", "1.7b"), default=["0.6b", "1.7b"])
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    return parser.parse_args()


def _csv_row(document: dict[str, Any]) -> dict[str, Any]:
    peak = document["gpu"]["inference_peak"]
    return {
        "file": document["audio_file"],
        "engine": document["engine"],
        "backend": document["backend"],
        "model": document["model"],
        "duration_sec": document["audio_duration_sec"],
        "elapsed_sec": document["elapsed_sec"],
        "rtf": document["rtf"],
        "vram_peak_torch_mib": peak["torch_allocated_mib"],
        "vram_peak_process_mib": peak["process_mib"],
        "transcript": document["transcript"],
        "error": document.get("error", ""),
    }


def _save(outputs: list[dict[str, Any]], results_dir: Path) -> tuple[Path, Path]:
    results_dir.mkdir(parents=True, exist_ok=True)
    outputs.sort(key=lambda item: (str(item.get("audio_file", "")), str(item.get("model", ""))))
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    json_path = results_dir / f"{stamp}_benchmark.json"
    csv_path = results_dir / f"{stamp}_benchmark.csv"
    payload = {
        "schema_version": 1,
        "timestamp": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "results": outputs,
    }
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    fieldnames = list(_csv_row(outputs[0]).keys()) if outputs else [
        "file", "engine", "backend", "model", "duration_sec", "elapsed_sec",
        "rtf", "vram_peak_torch_mib", "vram_peak_process_mib", "transcript", "error"
    ]
    with csv_path.open("w", encoding="utf-8-sig", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(_csv_row(item) for item in outputs)
    return json_path, csv_path


def main() -> int:
    args = parse_args()
    outputs: list[dict[str, Any]] = []
    try:
        files = list_audio_files(args.audio_dir)
        if not files:
            raise AudioError(f"対応音声ファイルがありません: {Path(args.audio_dir).resolve()}")
        config, base_dir = load_config(args.config)
        for alias in args.models:
            model_path = resolve_model(config, alias, base_dir)
            print(f"\n=== Loading {model_path} ===", flush=True)
            engine = create_engine(
                config, model_path, base_dir, offline=True if args.offline else None
            )
            try:
                engine.load()
                for index, audio_file in enumerate(files, start=1):
                    print(f"[{index}/{len(files)}] {audio_file.name}", flush=True)
                    try:
                        result = engine.transcribe(audio_file)
                        outputs.append(make_result_document(result))
                        print(f"  RTF={result.rtf:.4f}  {result.transcript}")
                    except (ASRError, AudioError) as exc:
                        if not args.continue_on_error:
                            raise
                        outputs.append({
                            "schema_version": 1,
                            "timestamp": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
                            "audio_file": str(audio_file),
                            "engine": "qwen3-asr",
                            "backend": "transformers",
                            "model": model_path,
                            "audio_duration_sec": None,
                            "elapsed_sec": None,
                            "rtf": None,
                            "transcript": "",
                            "error": str(exc),
                            "gpu": {"inference_peak": {"torch_allocated_mib": None, "process_mib": None}},
                        })
                        print(f"  Error: {exc}", file=sys.stderr)
            finally:
                after_unload = engine.unload()
                print(
                    f"Unloaded; process VRAM: "
                    f"{after_unload.process_mib if after_unload.process_mib is not None else 'N/A'} MiB"
                )
        json_path, csv_path = _save(outputs, Path(args.results_dir).resolve())
        print(f"\nSaved: {json_path}\nSaved: {csv_path}")
        return 0
    except (ASRError, AudioError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\n中断しました。", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
