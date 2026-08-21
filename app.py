#!/usr/bin/env python3
"""Transcribe one audio file with Qwen3-ASR."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from audio_utils import AudioError, validate_audio_file
from cli_common import create_engine, load_config, print_result, resolve_model, save_result
from qwen_asr_engine import ASRError
from runtime_paths import application_path
from version import APP_VERSION


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Qwen3-ASR 日本語音声認識テスト (Transformers backend)"
    )
    parser.add_argument("--version", action="version", version=APP_VERSION)
    parser.add_argument("audio_file", help="wav/mp3/m4a/flac 音声ファイル")
    parser.add_argument("--model", choices=("0.6b", "1.7b"), help="モデル切替")
    parser.add_argument(
        "--config", default=str(application_path("config.json")), help="設定JSON"
    )
    parser.add_argument(
        "--results-dir", default=str(application_path("results")), help="結果保存先"
    )
    parser.add_argument("--offline", action="store_true", help="外部アクセスを禁止")
    unload = parser.add_mutually_exclusive_group()
    unload.add_argument("--unload-after", dest="unload_after", action="store_true")
    unload.add_argument("--no-unload-after", dest="unload_after", action="store_false")
    parser.set_defaults(unload_after=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    engine = None
    try:
        config, base_dir = load_config(args.config)
        model = resolve_model(config, args.model, base_dir)
        validate_audio_file(args.audio_file)
        engine = create_engine(
            config, model, base_dir, offline=True if args.offline else None
        )
        engine.load()
        result = engine.transcribe(args.audio_file)
        unload_after = (
            bool(config.get("unload_after", True))
            if args.unload_after is None
            else args.unload_after
        )
        if unload_after:
            result.after_unload = engine.unload()
        output = save_result(result, Path(args.results_dir))
        print_result(result)
        print(f"\nSaved: {output}")
        return 0
    except (ASRError, AudioError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\n中断しました。", file=sys.stderr)
        return 130
    finally:
        if engine is not None and engine.model is not None:
            engine.unload()


if __name__ == "__main__":
    raise SystemExit(main())
