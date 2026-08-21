"""Audio validation, duration probing, and optional ffmpeg conversion."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
import wave
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


SUPPORTED_EXTENSIONS = {".wav", ".mp3", ".m4a", ".flac"}


class AudioError(RuntimeError):
    """Raised when an input cannot be prepared for ASR."""


def _find_program(name: str) -> str | None:
    """Find a tool on PATH or next to a frozen Windows executable."""
    located = shutil.which(name)
    if located is not None:
        return located
    if getattr(sys, "frozen", False):
        executable_dir = Path(sys.executable).resolve().parent
        candidates = (executable_dir / name, executable_dir / "bin" / name)
        if sys.platform == "win32" and not name.lower().endswith(".exe"):
            candidates += (
                executable_dir / f"{name}.exe",
                executable_dir / "bin" / f"{name}.exe",
            )
        for candidate in candidates:
            if candidate.is_file():
                return str(candidate)
    return None


def validate_audio_file(path: str | Path) -> Path:
    audio_path = Path(path).expanduser().resolve()
    if not audio_path.is_file():
        raise AudioError(f"音声ファイルが見つかりません: {audio_path}")
    if audio_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise AudioError(
            f"未対応の音声形式です: {audio_path.suffix or '(拡張子なし)'} "
            f"(対応形式: {supported})"
        )
    return audio_path


def _duration_with_soundfile(path: Path) -> float:
    import soundfile as sf

    info = sf.info(str(path))
    if info.samplerate <= 0:
        raise AudioError(f"サンプルレートを取得できません: {path}")
    return float(info.frames) / float(info.samplerate)


def _duration_with_wave(path: Path) -> float:
    with wave.open(str(path), "rb") as wav_file:
        rate = wav_file.getframerate()
        if rate <= 0:
            raise AudioError(f"サンプルレートを取得できません: {path}")
        return wav_file.getnframes() / float(rate)


def _duration_with_ffprobe(path: Path) -> float:
    ffprobe = _find_program("ffprobe")
    if ffprobe is None:
        raise AudioError(
            "音声長を取得できず、ffprobeも見つかりません。ffmpegをインストールしてください。"
        )
    command = [
        ffprobe,
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "json",
        str(path),
    ]
    try:
        completed = subprocess.run(
            command, check=True, capture_output=True, text=True, timeout=30
        )
        duration = float(json.loads(completed.stdout)["format"]["duration"])
    except (subprocess.SubprocessError, KeyError, ValueError, json.JSONDecodeError) as exc:
        raise AudioError(f"音声長を取得できません: {path}\n{exc}") from exc
    if duration <= 0:
        raise AudioError(f"音声長が0秒以下です: {path}")
    return duration


def get_audio_duration(path: str | Path) -> float:
    audio_path = validate_audio_file(path)
    try:
        return _duration_with_soundfile(audio_path)
    except (ImportError, RuntimeError):
        if audio_path.suffix.lower() == ".wav":
            try:
                return _duration_with_wave(audio_path)
            except (wave.Error, EOFError):
                pass
        return _duration_with_ffprobe(audio_path)


def _convert_to_wav(source: Path, destination: Path) -> None:
    ffmpeg = _find_program("ffmpeg")
    if ffmpeg is None:
        raise AudioError(
            f"{source.suffix.lower()} の変換に必要なffmpegが見つかりません。"
        )
    command = [
        ffmpeg,
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(source),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        str(destination),
    ]
    try:
        subprocess.run(command, check=True, capture_output=True, text=True, timeout=600)
    except FileNotFoundError as exc:
        raise AudioError("ffmpegが見つかりません。") from exc
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or "ffmpeg conversion failed").strip()
        raise AudioError(f"音声変換に失敗しました: {source}\n{detail}") from exc
    except subprocess.TimeoutExpired as exc:
        raise AudioError(f"音声変換がタイムアウトしました: {source}") from exc


@contextmanager
def prepared_audio(path: str | Path) -> Iterator[Path]:
    """Yield an ASR-ready path, converting MP3/M4A to mono 16 kHz WAV."""
    audio_path = validate_audio_file(path)
    if audio_path.suffix.lower() not in {".mp3", ".m4a"}:
        yield audio_path
        return

    with tempfile.TemporaryDirectory(prefix="qwen_asr_") as temp_dir:
        converted = Path(temp_dir) / f"{audio_path.stem}.wav"
        _convert_to_wav(audio_path, converted)
        yield converted


def list_audio_files(directory: str | Path) -> list[Path]:
    root = Path(directory).expanduser().resolve()
    if not root.is_dir():
        raise AudioError(f"音声フォルダーが見つかりません: {root}")
    return sorted(
        path for path in root.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )
