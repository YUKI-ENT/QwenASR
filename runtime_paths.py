"""Paths that work consistently from source and from a frozen executable."""

from __future__ import annotations

import sys
from pathlib import Path


def application_dir() -> Path:
    """Return the directory containing the executable, or the current source cwd."""
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path.cwd()


def application_path(name: str) -> Path:
    return application_dir() / name
