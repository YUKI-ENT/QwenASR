"""PyInstaller hook for qwen-asr's runtime imports and package metadata."""

from PyInstaller.utils.hooks import collect_all


datas, binaries, hiddenimports = collect_all("qwen_asr")
