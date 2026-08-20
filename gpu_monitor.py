"""PyTorch and nvidia-smi GPU memory measurement helpers."""

from __future__ import annotations

import os
import shutil
import subprocess
import threading
from dataclasses import asdict, dataclass
from typing import Any


def _mib(value: int | float) -> float:
    return round(float(value) / (1024.0 * 1024.0), 1)


@dataclass
class GPUSnapshot:
    torch_allocated_mib: float | None = None
    torch_reserved_mib: float | None = None
    process_mib: float | None = None

    def to_dict(self) -> dict[str, float | None]:
        return asdict(self)


class GPUMonitor:
    def __init__(self, device: str, poll_interval_sec: float = 0.05):
        self.device = device
        self.poll_interval_sec = max(0.01, float(poll_interval_sec))
        self._torch: Any = None
        self._device_index = self._parse_device_index(device)
        self._peak_process_mib: float | None = None
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    @staticmethod
    def _parse_device_index(device: str) -> int:
        if device == "cuda":
            return 0
        if device.startswith("cuda:"):
            try:
                return int(device.split(":", 1)[1])
            except ValueError as exc:
                raise ValueError(f"不正なCUDA deviceです: {device}") from exc
        return 0

    @property
    def is_cuda(self) -> bool:
        return self.device.startswith("cuda")

    def initialize(self) -> None:
        if not self.is_cuda:
            return
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("PyTorchが未導入です。READMEの手順でインストールしてください。") from exc
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDAが使用できません。NVIDIA driverとCUDA対応PyTorchを確認してください。"
            )
        if self._device_index >= torch.cuda.device_count():
            raise RuntimeError(
                f"CUDA device {self._device_index} が見つかりません "
                f"(検出数: {torch.cuda.device_count()})"
            )
        self._torch = torch
        torch.cuda.set_device(self._device_index)

    def synchronize(self) -> None:
        if self._torch is not None:
            self._torch.cuda.synchronize(self._device_index)

    def reset_peak(self) -> None:
        if self._torch is not None:
            self.synchronize()
            self._torch.cuda.reset_peak_memory_stats(self._device_index)
        self._peak_process_mib = self._nvidia_smi_process_mib()

    def _nvidia_smi_process_mib(self) -> float | None:
        executable = shutil.which("nvidia-smi")
        if executable is None:
            return None
        command = [
            executable,
            f"--id={self._device_index}",
            "--query-compute-apps=pid,used_memory",
            "--format=csv,noheader,nounits",
        ]
        try:
            output = subprocess.run(
                command, check=True, capture_output=True, text=True, timeout=3
            ).stdout
        except (OSError, subprocess.SubprocessError):
            return None
        current_pid = os.getpid()
        for line in output.splitlines():
            columns = [value.strip() for value in line.split(",")]
            if len(columns) >= 2:
                try:
                    if int(columns[0]) == current_pid:
                        return float(columns[1])
                except ValueError:
                    continue
        return 0.0

    def snapshot(self) -> GPUSnapshot:
        allocated = reserved = None
        if self._torch is not None:
            allocated = _mib(self._torch.cuda.memory_allocated(self._device_index))
            reserved = _mib(self._torch.cuda.memory_reserved(self._device_index))
        return GPUSnapshot(allocated, reserved, self._nvidia_smi_process_mib())

    def _poll(self) -> None:
        while not self._stop_event.wait(self.poll_interval_sec):
            value = self._nvidia_smi_process_mib()
            if value is not None:
                self._peak_process_mib = max(self._peak_process_mib or 0.0, value)

    def start_peak_polling(self) -> None:
        if not self.is_cuda or self._thread is not None:
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()

    def stop_peak_polling(self) -> None:
        if self._thread is None:
            return
        self._stop_event.set()
        self._thread.join(timeout=max(1.0, self.poll_interval_sec * 4))
        self._thread = None
        final_value = self._nvidia_smi_process_mib()
        if final_value is not None:
            self._peak_process_mib = max(self._peak_process_mib or 0.0, final_value)

    def peak(self) -> GPUSnapshot:
        peak_allocated = peak_reserved = None
        if self._torch is not None:
            peak_allocated = _mib(
                self._torch.cuda.max_memory_allocated(self._device_index)
            )
            peak_reserved_fn = getattr(self._torch.cuda, "max_memory_reserved", None)
            if peak_reserved_fn is not None:
                peak_reserved = _mib(peak_reserved_fn(self._device_index))
        return GPUSnapshot(peak_allocated, peak_reserved, self._peak_process_mib)

    def gpu_name(self) -> str | None:
        if self._torch is None:
            return None
        return str(self._torch.cuda.get_device_name(self._device_index))

