"""Wall-clock and GPU-memory measurement for experiments."""
from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Iterator


class ResourceMonitor:
    def __init__(self) -> None:
        self.seconds = 0.0
        self.peak_gpu_mb: float | None = None

    @contextmanager
    def track(self) -> Iterator["ResourceMonitor"]:
        try:
            import torch

            cuda = torch.cuda.is_available()
        except ImportError:
            cuda = False
        if cuda:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        try:
            yield self
        finally:
            if cuda:
                torch.cuda.synchronize()
                self.peak_gpu_mb = round(torch.cuda.max_memory_allocated() / 1024**2, 1)
            self.seconds = time.perf_counter() - t0
