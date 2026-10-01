"""Hardware / environment inspection. Writes results/environment.json.

Usage:
    python -m src.utils.hardware
"""
from __future__ import annotations

import json
import logging
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

log = logging.getLogger("hardware")


def inspect_environment() -> dict[str, Any]:
    import psutil

    info: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "cpu": platform.processor(),
        "cpu_physical_cores": psutil.cpu_count(logical=False),
        "cpu_logical_cores": psutil.cpu_count(logical=True),
        "ram_gb": round(psutil.virtual_memory().total / 1024**3, 1),
    }
    try:
        import torch

        info["torch"] = torch.__version__
        info["torch_cuda_build"] = torch.version.cuda
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            info["gpu"] = p.name
            info["gpu_vram_gb"] = round(p.total_memory / 1024**3, 2)
            info["gpu_compute_capability"] = f"{p.major}.{p.minor}"
            info["bf16_supported"] = torch.cuda.is_bf16_supported()
            # Real kernel launch: catches wheels that lack kernels for this architecture.
            x = torch.randn(1024, 1024, device="cuda", dtype=torch.float16)
            info["cuda_matmul_smoke_test"] = bool(torch.isfinite(x @ x).all().item())
    except ImportError:
        info["torch"] = "NOT INSTALLED"
    except Exception as e:  # noqa: BLE001 - report any CUDA failure verbatim
        info["cuda_error"] = repr(e)
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total,power.limit", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=20,
        )
        info["nvidia_smi"] = out.stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        info["nvidia_smi"] = "unavailable"
    return info


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    info = inspect_environment()
    out = Path("results/environment.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
