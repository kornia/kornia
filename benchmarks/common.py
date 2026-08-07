"""Shared methodology utilities for kornia benchmarks.

Non-negotiable methodology (W3 of the agent-era plan): warmup, device sync inside timed
regions, multiple repeats with median + spread, pinned seeds, and recorded hardware/software
metadata. ``torch.utils.benchmark.Timer.blocked_autorange`` supplies warmup, repeats and CUDA
sync; this module adds spread reporting, MPS sync support, metadata capture, and
machine-readable JSON export so every run is comparable and citable.
"""

from __future__ import annotations

import json
import math
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

import torch
import torch.utils.benchmark as bench


def time_us(
    fn: Callable[[], object], min_run_time: float = 1.0, sync: Optional[Callable[[], None]] = None
) -> tuple[float, float]:
    """Median and interquartile-range wall clock of ``fn`` in microseconds.

    ``blocked_autorange`` warms up, runs many repeats, and synchronizes CUDA. Devices it does
    not sync (MPS) pass their sync as ``sync`` so it lands inside the timed region. Returns
    ``(nan, nan)`` if ``fn`` raises, so callers can render a skip cell instead of dying.
    """
    stmt = "fn(); sync()" if sync is not None else "fn()"
    try:
        m = bench.Timer(stmt=stmt, globals={"fn": fn, "sync": sync}).blocked_autorange(min_run_time=min_run_time)
        return m.median * 1e6, m.iqr * 1e6
    except Exception:
        return float("nan"), float("nan")


def git_commit() -> str:
    """Short hash of HEAD, or 'unknown' outside a git checkout."""
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()  # noqa: S607
    except Exception:
        return "unknown"


def _optional_version(module: str) -> Optional[str]:
    try:
        return __import__(module).__version__
    except Exception:
        return None


def run_metadata(device: torch.device) -> dict[str, Any]:
    """Hardware/software metadata embedded in every result file (W3: date, hardware, versions)."""
    import kornia

    meta: dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "kornia": kornia.__version__,
        "device": str(device),
        "torch_num_threads": torch.get_num_threads(),
        "opencv": _optional_version("cv2"),
        "torchvision": _optional_version("torchvision"),
        "numpy": _optional_version("numpy"),
    }
    if device.type == "cuda":
        meta["cuda_device"] = torch.cuda.get_device_name(device)
        meta["cuda_version"] = torch.version.cuda
    return meta


def _sanitize(obj: Any) -> Any:
    if isinstance(obj, float) and math.isnan(obj):
        return None
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    return obj


def save_json(path: "str | Path", metadata: dict[str, Any], results: list[dict[str, Any]]) -> Path:
    """Write one run as strict-valid JSON ``{"metadata": ..., "results": [...]}`` (NaN → null)."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(_sanitize({"metadata": metadata, "results": results}), indent=2) + "\n")
    return out
