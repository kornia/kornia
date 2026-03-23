# LICENSE HEADER MANAGED BY add-license-header
#
# Copyright 2018 Kornia Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""Benchmark for kornia.geometry.depth.depth_to_normals across image sizes.

Usage:
    python benchmarks/geometry/depth_to_normals.py
    python benchmarks/geometry/depth_to_normals.py --cuda
"""

from __future__ import annotations

import argparse
import datetime
import platform
import shutil
import subprocess
import time

import torch

from kornia.geometry.depth import depth_to_normals

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _sync(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()


def bench(fn, *args, warmup: int = 5, reps: int = 20, device: str = "cpu", label: str = "") -> float:
    for _ in range(warmup):
        fn(*args)
    _sync(device)
    t0 = time.perf_counter()
    for _ in range(reps):
        fn(*args)
    _sync(device)
    ms = (time.perf_counter() - t0) / reps * 1000
    print(f"  {label:<64s}: {ms:8.3f} ms")
    return ms


def _print_env() -> None:
    date = datetime.datetime.now(tz=datetime.UTC).strftime("%Y-%m-%d %H:%M:%S UTC")
    git = shutil.which("git") or "git"
    try:
        commit = subprocess.check_output([git, "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        commit = "unknown"
    cpu = platform.processor() or platform.machine()
    print(f"  date   : {date}")
    print(f"  commit : {commit}")
    print(f"  cpu    : {cpu}")
    if torch.cuda.is_available():
        print(f"  gpu    : {torch.cuda.get_device_name(0)}")


# ─────────────────────────────────────────────────────────────────────────────
# Benchmark
# ─────────────────────────────────────────────────────────────────────────────


def run(device: str) -> None:
    print(f"\n{'=' * 72}")
    print(f"  DEVICE : {device.upper()}")
    print(f"{'=' * 72}")

    K_base = torch.eye(3, device=device, dtype=torch.float32)
    K_base[0, 0] = K_base[1, 1] = 500.0
    K_base[0, 2] = K_base[1, 2] = 320.0

    configs = [
        ("B=1 H=64   W=64  ", 1, 64, 64),
        ("B=1 H=256  W=256 ", 1, 256, 256),
        ("B=4 H=480  W=640 ", 4, 480, 640),
        ("B=1 H=720  W=1280", 1, 720, 1280),
    ]

    print(f"\n  {'config':<20s}  {'depth_to_normals':>20s}")
    print(f"  {'-' * 20}  {'-' * 20}")
    for label, B, H, W in configs:
        K = K_base.unsqueeze(0).expand(B, -1, -1).contiguous()
        depth = torch.rand(B, 1, H, W, device=device, dtype=torch.float32).add_(0.1)
        bench(depth_to_normals, depth, K, device=device, label=label)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cuda", action="store_true")
    args = parser.parse_args()

    _print_env()
    run("cpu")
    if args.cuda:
        if torch.cuda.is_available():
            run("cuda")
        else:
            print("\nWarning: --cuda requested but CUDA is not available.")
