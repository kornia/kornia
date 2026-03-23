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
"""Benchmark for project/unproject_points and calibration distortion functions.

Usage:
    python benchmarks/geometry/project_distort.py
    python benchmarks/geometry/project_distort.py --cuda
"""

from __future__ import annotations

import argparse
import datetime
import platform
import subprocess
import time

import torch

from kornia.geometry.calibration.distort import distort_points
from kornia.geometry.calibration.undistort import undistort_points
from kornia.geometry.camera import project_points, unproject_points
from kornia.geometry.conversions import denormalize_points_with_intrinsics, normalize_points_with_intrinsics

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _sync(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()


def bench(fn, *args, warmup: int = 10, reps: int = 50, device: str = "cpu", label: str = "") -> float:
    for _ in range(warmup):
        fn(*args)
    _sync(device)
    t0 = time.perf_counter()
    for _ in range(reps):
        fn(*args)
    _sync(device)
    ms = (time.perf_counter() - t0) / reps * 1000
    print(f"  {label:<60s}: {ms:8.3f} ms")
    return ms


def _print_env() -> None:
    date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        commit = "unknown"
    cpu = platform.processor() or platform.machine()
    print(f"  date   : {date}")
    print(f"  commit : {commit}")
    print(f"  cpu    : {cpu}")
    if torch.cuda.is_available():
        print(f"  gpu    : {torch.cuda.get_device_name(0)}")


# ─────────────────────────────────────────────────────────────────────────────
# Benchmarks
# ─────────────────────────────────────────────────────────────────────────────


def bench_project_unproject(device: str) -> None:
    print(f"\n--- project_points / unproject_points  device={device} ---")

    K_base = torch.eye(3, device=device)
    K_base[0, 0] = K_base[1, 1] = 500.0
    K_base[0, 2] = K_base[1, 2] = 320.0

    configs = [
        ("B=1   N=1K  ", 1, 1_000),
        ("B=8   N=10K ", 8, 10_000),
        ("B=32  N=100K", 32, 100_000),
    ]

    for label, B, N in configs:
        K_b = K_base.unsqueeze(0).expand(B, -1, -1)
        pts3 = torch.rand(B, N, 3, device=device).add_(0.5)
        pts2 = torch.rand(B, N, 2, device=device).mul_(640.0)
        pts2_norm = torch.rand(B, N, 2, device=device)
        depth = torch.ones(B, N, 1, device=device)

        print(f"\n  {label}")
        bench(project_points, pts3, K_b, device=device, label="project_points")
        bench(unproject_points, pts2, depth, K_b, device=device, label="unproject_points")
        bench(normalize_points_with_intrinsics, pts2_norm, K_b, device=device, label="normalize_points_with_intrinsics")
        bench(
            denormalize_points_with_intrinsics,
            pts2_norm,
            K_b,
            device=device,
            label="denormalize_points_with_intrinsics",
        )


def bench_distort_undistort(device: str) -> None:
    print(f"\n--- distort_points / undistort_points  device={device} ---")

    K_base = torch.eye(3, device=device)
    K_base[0, 0] = K_base[1, 1] = 500.0
    K_base[0, 2] = K_base[1, 2] = 320.0
    dist_base = torch.tensor([0.1, -0.05, 0.001, 0.001, 0.02, 0.01, -0.005, 0.002], device=device)

    configs = [
        ("B=1   N=1K  ", 1, 1_000),
        ("B=1   N=100K", 1, 100_000),
        ("B=32  N=10K ", 32, 10_000),
    ]

    for label, B, N in configs:
        K_b = K_base.unsqueeze(0).expand(B, -1, -1).contiguous()
        dist_b = dist_base.unsqueeze(0).expand(B, -1).contiguous()
        pts2 = torch.rand(B, N, 2, device=device).mul_(640.0)

        print(f"\n  {label}")
        bench(distort_points, pts2, K_b, dist_b, device=device, label="distort_points")
        bench(undistort_points, pts2, K_b, dist_b, device=device, label="undistort_points (5 iters)")


def run(device: str) -> None:
    sep = "=" * 72
    print(f"\n{sep}\n  DEVICE: {device.upper()}\n{sep}")
    bench_project_unproject(device)
    bench_distort_undistort(device)


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
