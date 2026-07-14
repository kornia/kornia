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
"""Benchmark for linalg, epipolar projection/triangulation, and calibration distortion functions.

Usage:
    python benchmarks/geometry/linalg_epipolar_distort.py
    python benchmarks/geometry/linalg_epipolar_distort.py --cuda
"""

from __future__ import annotations

import argparse
import datetime
import platform
import shutil
import subprocess
import time

import torch

from kornia.geometry.calibration.distort import distort_points, tilt_projection
from kornia.geometry.epipolar.projection import (
    KRt_from_projection,
    depth_from_point,
    projection_from_KRt,
    projections_from_fundamental,
    scale_intrinsics,
)
from kornia.geometry.epipolar.triangulation import triangulate_points
from kornia.geometry.linalg import (
    batched_dot_product,
    compose_transformations,
    euclidean_distance,
    inverse_transformation,
    point_line_distance,
    relative_transformation,
    transform_points,
)

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
    print(f"  {label:<60s}: {ms:8.4f} ms")
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
# Benchmarks
# ─────────────────────────────────────────────────────────────────────────────


def bench_linalg(device: str) -> None:
    print(f"\n--- linalg  device={device} ---")

    configs = [
        ("B=1  ", 1),
        ("B=16 ", 16),
        ("B=256", 256),
    ]

    for label, B in configs:
        T = torch.eye(4, device=device).unsqueeze(0).expand(B, -1, -1).contiguous()
        pts = torch.rand(B, 1000, 3, device=device)
        pts_flat = torch.rand(B, 1000, 4, device=device)
        lines = torch.rand(B, 1000, 3, device=device)
        x = torch.rand(B, 1000, 8, device=device)
        y = torch.rand(B, 1000, 8, device=device)

        print(f"\n  {label}")
        bench(compose_transformations, T, T, device=device, label="compose_transformations")
        bench(inverse_transformation, T, device=device, label="inverse_transformation")
        bench(relative_transformation, T, T, device=device, label="relative_transformation")
        bench(transform_points, T, pts, device=device, label="transform_points")
        bench(point_line_distance, pts_flat[..., :3], lines, device=device, label="point_line_distance")
        bench(batched_dot_product, x, y, device=device, label="batched_dot_product")
        bench(euclidean_distance, x, y, device=device, label="euclidean_distance")


def bench_epipolar(device: str) -> None:
    print(f"\n--- epipolar projection / triangulation  device={device} ---")

    configs = [
        ("B=1  N=100 ", 1, 100),
        ("B=8  N=1K  ", 8, 1000),
        ("B=32 N=1K  ", 32, 1000),
    ]

    K_base = torch.eye(3, device=device)
    K_base[0, 0] = K_base[1, 1] = 500.0
    K_base[0, 2] = K_base[1, 2] = 320.0

    for label, B, N in configs:
        K = K_base.unsqueeze(0).expand(B, -1, -1).contiguous()
        R = torch.eye(3, device=device).unsqueeze(0).expand(B, -1, -1).contiguous()
        t = torch.zeros(B, 3, 1, device=device)
        X = torch.rand(B, N, 3, device=device).add_(1.0)
        P1 = torch.cat([R, t], dim=-1)  # B x 3 x 4
        P2 = torch.cat([R, t + 0.1], dim=-1)
        pts1 = torch.rand(B, N, 2, device=device)
        pts2 = torch.rand(B, N, 2, device=device)
        F = torch.rand(B, 3, 3, device=device)
        P_proj = torch.rand(B, 3, 4, device=device)

        print(f"\n  {label}")
        bench(projection_from_KRt, K, R, t, device=device, label="projection_from_KRt")
        bench(scale_intrinsics, K, 2.0, device=device, label="scale_intrinsics")
        bench(depth_from_point, R, t, X, device=device, label="depth_from_point")
        bench(projections_from_fundamental, F, device=device, label="projections_from_fundamental")
        bench(triangulate_points, P1, P2, pts1, pts2, device=device, label="triangulate_points")
        bench(KRt_from_projection, P_proj, device=device, label="KRt_from_projection")


def bench_distort(device: str) -> None:
    print(f"\n--- calibration distortion  device={device} ---")

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
        taux = torch.zeros(B, device=device)
        tauy = torch.zeros(B, device=device)

        print(f"\n  {label}")
        bench(distort_points, pts2, K_b, dist_b, device=device, label="distort_points")
        bench(tilt_projection, taux, tauy, device=device, label="tilt_projection")


def run(device: str) -> None:
    sep = "=" * 72
    print(f"\n{sep}\n  DEVICE: {device.upper()}\n{sep}")
    bench_linalg(device)
    bench_epipolar(device)
    bench_distort(device)


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
