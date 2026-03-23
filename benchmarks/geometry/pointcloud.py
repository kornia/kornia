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
"""Benchmark for kornia.geometry point cloud operations on CPU and CUDA.

Covers:
  - transform_points
  - project_points / unproject_points
  - convert_points_to/from_homogeneous
  - depth_to_3d, depth_to_3d_v2  (uncached vs cached grid)
  - depth_to_normals
  - warp_frame_depth

Usage:
    python benchmarks/geometry/pointcloud.py            # CPU only
    python benchmarks/geometry/pointcloud.py --cuda     # CPU + CUDA
    python benchmarks/geometry/pointcloud.py --compile  # include torch.compile variants
    python benchmarks/geometry/pointcloud.py --cuda --compile
"""
from __future__ import annotations

import argparse
import datetime
import functools
import platform
import subprocess
import time

import torch

from kornia.geometry.camera import project_points, unproject_points
from kornia.geometry.conversions import convert_points_from_homogeneous, convert_points_to_homogeneous
from kornia.geometry.depth import depth_to_3d, depth_to_3d_v2, depth_to_normals, unproject_meshgrid, warp_frame_depth
from kornia.geometry.linalg import transform_points


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _sync(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()


def bench(fn, *args, warmup: int = 5, reps: int = 20, device: str = "cpu", label: str = "") -> float:
    """Return mean wall-clock time in milliseconds."""
    for _ in range(warmup):
        fn(*args)
    _sync(device)
    t0 = time.perf_counter()
    for _ in range(reps):
        fn(*args)
    _sync(device)
    ms = (time.perf_counter() - t0) / reps * 1000
    print(f"  {label:<66s}: {ms:8.3f} ms")
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
# transform_points
# ─────────────────────────────────────────────────────────────────────────────

def bench_transform_points(device: str, dtype: torch.dtype = torch.float32, compile_: bool = False) -> None:
    print(f"\n--- transform_points  device={device} dtype={dtype} ---")

    fn = transform_points
    fn_c = torch.compile(fn) if compile_ else None

    configs = [
        ("B=1   N=1K   single transform",   1,   1_000,  True),
        ("B=8   N=10K  single transform",   8,  10_000,  True),
        ("B=32  N=100K single transform",  32, 100_000,  True),
        ("B=1   N=1K   per-sample transform",  1,   1_000,  False),
        ("B=8   N=10K  per-sample transform",  8,  10_000,  False),
        ("B=32  N=100K per-sample transform", 32, 100_000,  False),
    ]

    for label, B, N, single_T in configs:
        pts = torch.randn(B, N, 3, device=device, dtype=dtype)
        T = torch.eye(4, device=device, dtype=dtype).unsqueeze(0)
        if not single_T:
            T = T.expand(B, -1, -1).contiguous()

        bench(fn, T, pts, device=device, label=label)
        if compile_:
            bench(fn_c, T, pts, device=device, label=f"{label} (compiled)")


# ─────────────────────────────────────────────────────────────────────────────
# project / unproject
# ─────────────────────────────────────────────────────────────────────────────

def bench_project_unproject(device: str, dtype: torch.dtype = torch.float32, compile_: bool = False) -> None:
    print(f"\n--- project_points / unproject_points  device={device} dtype={dtype} ---")

    K_base = torch.eye(3, device=device, dtype=dtype)
    K_base[0, 0] = K_base[1, 1] = 500.0
    K_base[0, 2] = K_base[1, 2] = 320.0

    proj_fn = project_points
    unproj_fn = unproject_points
    if compile_:
        proj_fn = torch.compile(proj_fn)
        unproj_fn = torch.compile(unproj_fn)

    configs = [
        ("B=1   N=1K  ", 1,   1_000),
        ("B=8   N=10K ", 8,  10_000),
        ("B=32  N=100K", 32, 100_000),
    ]

    for label, B, N in configs:
        K_b = K_base.unsqueeze(0).expand(B, -1, -1)
        pts3 = torch.rand(B, N, 3, device=device, dtype=dtype).add_(0.5)
        pts2 = torch.rand(B, N, 2, device=device, dtype=dtype).mul_(640.0)
        depth = torch.ones(B, N, 1, device=device, dtype=dtype)

        bench(proj_fn, pts3, K_b, device=device, label=f"{label} project_points")
        bench(unproj_fn, pts2, depth, K_b, device=device, label=f"{label} unproject_points")


# ─────────────────────────────────────────────────────────────────────────────
# homogeneous conversions
# ─────────────────────────────────────────────────────────────────────────────

def bench_homogeneous_conversions(device: str, dtype: torch.dtype = torch.float32) -> None:
    print(f"\n--- homogeneous conversions  device={device} dtype={dtype} ---")

    for label, shape, fn in [
        ("N=1M  3-D → homogeneous",         (1_000_000, 3),    convert_points_to_homogeneous),
        ("N=1M  4-D → euclidean",            (1_000_000, 4),    convert_points_from_homogeneous),
        ("B=32  N=100K  3-D → homogeneous",  (32, 100_000, 3),  convert_points_to_homogeneous),
        ("B=32  N=100K  4-D → euclidean",    (32, 100_000, 4),  convert_points_from_homogeneous),
    ]:
        x = torch.randn(*shape, device=device, dtype=dtype)
        bench(fn, x, device=device, label=label)


# ─────────────────────────────────────────────────────────────────────────────
# depth_to_3d / depth_to_3d_v2 / depth_to_normals
# ─────────────────────────────────────────────────────────────────────────────

def bench_depth_functions(device: str, dtype: torch.dtype = torch.float32, compile_: bool = False) -> None:
    print(f"\n--- depth_to_3d / depth_to_3d_v2 / depth_to_normals  device={device} dtype={dtype} ---")

    K_base = torch.eye(3, device=device, dtype=dtype)
    K_base[0, 0] = K_base[1, 1] = 500.0
    K_base[0, 2] = K_base[1, 2] = 320.0

    configs = [
        ("B=1 H=64   W=64  ",  1,  64,  64),
        ("B=1 H=256  W=256 ",  1, 256, 256),
        ("B=4 H=480  W=640 ",  4, 480, 640),
        ("B=1 H=720  W=1280",  1, 720, 1280),
    ]

    for label, B, H, W in configs:
        K_b = K_base.unsqueeze(0).expand(B, -1, -1).contiguous()
        depth4 = torch.rand(B, 1, H, W, device=device, dtype=dtype).add_(0.1)
        depth3 = depth4.squeeze(1)

        bench(depth_to_3d, depth4, K_b, device=device, label=f"{label} depth_to_3d")
        bench(depth_to_3d_v2, depth3, K_b, device=device, label=f"{label} depth_to_3d_v2 (no cache)")

        grid = unproject_meshgrid(H, W, K_b, device=device, dtype=dtype)
        bench(
            functools.partial(depth_to_3d_v2, xyz_grid=grid),
            depth3, K_b,
            device=device, label=f"{label} depth_to_3d_v2 (cached grid)",
        )

        bench(depth_to_normals, depth4, K_b, device=device, label=f"{label} depth_to_normals")

        if compile_:
            fn_c = torch.compile(depth_to_3d_v2)
            bench(
                functools.partial(fn_c, xyz_grid=grid),
                depth3, K_b,
                device=device, label=f"{label} depth_to_3d_v2 (cached + compiled)",
            )


# ─────────────────────────────────────────────────────────────────────────────
# warp_frame_depth
# ─────────────────────────────────────────────────────────────────────────────

def bench_warp_frame_depth(device: str, dtype: torch.dtype = torch.float32) -> None:
    print(f"\n--- warp_frame_depth  device={device} dtype={dtype} ---")

    K_base = torch.eye(3, device=device, dtype=dtype)
    K_base[0, 0] = K_base[1, 1] = 500.0
    K_base[0, 2] = K_base[1, 2] = 320.0

    configs = [
        ("B=1 C=3 H=256  W=256 ",  1, 3, 256, 256),
        ("B=4 C=3 H=480  W=640 ",  4, 3, 480, 640),
        ("B=1 C=3 H=720  W=1280",  1, 3, 720, 1280),
    ]

    for label, B, C, H, W in configs:
        K_b = K_base.unsqueeze(0).expand(B, -1, -1).contiguous()
        depth = torch.rand(B, 1, H, W, device=device, dtype=dtype).add_(0.1)
        image = torch.rand(B, C, H, W, device=device, dtype=dtype)
        T = torch.eye(4, device=device, dtype=dtype).unsqueeze(0).expand(B, -1, -1).contiguous()

        bench(warp_frame_depth, image, depth, T, K_b, device=device, label=label)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def run_all(device: str, compile_: bool = False) -> None:
    sep = "=" * 72
    print(f"\n{sep}")
    print(f"  DEVICE : {device.upper()}")
    print(f"  compile: {compile_}")
    print(sep)
    bench_transform_points(device, compile_=compile_)
    bench_project_unproject(device, compile_=compile_)
    bench_homogeneous_conversions(device)
    bench_depth_functions(device, compile_=compile_)
    bench_warp_frame_depth(device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--cuda", action="store_true", help="also run on CUDA")
    parser.add_argument("--compile", action="store_true", dest="compile_",
                        help="include torch.compile variants")
    args = parser.parse_args()

    _print_env()
    run_all("cpu", compile_=args.compile_)
    if args.cuda:
        if torch.cuda.is_available():
            run_all("cuda", compile_=args.compile_)
        else:
            print("\nWarning: --cuda requested but CUDA is not available.")
