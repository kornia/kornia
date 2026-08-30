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

"""Microbenchmarks for the LAF operations in ``kornia.feature``.

These ops are the shared substrate of every kornia local-feature pipeline: each detector call
converts, validates, and normalizes LAFs, and each descriptor call extracts patches from them.
There is no cross-library baseline — no other library exposes LAFs — so the columns are kornia
eager vs ``torch.compile`` only, and the numbers exist to rank kornia's own hot spots and to
hold before/after evidence for optimization PRs.

Covered ops (all public ``kornia.feature`` API):

=============================   =============================================================
op                              why it is here
=============================   =============================================================
``laf_from_center_scale_ori``   LAF construction, runs once per detector forward
``make_upright``                orientation reset, per detector forward
``ellipse_to_laf``              Oxford-format import; known hot spot (batched 2x2 ``inverse``)
``laf_to_boundary_points``      visualization export; builds its basis on CPU every call
``laf_is_inside_image``         border filtering, per detector forward
``extract_patches_simple``      patch sampling; Python loop over the batch dimension
``extract_patches_from_pyramid``  patch sampling; batch loop x full grid_sample per level
=============================   =============================================================

Throughput counts **LAFs per second** (``B*N`` per call) — the README's "items" for LAF ops.
Patch extraction samples a ``(B, 1, size, size)`` float image; the pyramid variant's cost also
scales with ``min(size) // PS`` pyramid levels, so ``--size`` is part of the config, not noise.

Usage:
    python benchmarks/feature/laf_ops.py --device cpu
    python benchmarks/feature/laf_ops.py --device mps --compile
    python benchmarks/feature/laf_ops.py --device cuda --compile --json laf_ops_cuda.json
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Callable, Optional

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import (
    add_contribute_args,
    collect_load_metrics,
    contribute_result,
    print_preflight,
    run_metadata,
    save_json,
    time_us,
    versions_line,
)

import kornia.feature as KF

Backend = Optional[Callable[[], object]]

PATCH_SIZE = 32


def make_lafs(b: int, n: int, size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Random but realistic LAFs: centers inside the image, scales 4-24 px, any orientation."""
    xy = PATCH_SIZE + torch.rand(b, n, 2, device=device, dtype=dtype) * (size - 2 * PATCH_SIZE)
    scale = 4.0 + 20.0 * torch.rand(b, n, 1, 1, device=device, dtype=dtype)
    ori = 360.0 * torch.rand(b, n, 1, device=device, dtype=dtype) - 180.0
    return KF.laf_from_center_scale_ori(xy, scale, ori)


def make_ellipses(b: int, n: int, size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Random Oxford-format ellipses ``[x y a b c]`` with a positive-definite ``[a b; b c]``."""
    xy = PATCH_SIZE + torch.rand(b, n, 2, device=device, dtype=dtype) * (size - 2 * PATCH_SIZE)
    sx = 4.0 + 20.0 * torch.rand(b, n, device=device, dtype=dtype)
    sy = 4.0 + 20.0 * torch.rand(b, n, device=device, dtype=dtype)
    a = 1.0 / (sx * sx)
    c = 1.0 / (sy * sy)
    rho = torch.rand(b, n, device=device, dtype=dtype) - 0.5  # |b| < sqrt(a*c)/2 keeps it positive-definite
    bb = rho * (a * c).sqrt()
    return torch.cat([xy, torch.stack([a, bb, c], dim=-1)], dim=-1)


def build_ops(
    b: int, n: int, size: int, device: torch.device, dtype: torch.dtype, do_compile: bool
) -> tuple[dict[str, dict[str, Backend]], dict[str, str]]:
    """Zero-arg callables per (op, backend) for one config, plus the compile-warmup failures."""
    torch.manual_seed(0)
    lafs = make_lafs(b, n, size, device, dtype)
    ells = make_ellipses(b, n, size, device, dtype)
    img = torch.rand(b, 1, size, size, device=device, dtype=dtype)
    xy = PATCH_SIZE + torch.rand(b, n, 2, device=device, dtype=dtype) * (size - 2 * PATCH_SIZE)
    scale = 4.0 + 20.0 * torch.rand(b, n, 1, 1, device=device, dtype=dtype)
    ori = 360.0 * torch.rand(b, n, 1, device=device, dtype=dtype) - 180.0

    cases: list[tuple[str, Callable[..., object], tuple[object, ...]]] = [
        ("laf_from_center_scale_ori", KF.laf_from_center_scale_ori, (xy, scale, ori)),
        ("make_upright", KF.make_upright, (lafs,)),
        ("ellipse_to_laf", KF.ellipse_to_laf, (ells,)),
        ("laf_to_boundary_points", KF.laf_to_boundary_points, (lafs,)),
        ("laf_is_inside_image", KF.laf_is_inside_image, (lafs, img)),
        ("extract_patches_simple", KF.extract_patches_simple, (img, lafs, PATCH_SIZE)),
        ("extract_patches_from_pyramid", KF.extract_patches_from_pyramid, (img, lafs, PATCH_SIZE)),
    ]

    ops: dict[str, dict[str, Backend]] = {}
    compile_failures: dict[str, str] = {}
    for name, fn, args in cases:
        row: dict[str, Backend] = {"kornia (eager)": (lambda fn=fn, args=args: fn(*args))}
        if do_compile:
            try:
                cfn = torch.compile(fn)
                cfn(*args)  # warmup: compilation happens on the first call
                row["kornia (compiled)"] = lambda cfn=cfn, args=args: cfn(*args)
            except Exception as exc:
                compile_failures[name] = type(exc).__name__
                row["kornia (compiled)"] = None
        ops[name] = row
    return ops, compile_failures


def run_sweep(
    configs: list[tuple[int, int]],
    size: int,
    device: torch.device,
    dtype: torch.dtype,
    do_compile: bool,
    min_run_time: float,
) -> list[dict[str, object]]:
    backends = ["kornia (eager)"] + (["kornia (compiled)"] if do_compile else [])
    sync = torch.mps.synchronize if device.type == "mps" else None
    label_width, col_width = 30, 16
    results: list[dict[str, object]] = []
    header = ""
    for b, n in configs:
        ops, compile_failures = build_ops(b, n, size, device, dtype, do_compile)
        if compile_failures:
            exc_names = sorted(set(compile_failures.values()))
            print(f"# NOTE: torch.compile warmup failed ({', '.join(exc_names)}) for: {', '.join(compile_failures)}")
        header = f"{f'B={b} N={n}':<{label_width}}" + "".join(f"{be[:col_width]:>{col_width + 1}}" for be in backends)
        print("-" * len(header))
        print(header + "   (LAFs/s)")
        print("-" * len(header))
        for op_name, row in ops.items():
            cells = []
            for backend in backends:
                fn = row.get(backend)
                if fn is None:
                    cells.append(f"{'-':>{col_width + 1}}")
                    continue
                median, iqr = time_us(fn, min_run_time=min_run_time, sync=sync)
                thr = (b * n) / (median * 1e-6) if not math.isnan(median) else float("nan")
                results.append(
                    {
                        "op": op_name,
                        "backend": backend,
                        "batch": b,
                        "n_lafs": n,
                        "size": size,
                        "dtype": str(dtype).replace("torch.", ""),
                        "median_us": median,
                        "iqr_us": iqr,
                        "throughput_per_s": thr,
                    }
                )
                cells.append(f"{thr:>{col_width + 1}.0f}")
            print(f"{op_name:<{label_width}}" + "".join(cells))
    if header:
        print("-" * len(header))
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", default="cpu", help="cpu, cuda, or mps")
    parser.add_argument("--configs", default="1x2000,1x20000,8x2000", help="comma-separated BxN pairs")
    parser.add_argument("--size", type=int, default=256, help="square image side for the patch-extraction ops")
    parser.add_argument("--compile", action="store_true", help="add a torch.compile column")
    parser.add_argument("--min-run-time", type=float, default=1.0, help="seconds of repeats per measurement")
    parser.add_argument("--json", type=str, default=None, help="write results to this path as JSON")
    add_contribute_args(parser)
    args = parser.parse_args()

    device = torch.device(args.device)
    configs = [(int(b), int(n)) for b, n in (c.split("x") for c in args.configs.split(","))]

    meta = run_metadata(device)
    meta["load"] = collect_load_metrics()
    print_preflight(meta["load"])
    print(f"# laf_ops | commit {meta['git_commit']} | {meta['platform']} | device {device}")
    print(versions_line(meta))

    results = run_sweep(configs, args.size, device, torch.float32, args.compile, args.min_run_time)

    if args.json:
        save_json(args.json, meta, results)
        print(f"# wrote {args.json}")
    if args.contribute:
        contribute_result(args.contribute, "feature-laf-ops", meta, results, args.machine_slug)


if __name__ == "__main__":
    main()
