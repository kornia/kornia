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
"""Microbenchmark for ``kornia.feature.ellipse_to_laf``.

No other library exposes this conversion, so there is no cross-library column: the baseline is
the same script run on another kornia revision (see AGENTS.md, "Comparing a branch against
another revision" -- run from a worktree root so the worktree shadows the editable install; the
header prints the kornia module path so a mismeasured revision is visible in the output).

Input regime: a batch of well-conditioned Oxford-format ellipses ``(1, N, 5)`` in the given
dtype, pinned seed. Throughput counts ellipses per second. The compiled column uses
``torch.compile(fullgraph=True)``; a warmup failure is reported as a NOTE, never silently
skipped.

Usage:
    python benchmarks/feature/ellipse_to_laf.py --device cpu
    python benchmarks/feature/ellipse_to_laf.py --device cuda --compile --json ellipse_cuda.json
    python benchmarks/feature/ellipse_to_laf.py --device mps --compile --dtype float32
"""

from __future__ import annotations

import argparse
import platform
import sys
from pathlib import Path
from typing import Callable, Optional

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import run_batch_sweep, run_metadata, save_json, versions_line

import kornia.feature as KF

Backend = Optional[Callable[[], object]]


def make_ellipses(n: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Well-conditioned Oxford-format ellipses ``(1, N, 5)``: [x, y, a, b, c], positive definite."""
    torch.manual_seed(0)
    ells = torch.rand(1, n, 5, device=device, dtype=dtype)
    ells[..., 2] = ells[..., 3] + 0.3
    ells[..., 4] += 1.0
    return ells


def build_ops(
    n: int, device: torch.device, dtype: torch.dtype, do_compile: bool
) -> tuple[dict[str, dict[str, Backend]], dict[str, str]]:
    ells = make_ellipses(n, device, dtype)
    row: dict[str, Backend] = {"kornia (eager)": lambda: KF.ellipse_to_laf(ells)}
    compile_failures: dict[str, str] = {}
    if do_compile:
        compiled = torch.compile(KF.ellipse_to_laf, fullgraph=True)
        try:
            compiled(ells)
            row["kornia (compiled)"] = lambda: compiled(ells)
        except Exception as e:
            compile_failures["ellipse_to_laf"] = type(e).__name__
    return {"ellipse_to_laf": row}, compile_failures


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default="float32", choices=["float16", "bfloat16", "float32", "float64"])
    parser.add_argument("--sizes", default="1000,20000,100000", help="comma-separated N (ellipses per call)")
    parser.add_argument("--compile", action="store_true", help="add a torch.compile(fullgraph=True) column")
    parser.add_argument("--json", default=None, help="write results to this path as strict JSON")
    args = parser.parse_args()

    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)
    sizes = [int(s) for s in args.sizes.split(",")]
    sync = torch.mps.synchronize if device.type == "mps" else None  # blocked_autorange syncs CUDA itself

    meta = run_metadata(device)
    kornia_file = sys.modules["kornia"].__file__
    print(f"# ellipse_to_laf | commit {meta['git_commit']} | {platform.platform()} | {device} | {args.dtype}")
    print(f"# kornia module: {kornia_file}")
    print(versions_line(meta))

    backends = ["kornia (eager)"] + (["kornia (compiled)"] if args.compile else [])
    results = run_batch_sweep(
        sizes,
        lambda n: build_ops(n, device, dtype, args.compile),
        backends,
        row_fields=lambda n: {"n": n, "dtype": args.dtype},
        sync=sync,
    )
    if args.json:
        out = save_json(args.json, {**meta, "kornia_module": kornia_file}, results)
        print(f"# wrote {out}")


if __name__ == "__main__":
    main()
