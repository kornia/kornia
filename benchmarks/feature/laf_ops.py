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
There is no cross-library baseline -- no other library exposes LAFs -- so the columns are kornia
eager vs ``torch.compile``, and the baseline for a change is the same script run on another
kornia revision (see AGENTS.md, "Comparing a branch against another revision"). Running a file
under ``benchmarks/`` puts its own directory on ``sys.path[0]``, not the checkout root, so the
editable finder can resolve ``kornia`` to the primary checkout while ``git_commit()`` reports the
worktree HEAD -- an A/B that silently measures one revision twice. This script therefore puts its
own checkout root ahead of everything else on ``sys.path`` before importing kornia, prints the
resolved module path, records its checkout-relative form in the exported metadata as
``kornia_module``, and warns loudly if it still resolved outside this checkout. Read that line
before trusting a comparison.

Covered ops (all public ``kornia.feature`` API):

===============================  =============================================================
op                               why it is here
===============================  =============================================================
``laf_from_center_scale_ori``    LAF construction, runs once per detector forward
``make_upright``                 orientation reset, per detector forward
``ellipse_to_laf``               Oxford-format import; :mod:`ellipse_to_laf` drills into it
``laf_to_boundary_points``       visualization export; also drives ``laf_is_inside_image``
``laf_is_inside_image``          border filtering, per detector forward
``extract_patches_simple``       patch sampling; one folded ``grid_sample`` since #4128
``extract_patches_from_pyramid`` patch sampling; one ``grid_sample`` over a packed atlas
===============================  =============================================================

Throughput counts **LAFs per second** (``B*N`` per call) -- the README's "items" for LAF ops.
Patch extraction samples a ``(B, 1, size, size)`` float image; the pyramid variant's cost also
scales with ``min(size) // PS`` pyramid levels, so ``--size`` is part of the config, not noise.

Usage:
    python benchmarks/feature/laf_ops.py --device cpu
    python benchmarks/feature/laf_ops.py --device mps --compile
    python benchmarks/feature/laf_ops.py --device cuda --compile --json laf_ops_cuda.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Callable, Optional

import torch

# Order matters: the checkout root ends up at sys.path[0], ahead of any editable install, so the
# kornia measured is the one this file lives in. `main` verifies that it actually won.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common import (
    add_contribute_args,
    collect_load_metrics,
    contribute_result,
    print_preflight,
    run_batch_sweep,
    run_metadata,
    save_json,
    versions_line,
)

import kornia.feature as KF

Backend = Optional[Callable[[], object]]

REPO_ROOT = Path(__file__).resolve().parents[2]
PATCH_SIZE = 32


def pyramid_levels(size: int, ps: int) -> int:
    """Levels ``extract_patches_from_pyramid`` builds for a square ``size`` image at patch ``ps``.

    Mirrors its halving loop: the pyramid stops at the last level that can still provide a full
    ``ps``-sized patch.
    """
    levels, side = 1, size
    while side // 2 >= ps and side > 2:
        side //= 2
        levels += 1
    return levels


def make_lafs(b: int, n: int, size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Random LAFs: centers inside the image, any orientation, scales stratified across levels.

    Scale matters here beyond realism. ``extract_patches_from_pyramid`` selects level
    ``floor(log2(2 * get_laf_scale(laf) / PS))``, and ``get_laf_scale`` of a frame built by
    ``laf_from_center_scale_ori(scale=s)`` is ``s``, so level ``l`` covers ``s`` in
    ``[PS/2 * 2**l, PS/2 * 2**(l + 1))``. A plain 4-24 px scale range therefore puts *every* LAF
    on level 0, and a future optimization that skips unused levels would look free on that
    workload. LAFs are assigned round-robin to the levels the extractor will actually build, so
    each level carries a share of the batch. Level 0 also absorbs the small-feature regime (from
    4 px), which selects level 0 anyway; coarser levels sample uniformly inside their own octave.
    """
    xy = PATCH_SIZE + torch.rand(b, n, 2, device=device, dtype=dtype) * (size - 2 * PATCH_SIZE)
    level = torch.arange(n, device=device).remainder(pyramid_levels(size, PATCH_SIZE))
    lo = torch.where(level == 0, torch.full_like(level, 4), (PATCH_SIZE // 2) * 2**level).to(dtype)
    hi = (PATCH_SIZE * 2**level).to(dtype)
    frac = torch.rand(b, n, device=device, dtype=dtype)
    scale = (lo + (hi - lo) * frac).view(b, n, 1, 1)
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
    config: tuple[int, int], size: int, device: torch.device, dtype: torch.dtype, do_compile: bool
) -> tuple[dict[str, dict[str, Backend]], dict[str, str]]:
    """Zero-arg callables per (op, backend) for one config, plus the compile-warmup failures."""
    b, n = config
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
    if do_compile:
        # Reset dynamo once per config so every op compiles a fresh static-shape graph here;
        # without it the first config is timed on a static graph and later ones on the
        # automatic-dynamic recompile. Mirrors ellipse_to_laf.py. The reset belongs outside the
        # op loop: it invalidates every compiled callable, so resetting per op would push the
        # earlier ops' recompiles into the timed region instead of this warmup.
        torch._dynamo.reset()
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", default="cpu", help="cpu, cuda, or mps")
    parser.add_argument("--dtype", default="float32", choices=["float16", "bfloat16", "float32", "float64"])
    parser.add_argument("--configs", default="1x2000,1x20000,8x2000", help="comma-separated BxN pairs")
    parser.add_argument("--size", type=int, default=256, help="square image side for the patch-extraction ops")
    parser.add_argument("--compile", action="store_true", help="add a torch.compile column")
    parser.add_argument("--min-run-time", type=float, default=1.0, help="seconds of repeats per measurement")
    parser.add_argument("--json", type=str, default=None, help="write results to this path as JSON")
    add_contribute_args(parser)
    args = parser.parse_args()

    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)
    configs = [(int(b), int(n)) for b, n in (c.split("x") for c in args.configs.split(","))]
    sync = torch.mps.synchronize if device.type == "mps" else None  # blocked_autorange syncs CUDA itself

    meta = run_metadata(device)
    meta["load"] = collect_load_metrics()
    kornia_file = sys.modules["kornia"].__file__
    resolved = Path(kornia_file).resolve() if kornia_file is not None else None
    inside = resolved is not None and REPO_ROOT in resolved.parents
    print_preflight(meta["load"])
    print(f"# laf_ops | commit {meta['git_commit']} | {meta['platform']} | {device} | {args.dtype}")
    print(f"# kornia module: {kornia_file}")
    if not inside:
        # git_commit() reports this checkout's HEAD, so a kornia from anywhere else means the
        # numbers and the commit label describe different code.
        print(f"# WARNING: kornia resolved outside {REPO_ROOT} - these numbers are NOT this checkout's.")
    print(versions_line(meta))

    backends = ["kornia (eager)"] + (["kornia (compiled)"] if args.compile else [])

    def row_fields(config: tuple[int, int]) -> dict[str, Any]:
        b, n = config
        return {"batch": b, "n_lafs": n, "size": args.size, "dtype": args.dtype}

    results = run_batch_sweep(
        configs,
        lambda config: build_ops(config, args.size, device, dtype, args.compile),
        backends,
        row_fields=row_fields,
        sync=sync,
        label_fn=lambda c: f"B={c[0]} N={c[1]}",
        items_fn=lambda c: c[0] * c[1],
        units="LAFs/s",
        label_width=30,
        col_width=16,
        min_run_time=args.min_run_time,
    )

    # The exported form is checkout-relative. The absolute path above answers "which tree?" on
    # the console, where it is needed; a contributed file is public, and README's privacy rule
    # keeps home directories and machine layout out of it. Relative or "outside-checkout" still
    # carries the only bit a reader of the file needs: did kornia resolve to the measured tree?
    module_field = resolved.relative_to(REPO_ROOT).as_posix() if inside else "outside-checkout"
    # The docs page and the llms digest label the throughput column from this key; without it a
    # committed run reads as "items/s", and the item here is a LAF, not an image.
    export_meta = {**meta, "kornia_module": module_field, "units": "LAFs/s"}
    if args.json:
        out = save_json(args.json, export_meta, results)
        print(f"# wrote {out}")
    if args.contribute:
        contribute_result(args.contribute, "feature-laf-ops", export_meta, results, args.machine_slug)


if __name__ == "__main__":
    main()
