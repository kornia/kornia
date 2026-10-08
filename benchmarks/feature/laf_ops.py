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
``kornia_module``, and warns loudly if it still resolved outside this checkout (all suites share
this via ``common.start_run``). Read that line before trusting a comparison.

Covered ops (all public ``kornia.feature`` API):

================================  ==========================================================
op                                why it is here
================================  ==========================================================
``laf_from_center_scale_ori``     LAF construction, runs once per detector forward
``make_upright``                  orientation reset, per detector forward
``ellipse_to_laf``                Oxford-format import; :mod:`ellipse_to_laf` drills into it
``laf_to_boundary_points``        visualization export; also drives ``laf_is_inside_image``
``laf_is_inside_image``           border filtering, per detector forward
``extract_patches_simple``        patch sampling; one folded ``grid_sample`` since #4128
``extract_patches_from_pyramid``  patch sampling; one ``grid_sample`` over a packed atlas
================================  ==========================================================

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
from typing import Any, Callable

import torch

# Order matters: the checkout root ends up at sys.path[0], ahead of any editable install, so the
# kornia measured is the one this file lives in. `main` verifies that it actually won.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common import Backend, KorniaRows, add_flagship_args, finish_run, run_batch_sweep, setup_run, start_run

import kornia.feature as KF

PATCH_SIZE = 32
OPS = (
    "laf_from_center_scale_ori",
    "make_upright",
    "ellipse_to_laf",
    "laf_to_boundary_points",
    "laf_is_inside_image",
    "extract_patches_simple",
    "extract_patches_from_pyramid",
)


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
    config: tuple[int, int],
    size: int,
    device: torch.device,
    dtype: torch.dtype,
    do_compile: bool,
    skip_compile: frozenset[str] = frozenset(),
    selected: frozenset[str] | None = None,
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

    # Reset dynamo once per config so every op compiles a fresh static-shape graph here;
    # without it the first config is timed on a static graph and later ones on the
    # automatic-dynamic recompile. Mirrors ellipse_to_laf.py. The reset belongs outside the
    # op loop: it invalidates every compiled callable, so resetting per op would push the
    # earlier ops' recompiles into the timed region instead of this warmup. These are distinct
    # functions, so the recompile limit that forces per-op resets on the image suites does not bite.
    kornia_row = KorniaRows(device, do_compile, skip_compile, reset_per_op=False)
    ops = {name: kornia_row(name, fn, *args) for name, fn, args in cases if selected is None or name in selected}
    return ops, kornia_row.compile_failures


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_flagship_args(parser, ops=OPS, batches="")
    parser.add_argument("--configs", default="1x2000,1x20000,8x2000", help="comma-separated BxN pairs")
    args = parser.parse_args()
    device, dtype, sync = setup_run(args, opencv=False)
    configs = [(int(b), int(n)) for b, n in (c.split("x") for c in args.configs.split(","))]

    meta = start_run(
        "feature LAF ops",
        args,
        device,
        units="LAFs/s",
        regimes=["kornia only: no other library exposes LAFs; compare against another kornia revision"],
    )
    backends = ["kornia (eager)"] + (["kornia (compiled)"] if args.compile else [])

    def row_fields(config: tuple[int, int]) -> dict[str, Any]:
        b, n = config
        return {"batch": b, "n_lafs": n, "height": args.size, "width": args.size, "dtype": args.dtype}

    results = run_batch_sweep(
        configs,
        lambda config: build_ops(config, args.size, device, dtype, args.compile, args.skip_compile_ops, args.ops),
        backends,
        row_fields=row_fields,
        sync=sync,
        label_fn=lambda c: f"B={c[0]} N={c[1]}",
        items_fn=lambda c: c[0] * c[1],
        units="LAFs/s",
        min_run_time=args.min_run_time,
    )
    finish_run(args, "feature-laf-ops", meta, results)


if __name__ == "__main__":
    main()
