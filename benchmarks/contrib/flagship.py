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

"""Flagship contrib benchmark: kornia.contrib binary-image ops vs OpenCV, scikit-image and SciPy.

``kornia.contrib`` mixes model wrappers (detectors, super-resolution, ViTs) that need downloaded
weights with classic binary-image operations. This suite covers the latter; the dedicated
``connected_components.py`` script compares the two labeling algorithms in depth.

====================================  ===================================  ==========================================
kornia.contrib                        OpenCV (uint8 per-image loop)        scikit-image / SciPy (per-image loop)
====================================  ===================================  ==========================================
``connected_components_union_find``   ``connectedComponents(conn=8)``      ``measure.label(connectivity=2)``;
                                                                           ``ndimage.label`` (3x3 structure)
``distance_transform``                ``distanceTransform(DIST_L2, 5)``    ``ndimage.distance_transform_edt``
====================================  ===================================  ==========================================

Regimes (see ``benchmarks/README.md``): kornia runs a batched float B1HW mask on CPU or GPU;
OpenCV, scikit-image and SciPy run single masks on CPU in a Python loop, their native regime. The
mask is thresholded Gaussian-blurred noise (about 30% foreground, blobs of varying size and
shape), the same for every backend. Connected components are exact 8-connected labelings in all
three columns; kornia's default ``connected_components`` is left out because its fixed pooling
budget does not guarantee a correct partition (see ``connected_components.md``). The distance
transforms differ in accuracy: kornia's is a cascaded-convolution approximation
(``kernel_size=3``), OpenCV's a 5x5 chamfer approximation, and SciPy's exact. kornia measures the
distance to the nearest foreground pixel, OpenCV and SciPy to the nearest zero, so the latter two
receive the inverted mask and all three answer the same question. The labelers get the mask in
their fastest input type, converted outside the timed call: ``uint8`` for OpenCV and ``bool`` for
scikit-image and SciPy (scikit-image's ``label`` takes a much faster path on ``bool`` and returns
the same labels). Throughput is img/s.

Usage:
    python benchmarks/contrib/flagship.py --batches 1,8,32 --size 256 --device cpu
    python benchmarks/contrib/flagship.py --device cuda --compile --json contrib_cuda.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import ModuleType
from typing import Optional

import numpy as np
import torch

# Prefer this checkout to an installed wheel or another editable checkout.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import (
    Backend,
    KorniaRows,
    add_flagship_args,
    batch_list,
    finish_run,
    image_row_fields,
    optional_import,
    run_batch_sweep,
    setup_run,
    start_run,
)

import kornia.contrib as KC
from kornia.filters import gaussian_blur2d

OPS = ("connected_components_union_find", "distance_transform")


def blob_masks(b: int, h: int, w: int) -> torch.Tensor:
    """Seeded ``(B, 1, H, W)`` float {0, 1} masks: blurred noise above its 70th percentile."""
    gen = torch.Generator().manual_seed(0)
    smooth = gaussian_blur2d(torch.rand(b, 1, h, w, generator=gen), (13, 13), (3.0, 3.0))
    threshold = smooth.flatten(1).quantile(0.7, dim=1).view(b, 1, 1, 1)
    return (smooth > threshold).float()


def build_ops(
    b: int,
    h: int,
    w: int,
    device: torch.device,
    dtype: torch.dtype,
    do_compile: bool,
    cv2: Optional[ModuleType],
    skmeasure: Optional[ModuleType],
    ndimage: Optional[ModuleType],
    skip_compile: frozenset[str] = frozenset(),
    selected: Optional[frozenset[str]] = None,
) -> tuple[dict[str, dict[str, Backend]], dict[str, str]]:
    """Build {op: {backend: zero-arg callable}}; each callable processes the whole batch once."""
    masks = blob_masks(b, h, w)
    masks_f = masks.to(device=device, dtype=dtype)
    masks_u8 = [m[0].numpy().astype(np.uint8) for m in masks]
    masks_bool = [m.astype(bool) for m in masks_u8]
    background_u8 = [1 - m for m in masks_u8]
    connectivity8 = np.ones((3, 3), dtype=bool)

    kornia_row = KorniaRows(device, do_compile, skip_compile)
    ops: dict[str, dict[str, Backend]] = {}

    def include(name: str) -> bool:
        return selected is None or name in selected

    if include("connected_components_union_find"):
        row = kornia_row("connected_components_union_find", KC.connected_components_union_find, masks_f)
        row["opencv"] = (lambda: [cv2.connectedComponents(m, connectivity=8) for m in masks_u8]) if cv2 else None
        row["scikit-image"] = (lambda: [skmeasure.label(m, connectivity=2) for m in masks_bool]) if skmeasure else None
        row["SciPy"] = (lambda: [ndimage.label(m, structure=connectivity8) for m in masks_bool]) if ndimage else None
        ops["connected_components_union_find"] = row

    if include("distance_transform"):
        row = kornia_row("distance_transform", KC.distance_transform, masks_f)
        row["opencv"] = (lambda: [cv2.distanceTransform(m, cv2.DIST_L2, 5) for m in background_u8]) if cv2 else None
        row["SciPy"] = (lambda: [ndimage.distance_transform_edt(m) for m in background_u8]) if ndimage else None
        ops["distance_transform"] = row

    return ops, kornia_row.compile_failures


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_flagship_args(parser, ops=OPS)
    args = parser.parse_args()
    device, dtype, sync = setup_run(args)

    cv2, cv2_error = optional_import("cv2")
    skmeasure, skmeasure_error = optional_import("skimage.measure")
    ndimage, ndimage_error = optional_import("scipy.ndimage")

    libs = [(skmeasure, "scikit-image", skmeasure_error), (ndimage, "SciPy", ndimage_error), (cv2, "opencv", cv2_error)]
    meta = start_run(
        "flagship contrib",
        args,
        device,
        units="img/s",
        regimes=[
            "kornia: batched float B1HW mask; opencv/scikit-image/SciPy: per-image loop (CPU), "
            "uint8 masks except bool for scikit-image and SciPy labeling"
        ],
        missing=[(name, err) for lib, name, err in libs if lib is None],
    )
    backends = ["kornia (eager)", "kornia (compiled)", "scikit-image", "SciPy", "opencv"]
    results = run_batch_sweep(
        batch_list(args),
        lambda b: build_ops(
            b,
            args.size,
            args.size,
            device,
            dtype,
            args.compile,
            cv2,
            skmeasure,
            ndimage,
            args.skip_compile_ops,
            args.ops,
        ),
        backends,
        row_fields=image_row_fields(args),
        sync=sync,
        units="img/s",
        min_run_time=args.min_run_time,
    )
    finish_run(args, "contrib", meta, results)


if __name__ == "__main__":
    main()
