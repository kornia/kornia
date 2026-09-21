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

"""Flagship geometry-op benchmark: kornia vs OpenCV vs torchvision v2.

Covers the differentiated-core ops named by the W3 benchmark plan:

=============================  ==========================================  ==============
kornia (batched float BCHW)    OpenCV (uint8 HWC, per-image Python loop)   torchvision v2
=============================  ==========================================  ==============
``warp_perspective``           ``cv2.warpPerspective``                     —
``warp_affine``                ``cv2.warpAffine``                          —
``rotate``                     ``cv2.warpAffine(getRotationMatrix2D)``     ``tvf.rotate``
``resize``                     ``cv2.resize``                              ``tvf.resize``
``get_perspective_transform``  ``cv2.getPerspectiveTransform`` (per pair)  —
=============================  ==========================================  ==============

Regimes (same framing as ``benchmarks/augmentation``): kornia/torchvision run a batched float
tensor on CPU or GPU and kornia is differentiable; OpenCV runs single uint8 images on CPU in
a Python loop — its native regime. Columns are regime comparisons, not apples-to-apples.

Equal footing: bilinear interpolation, no antialiasing anywhere (torchvision gets an explicit
``antialias=False``), identical transform parameters across backends, pinned seeds.

Usage:
    python benchmarks/geometry/flagship.py --batches 1,8,32 --size 256 --device cpu
    python benchmarks/geometry/flagship.py --device cuda --compile --json flagship_cuda.json
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
    image_batch,
    image_row_fields,
    optional_import,
    run_batch_sweep,
    setup_run,
    start_run,
)

import kornia.geometry as KG


def build_ops(
    b: int,
    h: int,
    w: int,
    device: torch.device,
    dtype: torch.dtype,
    do_compile: bool,
    cv2: Optional[ModuleType],
    tvf: Optional[ModuleType],
    skip_compile: frozenset[str] = frozenset(),
    selected: Optional[frozenset[str]] = None,
) -> tuple[dict[str, dict[str, Backend]], dict[str, str]]:
    """Build {op: {backend: zero-arg callable}} with identical transform params per backend.

    Also returns {op: exception name} for ops whose ``torch.compile`` warmup failed, so the
    caller can report them instead of leaving a silent skip cell.
    """
    imgs_u8, batch_f = image_batch(b, h, w, device, dtype)

    angle_deg = 30.0
    angle = torch.full((b,), angle_deg, device=device, dtype=dtype)
    center = torch.tensor([[w / 2, h / 2]], dtype=torch.float32).expand(b, 2).to(device=device, dtype=dtype)
    scale = torch.ones(b, 2, device=device, dtype=dtype)
    m_affine = KG.get_rotation_matrix2d(center, angle, scale)  # (B, 2, 3)

    quad = torch.tensor([[[0.0, 0.0], [w - 1.0, 0.0], [w - 1.0, h - 1.0], [0.0, h - 1.0]]], dtype=torch.float32)
    src_pts32 = quad.expand(b, 4, 2).contiguous()
    gen = torch.Generator().manual_seed(0)
    dst_pts32 = src_pts32 + 8.0 * torch.randn(b, 4, 2, generator=gen)
    src_pts = src_pts32.to(device=device, dtype=dtype)
    dst_pts = dst_pts32.to(device=device, dtype=dtype)
    h_mat = KG.get_perspective_transform(src_pts, dst_pts)  # (B, 3, 3)

    m_np = m_affine.float().cpu().numpy()
    h_np = h_mat.float().cpu().numpy().astype(np.float64)
    src_np, dst_np = src_pts32.numpy(), dst_pts32.numpy()

    kornia_row = KorniaRows(device, do_compile, skip_compile)
    ops: dict[str, dict[str, Backend]] = {}

    def include(name: str) -> bool:
        return selected is None or name in selected

    if include("warp_perspective"):
        row = kornia_row("warp_perspective", lambda: KG.warp_perspective(batch_f, h_mat, (h, w)))
        row["opencv"] = (
            (lambda: [cv2.warpPerspective(im, h_np[i], (w, h)) for i, im in enumerate(imgs_u8)]) if cv2 else None
        )
        ops["warp_perspective"] = row

    if include("warp_affine"):
        row = kornia_row("warp_affine", lambda: KG.warp_affine(batch_f, m_affine, (h, w)))
        row["opencv"] = (lambda: [cv2.warpAffine(im, m_np[i], (w, h)) for i, im in enumerate(imgs_u8)]) if cv2 else None
        ops["warp_affine"] = row

    if include("rotate"):
        row = kornia_row("rotate", lambda: KG.rotate(batch_f, angle))
        if cv2:
            m_rot = cv2.getRotationMatrix2D((w / 2, h / 2), angle_deg, 1.0)
            row["opencv"] = lambda: [cv2.warpAffine(im, m_rot, (w, h)) for im in imgs_u8]
        row["torchvision v2"] = (lambda: tvf.rotate(batch_f, angle_deg)) if tvf else None
        ops["rotate"] = row

    if include("resize"):
        dst_size = (h // 2, w // 2)
        row = kornia_row("resize", lambda: KG.resize(batch_f, dst_size, interpolation="bilinear"))
        row["opencv"] = (lambda: [cv2.resize(im, (dst_size[1], dst_size[0])) for im in imgs_u8]) if cv2 else None
        row["torchvision v2"] = (lambda: tvf.resize(batch_f, list(dst_size), antialias=False)) if tvf else None
        ops["resize"] = row

    if include("get_perspective_transform"):
        row = kornia_row("get_perspective_transform", lambda: KG.get_perspective_transform(src_pts, dst_pts))
        row["opencv"] = (lambda: [cv2.getPerspectiveTransform(src_np[i], dst_np[i]) for i in range(b)]) if cv2 else None
        ops["get_perspective_transform"] = row
    return ops, kornia_row.compile_failures


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_flagship_args(parser)
    args = parser.parse_args()
    device, dtype, sync = setup_run(args)

    cv2, cv2_error = optional_import("cv2")
    tvf, tvf_error = optional_import("torchvision.transforms.v2.functional")

    meta = start_run(
        "flagship geometry",
        args,
        device,
        units="items/s",
        regimes=["kornia/torchvision: batched float BCHW; opencv: uint8 HWC per-image loop (CPU)"],
        missing=[
            (name, err)
            for lib, name, err in [(cv2, "opencv", cv2_error), (tvf, "torchvision", tvf_error)]
            if lib is None
        ],
    )
    backends = ["kornia (eager)", "kornia (compiled)", "torchvision v2", "opencv"]
    results = run_batch_sweep(
        batch_list(args),
        lambda b: build_ops(
            b, args.size, args.size, device, dtype, args.compile, cv2, tvf, args.skip_compile_ops, args.ops
        ),
        backends,
        row_fields=image_row_fields(args),
        sync=sync,
        units="items/s",
        min_run_time=args.min_run_time,
    )
    finish_run(args, "geometry", meta, results)


if __name__ == "__main__":
    main()
