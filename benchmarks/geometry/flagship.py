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

"""Flagship geometry-op benchmark: kornia vs OpenCV, torchvision v2 and kornia-rs.

Covers image warps, point transforms and depth unprojection:

=============================  ==========================================  ==============
kornia (batched float BCHW)    OpenCV (uint8 HWC, per-image Python loop)   torchvision v2
=============================  ==========================================  ==============
``warp_perspective``           ``cv2.warpPerspective``                     —
``warp_affine``                ``cv2.warpAffine``                          —
``rotate``                     ``cv2.warpAffine`` (prepared matrix)       ``tvf.rotate``
``rotate_shared``              ``cv2.warpAffine`` (prepared matrix)       ``tvf.rotate``
``resize``                     ``cv2.resize``                              ``tvf.resize``
``get_perspective_transform``  ``cv2.getPerspectiveTransform`` (per pair)  —
``remap``                      ``cv2.remap`` (float32)                    —
``transform_points``           ``cv2.perspectiveTransform`` (float32)     —
``depth_to_3d``                —                                         —
=============================  ==========================================  ==============

Regimes (same framing as ``benchmarks/augmentation``): kornia/torchvision run a batched float
tensor on CPU or GPU and kornia is differentiable; OpenCV runs single uint8 images on CPU in
a Python loop — its native regime. kornia-rs runs uint8 HWC CPU image loops for resize,
affine/perspective warps and both rotate rows (via the same precomputed rotation matrix).
Columns are regime comparisons, not apples-to-apples. Rust's bilinear warps zero samples with
source coordinates outside [0,W)x[0,H), but clamp the second neighbor inside that domain;
Kornia/OpenCV instead blend with zero padding at the border. Rust also quantizes interpolation.
These edge differences are retained, not corrected with extra work in the timed call.

Equal footing: bilinear interpolation, no antialiasing anywhere (torchvision gets an explicit
``antialias=False``), identical transform parameters across backends, pinned seeds.
``rotate`` passes B angle parameters; ``rotate_shared`` passes a singleton angle so Kornia can
reuse one sampling grid. Both use 30 degrees, matching torchvision's shared-angle API.
``remap`` uses prepared float32 images and pixel maps offset by (+0.25, -0.5), bilinear
interpolation and zero padding, with ``align_corners=True`` in Kornia. These offsets are exact
on OpenCV's 1/32 interpolation table; arbitrary offsets can differ. ``transform_points`` applies
a fixed projective matrix to H*W seeded 2D points per batch item; OpenCV gets float32 point sets.
``depth_to_3d`` unprojects positive z-depth [0.5, 4.5] with focal length W and centered principal
point, with no normalization or external baseline. One item means one image, depth map, point
set (H*W points), or four-correspondence solve, depending on the row.

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
    kornia_rs_fn,
    optional_import,
    run_batch_sweep,
    setup_run,
    start_run,
)

import kornia.geometry as KG

OPS = (
    "warp_perspective",
    "warp_affine",
    "rotate",
    "rotate_shared",
    "resize",
    "get_perspective_transform",
    "remap",
    "transform_points",
    "depth_to_3d",
)

RS_OPS = {
    "warp_perspective": "warp_perspective",
    "warp_affine": "warp_affine",
    "rotate": "warp_affine",
    "rotate_shared": "warp_affine",
    "resize": "resize",
}


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
    # Python sequences are required by the Rust binding; prepare them before timing.
    m_rs = m_np.reshape(b, 6).tolist()
    h_rs = h_np.reshape(b, 9).tolist()
    rotation_center = torch.tensor([[(w - 1) / 2, (h - 1) / 2]], device=device, dtype=dtype)
    m_rot = KG.get_rotation_matrix2d(rotation_center, angle[:1], scale[:1])[0].float().cpu().numpy()
    m_rot_rs = m_rot.reshape(6).tolist()
    rs_affine = kornia_rs_fn("warp_affine")
    rs_perspective = kornia_rs_fn("warp_perspective")
    rs_resize = kornia_rs_fn("resize")

    kornia_row = KorniaRows(device, do_compile, skip_compile)
    ops: dict[str, dict[str, Backend]] = {}

    def include(name: str) -> bool:
        return selected is None or name in selected

    if include("warp_perspective"):
        row = kornia_row("warp_perspective", lambda: KG.warp_perspective(batch_f, h_mat, (h, w)))
        row["opencv"] = (
            (lambda: [cv2.warpPerspective(im, h_np[i], (w, h)) for i, im in enumerate(imgs_u8)]) if cv2 else None
        )
        row["kornia-rs"] = (
            (lambda: [rs_perspective(im, h_rs[i], (h, w), "bilinear") for i, im in enumerate(imgs_u8)])
            if rs_perspective
            else None
        )
        ops["warp_perspective"] = row

    if include("warp_affine"):
        row = kornia_row("warp_affine", lambda: KG.warp_affine(batch_f, m_affine, (h, w)))
        row["opencv"] = (lambda: [cv2.warpAffine(im, m_np[i], (w, h)) for i, im in enumerate(imgs_u8)]) if cv2 else None
        row["kornia-rs"] = (
            (lambda: [rs_affine(im, m_rs[i], (h, w), "bilinear") for i, im in enumerate(imgs_u8)])
            if rs_affine
            else None
        )
        ops["warp_affine"] = row

    if include("rotate"):
        row = kornia_row("rotate", lambda: KG.rotate(batch_f, angle))
        if cv2:
            row["opencv"] = lambda: [cv2.warpAffine(im, m_rot, (w, h)) for im in imgs_u8]
        row["torchvision v2"] = (
            (lambda: tvf.rotate(batch_f, angle_deg, interpolation=tvf.InterpolationMode.BILINEAR)) if tvf else None
        )
        row["kornia-rs"] = (
            (lambda: [rs_affine(im, m_rot_rs, (h, w), "bilinear") for im in imgs_u8]) if rs_affine else None
        )
        ops["rotate"] = row

    if include("rotate_shared"):
        # One transform shared by the batch; keep the independent-parameter row above too.
        shared_angle = angle[:1]
        row = kornia_row("rotate_shared", lambda: KG.rotate(batch_f, shared_angle))
        if cv2:
            row["opencv"] = lambda: [cv2.warpAffine(im, m_rot, (w, h)) for im in imgs_u8]
        row["torchvision v2"] = (
            (lambda: tvf.rotate(batch_f, angle_deg, interpolation=tvf.InterpolationMode.BILINEAR)) if tvf else None
        )
        row["kornia-rs"] = (
            (lambda: [rs_affine(im, m_rot_rs, (h, w), "bilinear") for im in imgs_u8]) if rs_affine else None
        )
        ops["rotate_shared"] = row

    if include("resize"):
        dst_size = (h // 2, w // 2)
        row = kornia_row("resize", lambda: KG.resize(batch_f, dst_size, interpolation="bilinear"))
        row["opencv"] = (lambda: [cv2.resize(im, (dst_size[1], dst_size[0])) for im in imgs_u8]) if cv2 else None
        row["torchvision v2"] = (lambda: tvf.resize(batch_f, list(dst_size), antialias=False)) if tvf else None
        row["kornia-rs"] = (
            (lambda: [rs_resize(im, dst_size, "bilinear", antialias=False) for im in imgs_u8]) if rs_resize else None
        )
        ops["resize"] = row

    if include("get_perspective_transform"):
        row = kornia_row("get_perspective_transform", lambda: KG.get_perspective_transform(src_pts, dst_pts))
        row["opencv"] = (lambda: [cv2.getPerspectiveTransform(src_np[i], dst_np[i]) for i in range(b)]) if cv2 else None
        ops["get_perspective_transform"] = row
    if include("remap"):
        # Quarter-pixel offsets are represented exactly by OpenCV's 1/32 interpolation table.
        yy, xx = np.mgrid[:h, :w].astype(np.float32)
        map_x_np, map_y_np = xx + 0.25, yy - 0.5
        map_x = torch.from_numpy(map_x_np).to(device=device, dtype=dtype).expand(b, -1, -1)
        map_y = torch.from_numpy(map_y_np).to(device=device, dtype=dtype).expand(b, -1, -1)
        imgs_f = [im.astype(np.float32) / 255 for im in imgs_u8]
        row = kornia_row("remap", lambda: KG.remap(batch_f, map_x, map_y, align_corners=True))
        row["opencv"] = (
            (
                lambda: [
                    cv2.remap(im, map_x_np, map_y_np, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT) for im in imgs_f
                ]
            )
            if cv2
            else None
        )
        ops["remap"] = row

    if include("transform_points"):
        points_cpu = torch.rand(b, h * w, 2, generator=gen) * torch.tensor([w, h])
        transform_cpu = torch.tensor([[1.05, 0.02, 2.0], [-0.01, 0.95, -1.0], [0.0001, -0.0002, 1.0]])
        transforms = transform_cpu.expand(b, -1, -1).to(device=device, dtype=dtype)
        points = points_cpu.to(device=device, dtype=dtype)
        point_arrays = [np.ascontiguousarray(p.numpy()[:, None]) for p in points_cpu]
        transform_np = transform_cpu.numpy()
        row = kornia_row("transform_points", KG.transform_points, transforms, points)
        row["opencv"] = (lambda: [cv2.perspectiveTransform(p, transform_np) for p in point_arrays]) if cv2 else None
        ops["transform_points"] = row

    if include("depth_to_3d"):
        depth = 0.5 + 4 * batch_f[:, :1]
        intrinsics = torch.tensor(
            [[w, 0.0, (w - 1) / 2], [0.0, w, (h - 1) / 2], [0.0, 0.0, 1.0]], device=device, dtype=dtype
        ).expand(b, -1, -1)
        row = kornia_row("depth_to_3d", KG.depth_to_3d, depth, intrinsics)
        ops["depth_to_3d"] = row
    return ops, kornia_row.compile_failures


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_flagship_args(parser, ops=OPS)
    args = parser.parse_args()
    device, dtype, sync = setup_run(args)

    cv2, cv2_error = optional_import("cv2")
    tvf, tvf_error = optional_import("torchvision.transforms.v2.functional")

    missing = [
        (name, err) for lib, name, err in [(cv2, "opencv", cv2_error), (tvf, "torchvision", tvf_error)] if lib is None
    ]
    missing.extend(
        (f"kornia-rs/{op}", f"{name} unavailable")
        for op, name in RS_OPS.items()
        if (args.ops is None or op in args.ops) and kornia_rs_fn(name) is None
    )
    meta = start_run(
        "flagship geometry",
        args,
        device,
        units="items/s",
        regimes=[
            "kornia/torchvision: batched float; opencv: uint8 images except float32 remap/point transforms (CPU)",
            "kornia-rs (CPU): uint8 HWC loop, bilinear/no antialias; prepared forward matrices; rotate via warp_affine",
            "Rust warp borders: zero outside [0,W)x[0,H), clamp last neighbor inside; differs from zero-pad blending",
            "remap: prepared pixel maps (+0.25,-0.5), bilinear, zero padding, align_corners=True",
            "items = images, depth maps, H*W point sets, or four-correspondence solves; depth_to_3d has no baseline",
        ],
        missing=missing,
    )
    backends = ["kornia (eager)", "kornia (compiled)", "torchvision v2", "opencv", "kornia-rs"]
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
