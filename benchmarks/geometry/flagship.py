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

===========================  ==============================================  ==================
kornia (batched float BCHW)  OpenCV (uint8 HWC, per-image Python loop)       torchvision v2
===========================  ==============================================  ==================
``warp_perspective``         ``cv2.warpPerspective``                         —
``warp_affine``              ``cv2.warpAffine``                              —
``rotate``                   ``cv2.warpAffine(getRotationMatrix2D)``         ``tvf.rotate``
``resize``                   ``cv2.resize``                                  ``tvf.resize``
``get_perspective_transform``  ``cv2.getPerspectiveTransform`` (per pair)    —
===========================  ==============================================  ==================

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
import platform
import sys
from pathlib import Path
from types import ModuleType
from typing import Callable, Optional

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import run_batch_sweep, run_metadata, save_json

import kornia.geometry as KG

Backend = Optional[Callable[[], object]]


def build_ops(
    b: int,
    h: int,
    w: int,
    device: torch.device,
    dtype: torch.dtype,
    do_compile: bool,
    cv2: Optional[ModuleType],
    tvf: Optional[ModuleType],
) -> tuple[dict[str, dict[str, Backend]], dict[str, str]]:
    """Build {op: {backend: zero-arg callable}} with identical transform params per backend.

    Also returns {op: exception name} for ops whose ``torch.compile`` warmup failed, so the
    caller can report them instead of leaving a silent skip cell.
    """
    rng = np.random.default_rng(0)
    imgs_u8 = [(rng.random((h, w, 3)) * 255).astype(np.uint8) for _ in range(b)]
    batch_f = (
        torch.stack([torch.from_numpy(im).permute(2, 0, 1) for im in imgs_u8]).to(device=device, dtype=dtype).div(255)
    )

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

    compile_failures: dict[str, str] = {}

    def kornia_row(label: str, fn: Callable[[], object]) -> dict[str, Backend]:
        row: dict[str, Backend] = {"kornia (eager)": fn}
        if do_compile:
            torch._dynamo.reset()
            compiled = torch.compile(fn)
            try:
                compiled()  # warmup: compile + autotune before the timed region
                row["kornia (compiled)"] = compiled
            except Exception as e:
                row["kornia (compiled)"] = None
                compile_failures[label] = type(e).__name__
        return row

    ops: dict[str, dict[str, Backend]] = {}

    row = kornia_row("warp_perspective", lambda: KG.warp_perspective(batch_f, h_mat, (h, w)))
    row["opencv"] = (
        (lambda: [cv2.warpPerspective(im, h_np[i], (w, h)) for i, im in enumerate(imgs_u8)]) if cv2 else None
    )
    row["torchvision v2"] = None
    ops["warp_perspective"] = row

    row = kornia_row("warp_affine", lambda: KG.warp_affine(batch_f, m_affine, (h, w)))
    row["opencv"] = (lambda: [cv2.warpAffine(im, m_np[i], (w, h)) for i, im in enumerate(imgs_u8)]) if cv2 else None
    row["torchvision v2"] = None
    ops["warp_affine"] = row

    row = kornia_row("rotate", lambda: KG.rotate(batch_f, angle))
    if cv2:
        m_rot = cv2.getRotationMatrix2D((w / 2, h / 2), angle_deg, 1.0)
        row["opencv"] = lambda: [cv2.warpAffine(im, m_rot, (w, h)) for im in imgs_u8]
    else:
        row["opencv"] = None
    row["torchvision v2"] = (lambda: tvf.rotate(batch_f, angle_deg)) if tvf else None
    ops["rotate"] = row

    dst_size = (h // 2, w // 2)
    row = kornia_row("resize", lambda: KG.resize(batch_f, dst_size, interpolation="bilinear"))
    row["opencv"] = (lambda: [cv2.resize(im, (dst_size[1], dst_size[0])) for im in imgs_u8]) if cv2 else None
    row["torchvision v2"] = (lambda: tvf.resize(batch_f, list(dst_size), antialias=False)) if tvf else None
    ops["resize"] = row

    row = kornia_row("get_perspective_transform", lambda: KG.get_perspective_transform(src_pts, dst_pts))
    row["opencv"] = (lambda: [cv2.getPerspectiveTransform(src_np[i], dst_np[i]) for i in range(b)]) if cv2 else None
    row["torchvision v2"] = None
    ops["get_perspective_transform"] = row
    return ops, compile_failures


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--batches", type=str, default="1,8,32", help="comma-separated batch sizes to sweep")
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--dtype", type=str, default="float32", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--compile", action="store_true", help="also time torch.compile'd kornia")
    parser.add_argument("--json", type=str, default=None, help="write machine-readable results to this path")
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)
    sync = torch.mps.synchronize if device.type == "mps" else None  # Timer only syncs CUDA

    try:
        import cv2
    except Exception:
        cv2 = None
    try:
        import torchvision.transforms.v2.functional as tvf
    except Exception:
        tvf = None

    meta = run_metadata(device)
    print(f"# flagship geometry benchmark — commit {meta['git_commit']} — {platform.platform()}")
    if device.type == "cuda":
        print(f"# CUDA device: {meta['cuda_device']}")
    print(f"# device={device}, dtype={args.dtype}, threads={args.threads}, size={args.size} — throughput items/s")
    print("# kornia/torchvision: batched float BCHW; opencv: uint8 HWC per-image loop (CPU); '-' = skipped")
    for lib, name in [(cv2, "opencv"), (tvf, "torchvision")]:
        if lib is None:
            print(f"# NOTE: {name} not installed — its column is skipped")

    backends = ["kornia (eager)", "kornia (compiled)", "torchvision v2", "opencv"]
    results = run_batch_sweep(
        [int(x) for x in args.batches.split(",") if x.strip()],
        lambda b: build_ops(b, args.size, args.size, device, dtype, args.compile, cv2, tvf),
        backends,
        row_fields=lambda b: {"height": args.size, "width": args.size, "dtype": args.dtype},
        sync=sync,
    )
    if args.json:
        out = save_json(args.json, meta, results)
        print(f"# results written to {out}")


if __name__ == "__main__":
    main()
