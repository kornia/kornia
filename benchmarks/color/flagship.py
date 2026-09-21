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

"""Flagship color benchmark: kornia.color vs OpenCV, torchvision v2, scikit-image and PIL.

Covers the conversions a pipeline typically pays for — the luma reduction, the two perceptual
spaces, a video space and demosaicing — not every pair the module exports:

====================  ==============================  ====================  =============  ====================
kornia.color          OpenCV (``cv2.cvtColor``)       torchvision v2        scikit-image   PIL
====================  ==============================  ====================  =============  ====================
``rgb_to_grayscale``  ``COLOR_RGB2GRAY`` (uint8)      ``rgb_to_grayscale``  ``rgb2gray``   ``convert("L")``
``rgb_to_hsv``        ``COLOR_RGB2HSV_FULL`` (uint8)  —                     ``rgb2hsv``    ``convert("HSV")``
``rgb_to_lab``        ``COLOR_RGB2Lab`` (float32)     —                     ``rgb2lab``    —
``rgb_to_ycbcr``      ``COLOR_RGB2YCrCb`` (uint8)     —                     ``rgb2ycbcr``  ``convert("YCbCr")``
``raw_to_rgb``        ``COLOR_BayerBG2RGB`` (uint8)   —                     —              —
====================  ==============================  ====================  =============  ====================

Regimes (see ``benchmarks/README.md``): kornia/torchvision run a batched float BCHW tensor on CPU
or GPU and kornia is differentiable; OpenCV and PIL run single uint8 HWC images on CPU in a Python
loop, their native regime, except Lab, which OpenCV computes on float32 input so its output is real
CIE Lab rather than the 8-bit rescaling; scikit-image runs normalized float HWC images per image.
PIL receives ready-made ``Image`` objects, as kornia receives a ready-made tensor. OpenCV
parallelizes ``cvtColor`` over rows, which a single 256x256 image barely feeds, so the pointwise
conversions also get an ``opencv (stacked)`` column: one call on the whole batch viewed as a
``(B*H, W, 3)`` image, built outside the timed call. Demosaicing is not pointwise and has no
stacked cell. Output conventions differ and are not normalized away: OpenCV's YCrCb swaps the
chroma channels, 8-bit HSV and YCbCr are rescaled to 0-255, scikit-image's ``rgb2ycbcr`` uses the
16-235 studio range, and kornia's hue is in radians. OpenCV's Bayer pattern names are shifted by
one pixel relative to kornia's ``CFA`` names; the ``raw_to_rgb`` row times bilinear demosaicing of
the same single-channel mosaic in both. These are throughput comparisons, not output-equivalent
ones. Throughput is img/s.

Usage:
    python benchmarks/color/flagship.py --batches 1,8,32 --size 256 --device cpu
    python benchmarks/color/flagship.py --device cuda --compile --json color_cuda.json
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

import kornia.color as KC

OPS = ("rgb_to_grayscale", "rgb_to_hsv", "rgb_to_lab", "rgb_to_ycbcr", "raw_to_rgb")


def build_ops(
    b: int,
    h: int,
    w: int,
    device: torch.device,
    dtype: torch.dtype,
    do_compile: bool,
    cv2: Optional[ModuleType],
    tvf: Optional[ModuleType],
    skc: Optional[ModuleType],
    pil: Optional[ModuleType],
    skip_compile: frozenset[str] = frozenset(),
    selected: Optional[frozenset[str]] = None,
) -> tuple[dict[str, dict[str, Backend]], dict[str, str]]:
    """Build {op: {backend: zero-arg callable}}; each callable converts the whole batch once."""
    imgs_u8, batch_f = image_batch(b, h, w, device, dtype)
    imgs_f = [im.astype(np.float32) / 255.0 for im in imgs_u8]
    # One sample per pixel, as a sensor delivers it: the green plane stands in for the mosaic.
    raws_u8 = [np.ascontiguousarray(im[..., 1]) for im in imgs_u8]
    raw_f = batch_f[:, 1:2].contiguous()
    stacked_u8 = np.stack(imgs_u8).reshape(b * h, w, 3)
    stacked_f = np.stack(imgs_f).reshape(b * h, w, 3)
    pil_imgs = [pil.fromarray(im) for im in imgs_u8] if pil else []

    kornia_row = KorniaRows(device, do_compile, skip_compile)
    ops: dict[str, dict[str, Backend]] = {}

    def include(name: str) -> bool:
        return selected is None or name in selected

    def cv_loop(code: int, images: list[np.ndarray]) -> Backend:
        return (lambda: [cv2.cvtColor(im, code) for im in images]) if cv2 else None

    def cv_stacked(code: int, stacked: np.ndarray) -> Backend:
        return (lambda: cv2.cvtColor(stacked, code)) if cv2 else None

    def sk_loop(fn_name: str) -> Backend:
        return (lambda: [getattr(skc, fn_name)(im) for im in imgs_f]) if skc else None

    def pil_loop(mode: str) -> Backend:
        return (lambda: [im.convert(mode) for im in pil_imgs]) if pil else None

    if include("rgb_to_grayscale"):
        row = kornia_row("rgb_to_grayscale", KC.rgb_to_grayscale, batch_f)
        row["torchvision v2"] = (lambda: tvf.rgb_to_grayscale(batch_f)) if tvf else None
        row["opencv"] = cv_loop(cv2.COLOR_RGB2GRAY, imgs_u8) if cv2 else None
        row["opencv (stacked)"] = cv_stacked(cv2.COLOR_RGB2GRAY, stacked_u8) if cv2 else None
        row["scikit-image"] = sk_loop("rgb2gray")
        row["PIL"] = pil_loop("L")
        ops["rgb_to_grayscale"] = row

    if include("rgb_to_hsv"):
        row = kornia_row("rgb_to_hsv", KC.rgb_to_hsv, batch_f)
        row["opencv"] = cv_loop(cv2.COLOR_RGB2HSV_FULL, imgs_u8) if cv2 else None
        row["opencv (stacked)"] = cv_stacked(cv2.COLOR_RGB2HSV_FULL, stacked_u8) if cv2 else None
        row["scikit-image"] = sk_loop("rgb2hsv")
        row["PIL"] = pil_loop("HSV")
        ops["rgb_to_hsv"] = row

    if include("rgb_to_lab"):
        row = kornia_row("rgb_to_lab", KC.rgb_to_lab, batch_f)
        row["opencv"] = cv_loop(cv2.COLOR_RGB2Lab, imgs_f) if cv2 else None
        row["opencv (stacked)"] = cv_stacked(cv2.COLOR_RGB2Lab, stacked_f) if cv2 else None
        row["scikit-image"] = sk_loop("rgb2lab")
        ops["rgb_to_lab"] = row

    if include("rgb_to_ycbcr"):
        row = kornia_row("rgb_to_ycbcr", KC.rgb_to_ycbcr, batch_f)
        row["opencv"] = cv_loop(cv2.COLOR_RGB2YCrCb, imgs_u8) if cv2 else None
        row["opencv (stacked)"] = cv_stacked(cv2.COLOR_RGB2YCrCb, stacked_u8) if cv2 else None
        row["scikit-image"] = sk_loop("rgb2ycbcr")
        row["PIL"] = pil_loop("YCbCr")
        ops["rgb_to_ycbcr"] = row

    if include("raw_to_rgb"):
        row = kornia_row("raw_to_rgb", KC.raw_to_rgb, raw_f, KC.CFA.BG)
        row["opencv"] = cv_loop(cv2.COLOR_BayerBG2RGB, raws_u8) if cv2 else None
        ops["raw_to_rgb"] = row

    return ops, kornia_row.compile_failures


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_flagship_args(parser, ops=OPS)
    args = parser.parse_args()
    device, dtype, sync = setup_run(args)

    cv2, cv2_error = optional_import("cv2")
    tvf, tvf_error = optional_import("torchvision.transforms.v2.functional")
    skc, skc_error = optional_import("skimage.color")
    pil, pil_error = optional_import("PIL.Image")

    libs = [
        (tvf, "torchvision", tvf_error),
        (skc, "scikit-image", skc_error),
        (cv2, "opencv", cv2_error),
        (pil, "PIL", pil_error),
    ]
    meta = start_run(
        "flagship color",
        args,
        device,
        units="img/s",
        regimes=[
            "kornia/torchvision: batched float BCHW; opencv/PIL: uint8 HWC per-image loop (CPU), "
            "opencv Lab on float32; scikit-image: normalized float HWC per-image loop",
            "opencv (stacked): one cvtColor on the batch viewed as (B*H, W, 3); PIL: prebuilt Image objects",
        ],
        missing=[(name, err) for lib, name, err in libs if lib is None],
    )
    backends = [
        "kornia (eager)",
        "kornia (compiled)",
        "torchvision v2",
        "scikit-image",
        "opencv",
        "opencv (stacked)",
        "PIL",
    ]
    results = run_batch_sweep(
        batch_list(args),
        lambda b: build_ops(
            b, args.size, args.size, device, dtype, args.compile, cv2, tvf, skc, pil, args.skip_compile_ops, args.ops
        ),
        backends,
        row_fields=image_row_fields(args),
        sync=sync,
        units="img/s",
        min_run_time=args.min_run_time,
    )
    finish_run(args, "color", meta, results)


if __name__ == "__main__":
    main()
