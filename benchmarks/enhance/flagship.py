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

"""Flagship enhance benchmark: kornia.enhance vs torchvision v2, albumentations, OpenCV, scikit-image and PIL.

Covers one representative of each family the module exports — normalization, a pointwise tone
curve, the two HSV-space adjustments, and global and local histogram equalization:

=====================  =====================  ============================  ===================================
kornia.enhance         torchvision v2         OpenCV / albumentations       scikit-image / PIL
=====================  =====================  ============================  ===================================
``normalize``          ``normalize``          albumentations ``Normalize``  —
``adjust_gamma``       ``adjust_gamma``       ``cv2.LUT`` (uint8)           ``exposure.adjust_gamma``
                                              ``RandomGamma``
``adjust_hue``         ``adjust_hue``         ``ColorJitter(hue=)``         —
``adjust_saturation``  ``adjust_saturation``  ``ColorJitter(saturation=)``  PIL ``ImageEnhance.Color``
``equalize``           ``equalize`` (uint8)   ``cv2.equalizeHist``/channel  ``equalize_hist``; PIL ``equalize``
                                              ``Equalize``
``equalize_clahe``     —                      ``cv2.createCLAHE``/channel   ``exposure.equalize_adapthist``
                                              ``CLAHE``
=====================  =====================  ============================  ===================================

Regimes (see ``benchmarks/README.md``): kornia/torchvision run a batched float BCHW tensor on CPU
or GPU and kornia is differentiable, except torchvision's ``equalize``, which runs the same batch
as uint8 BCHW (its documented input); OpenCV, albumentations and PIL run single uint8 HWC images on
CPU in a Python loop, their native regime; scikit-image runs normalized float HWC images per image.
PIL receives ready-made ``Image`` objects, as kornia receives a ready-made tensor. OpenCV's
per-channel ops split and merge with ``cv2.split``/``cv2.merge``. The albumentations transforms are
pinned to the same fixed parameter (``gamma_limit``, ``saturation`` and ``hue`` as one-point
ranges, the other ``ColorJitter`` factors at identity, which it skips). Parameters are matched:
mean/std ImageNet normalization, gamma 2.2, a hue shift of 0.1 turn (``0.2 * pi`` radians in
kornia, ``0.1`` in torchvision), saturation 1.5, and CLAHE with clip limit 40 on an 8x8 grid. The
gamma LUT is the native uint8 implementation of a pointwise curve. CLAHE clip limits are not
interchangeable across libraries: OpenCV and kornia use an absolute multiple of the uniform bin
height, scikit-image a normalized fraction, here set to ``40 / 256`` — equal in spirit only.
albumentations' ``CLAHE`` does less work on RGB input: it converts to Lab, equalizes only the L
channel and converts back, while kornia, OpenCV and scikit-image equalize all three channels; its
cell is the transform as users call it, not the same computation. scikit-image's ``adjust_gamma``
runs on float here, the documented regime; on ``uint8`` it uses a lookup table and is several times
faster, which the OpenCV LUT cell represents. ``adjust_saturation`` is not the same algorithm
across columns: kornia scales S in HSV space and converts back, while torchvision and PIL blend
with the grayscale image (kornia's ``adjust_saturation_with_gray_subtraction``); the row times each
library's saturation adjustment as users call it. Throughput is img/s.

Usage:
    python benchmarks/enhance/flagship.py --batches 1,8,32 --size 256 --device cpu
    python benchmarks/enhance/flagship.py --device cuda --compile --json enhance_cuda.json
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from types import ModuleType
from typing import Callable, Optional

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

import kornia.enhance as KE

OPS = ("normalize", "adjust_gamma", "adjust_hue", "adjust_saturation", "equalize", "equalize_clahe")
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
GAMMA, HUE_TURN, SATURATION, CLAHE_CLIP, CLAHE_GRID = 2.2, 0.1, 1.5, 40.0, (8, 8)


def build_ops(
    b: int,
    h: int,
    w: int,
    device: torch.device,
    dtype: torch.dtype,
    do_compile: bool,
    libs: dict[str, Optional[ModuleType]],
    skip_compile: frozenset[str] = frozenset(),
    selected: Optional[frozenset[str]] = None,
) -> tuple[dict[str, dict[str, Backend]], dict[str, str]]:
    """Build {op: {backend: zero-arg callable}}; each callable adjusts the whole batch once."""
    cv2, tvf, A, ske, pil_enh, pil_ops, pil = (libs[k] for k in ("cv2", "tvf", "A", "ske", "pil_enh", "pil_ops", "pil"))
    imgs_u8, batch_f = image_batch(b, h, w, device, dtype)
    imgs_f = [im.astype(np.float32) / 255.0 for im in imgs_u8]
    batch_u8 = (batch_f.float() * 255).round().to(torch.uint8)
    pil_imgs = [pil.fromarray(im) for im in imgs_u8] if pil else []
    mean = torch.tensor(MEAN, device=device, dtype=dtype)
    std = torch.tensor(STD, device=device, dtype=dtype)

    kornia_row = KorniaRows(device, do_compile, skip_compile)
    ops: dict[str, dict[str, Backend]] = {}

    def include(name: str) -> bool:
        return selected is None or name in selected

    def cv_per_channel(fn: Callable[[np.ndarray], np.ndarray]) -> Backend:
        return lambda: [cv2.merge([fn(c) for c in cv2.split(im)]) for im in imgs_u8]

    def sk_per_channel(fn: Callable[[np.ndarray], np.ndarray]) -> Backend:
        return lambda: [np.stack([fn(im[..., c]) for c in range(im.shape[-1])], axis=-1) for im in imgs_f]

    def alb(t: object) -> Backend:
        return lambda: [t(image=im)["image"] for im in imgs_u8]

    def color_jitter(**factor: tuple[float, float]) -> object:
        identity = {"brightness": (1.0, 1.0), "contrast": (1.0, 1.0), "saturation": (1.0, 1.0), "hue": (0.0, 0.0)}
        return A.ColorJitter(**{**identity, **factor}, p=1.0)

    if include("normalize"):
        row = kornia_row("normalize", KE.normalize, batch_f, mean, std)
        row["torchvision v2"] = (lambda: tvf.normalize(batch_f, list(MEAN), list(STD))) if tvf else None
        row["albumentations"] = alb(A.Normalize(mean=MEAN, std=STD, max_pixel_value=255.0, p=1.0)) if A else None
        ops["normalize"] = row

    if include("adjust_gamma"):
        row = kornia_row("adjust_gamma", KE.adjust_gamma, batch_f, GAMMA)
        row["torchvision v2"] = (lambda: tvf.adjust_gamma(batch_f, GAMMA)) if tvf else None
        if cv2:
            lut = np.clip(((np.arange(256) / 255.0) ** GAMMA) * 255.0 + 0.5, 0, 255).astype(np.uint8)
            row["opencv"] = lambda: [cv2.LUT(im, lut) for im in imgs_u8]
        row["albumentations"] = alb(A.RandomGamma(gamma_limit=(GAMMA * 100, GAMMA * 100), p=1.0)) if A else None
        row["scikit-image"] = (lambda: [ske.adjust_gamma(im, GAMMA) for im in imgs_f]) if ske else None
        ops["adjust_gamma"] = row

    if include("adjust_hue"):
        row = kornia_row("adjust_hue", KE.adjust_hue, batch_f, 2 * math.pi * HUE_TURN)
        row["torchvision v2"] = (lambda: tvf.adjust_hue(batch_f, HUE_TURN)) if tvf else None
        row["albumentations"] = alb(color_jitter(hue=(HUE_TURN, HUE_TURN))) if A else None
        ops["adjust_hue"] = row

    if include("adjust_saturation"):
        row = kornia_row("adjust_saturation", KE.adjust_saturation, batch_f, SATURATION)
        row["torchvision v2"] = (lambda: tvf.adjust_saturation(batch_f, SATURATION)) if tvf else None
        row["albumentations"] = alb(color_jitter(saturation=(SATURATION, SATURATION))) if A else None
        if pil and pil_enh:
            row["PIL"] = lambda: [pil_enh.Color(im).enhance(SATURATION) for im in pil_imgs]
        ops["adjust_saturation"] = row

    if include("equalize"):
        row = kornia_row("equalize", KE.equalize, batch_f)
        row["torchvision v2"] = (lambda: tvf.equalize(batch_u8)) if tvf else None
        row["albumentations"] = alb(A.Equalize(mode="cv", by_channels=True, p=1.0)) if A else None
        row["opencv"] = cv_per_channel(cv2.equalizeHist) if cv2 else None
        row["scikit-image"] = sk_per_channel(ske.equalize_hist) if ske else None
        if pil and pil_ops:
            row["PIL"] = lambda: [pil_ops.equalize(im) for im in pil_imgs]
        ops["equalize"] = row

    if include("equalize_clahe"):
        row = kornia_row("equalize_clahe", KE.equalize_clahe, batch_f, CLAHE_CLIP, CLAHE_GRID)
        if cv2:
            clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=CLAHE_GRID)
            row["opencv"] = cv_per_channel(clahe.apply)
        row["albumentations"] = (
            alb(A.CLAHE(clip_limit=(CLAHE_CLIP, CLAHE_CLIP), tile_grid_size=CLAHE_GRID, p=1.0)) if A else None
        )
        if ske:
            kernel = (h // CLAHE_GRID[0], w // CLAHE_GRID[1])
            row["scikit-image"] = lambda: [
                ske.equalize_adapthist(im, kernel_size=kernel, clip_limit=CLAHE_CLIP / 256) for im in imgs_f
            ]
        ops["equalize_clahe"] = row

    return ops, kornia_row.compile_failures


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_flagship_args(parser, ops=OPS)
    args = parser.parse_args()
    device, dtype, sync = setup_run(args)

    names = {
        "cv2": "cv2",
        "tvf": "torchvision.transforms.v2.functional",
        "A": "albumentations",
        "ske": "skimage.exposure",
        "pil": "PIL.Image",
        "pil_enh": "PIL.ImageEnhance",
        "pil_ops": "PIL.ImageOps",
    }
    imported = {key: optional_import(module) for key, module in names.items()}
    libs = {key: mod for key, (mod, _) in imported.items()}
    labels = {"cv2": "opencv", "tvf": "torchvision", "A": "albumentations", "ske": "scikit-image", "pil": "PIL"}
    meta = start_run(
        "flagship enhance",
        args,
        device,
        units="img/s",
        regimes=[
            "kornia/torchvision: batched float BCHW (torchvision equalize: uint8 BCHW); "
            "albumentations/opencv/PIL: uint8 HWC per-image loop (CPU); scikit-image: normalized float HWC loop",
            "PIL: prebuilt Image objects; albumentations CLAHE equalizes only L in Lab, the others all 3 channels",
        ],
        missing=[(label, imported[key][1]) for key, label in labels.items() if libs[key] is None],
    )
    backends = [
        "kornia (eager)",
        "kornia (compiled)",
        "torchvision v2",
        "albumentations",
        "scikit-image",
        "opencv",
        "PIL",
    ]
    results = run_batch_sweep(
        batch_list(args),
        lambda b: build_ops(
            b, args.size, args.size, device, dtype, args.compile, libs, args.skip_compile_ops, args.ops
        ),
        backends,
        row_fields=image_row_fields(args),
        sync=sync,
        units="img/s",
        min_run_time=args.min_run_time,
    )
    finish_run(args, "enhance", meta, results)


if __name__ == "__main__":
    main()
