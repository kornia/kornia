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

"""Flagship augmentation benchmark: each library's augmentation API, parameter sampling included.

Benchmarks augmentations **as augmentations** — through the user-facing random-transform classes
(kornia's ``forward_parameters`` + apply, torchvision v2 transform objects, albumentations
transforms), not the underlying deterministic functionals. Transform objects are constructed once,
outside the timed region; the timed region is parameter sampling + application, per call.

=====================  ==========================  =========================  ================  ==================
kornia.augmentation    torchvision.transforms.v2   albumentations             opencv            PIL
=====================  ==========================  =========================  ================  ==================
RandomHorizontalFlip   RandomHorizontalFlip        HorizontalFlip             ``cv2.flip``      ``Image.transpose``
RandomAffine           RandomAffine                Affine                     —                 —
RandomPerspective      RandomPerspective           Perspective                —                 —
RandomResizedCrop      RandomResizedCrop           RandomResizedCrop          —                 —
ColorJiggle            ColorJitter                 ColorJitter                —                 —
RandomGaussianBlur     GaussianBlur                GaussianBlur               —                 —
RandomBrightness       ColorJitter(brightness=)    RandomBrightnessContrast   —                 —
RandomGrayscale        RandomGrayscale             ToGray                     ``cv2.cvtColor``  ``convert("L")``
=====================  ==========================  =========================  ================  ==================

Regimes (see ``benchmarks/README.md``): kornia/torchvision run a batched float BCHW tensor on
CPU or GPU and kornia is differentiable; albumentations/OpenCV/PIL run single uint8 HWC images on
CPU in a Python loop — their native regime. OpenCV and PIL are only listed where the augmentation
is parameter-free (flip via ``Image.transpose``, grayscale via ``convert("L")``): for
randomly-parameterized augmentations, albumentations *is* the OpenCV-backed baseline. PIL is
usually the slowest but serves as the signal-processing-correct reference implementation.
Parameter distributions are matched in spirit across libraries, but
parameterizations differ (e.g. perspective distortion scales) — columns are regime comparisons,
not bit-exact races. RandomResizedCrop outputs size//2 per side for every backend;
throughput is img/s of input images.

Usage:
    python benchmarks/augmentation/flagship.py --batches 1,8,32 --size 256 --device cpu
    python benchmarks/augmentation/flagship.py --device cuda --compile --json aug_cuda.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import ModuleType
from typing import Optional

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

import kornia.augmentation as KA


def build_ops(
    b: int,
    h: int,
    w: int,
    device: torch.device,
    dtype: torch.dtype,
    do_compile: bool,
    T2: Optional[ModuleType],
    A: Optional[ModuleType],
    cv2: Optional[ModuleType],
    pil: Optional[ModuleType],
    skip_compile: frozenset[str] = frozenset(),
    selected: Optional[frozenset[str]] = None,
) -> tuple[dict[str, dict[str, Backend]], dict[str, str]]:
    """Build {op: {backend: zero-arg callable}}; each callable transforms the whole batch once.

    Rows outside ``selected`` (``--ops``) are neither compiled nor timed.
    """
    imgs_u8, batch_f = image_batch(b, h, w, device, dtype)
    compiled_rows = KorniaRows(device, do_compile, skip_compile)

    def kornia_row(label: str, aug: torch.nn.Module) -> dict[str, Backend]:
        if selected is not None and label not in selected:
            return {}
        return compiled_rows(label, aug.to(device), batch_f)

    def tv(t: object) -> Backend:
        return lambda: t(batch_f)

    def alb(t: object) -> Backend:
        return lambda: [t(image=im)["image"] for im in imgs_u8]

    ops: dict[str, dict[str, Backend]] = {}

    row = kornia_row("RandomHorizontalFlip", KA.RandomHorizontalFlip(p=1.0))
    row["torchvision v2"] = tv(T2.RandomHorizontalFlip(p=1.0)) if T2 else None
    row["albumentations"] = alb(A.HorizontalFlip(p=1.0)) if A else None
    row["opencv"] = (lambda: [cv2.flip(im, 1) for im in imgs_u8]) if cv2 else None
    row["PIL"] = (
        (lambda: [pil.fromarray(im).transpose(pil.Transpose.FLIP_LEFT_RIGHT) for im in imgs_u8]) if pil else None
    )
    ops["RandomHorizontalFlip"] = row

    row = kornia_row("RandomAffine", KA.RandomAffine(degrees=30.0, translate=(0.1, 0.1), scale=(0.8, 1.2), p=1.0))
    row["torchvision v2"] = tv(T2.RandomAffine(degrees=30.0, translate=(0.1, 0.1), scale=(0.8, 1.2))) if T2 else None
    row["albumentations"] = (
        alb(A.Affine(rotate=(-30.0, 30.0), translate_percent=(0.0, 0.1), scale=(0.8, 1.2), p=1.0)) if A else None
    )
    ops["RandomAffine"] = row

    row = kornia_row("RandomPerspective", KA.RandomPerspective(0.5, p=1.0))
    row["torchvision v2"] = tv(T2.RandomPerspective(distortion_scale=0.5, p=1.0)) if T2 else None
    row["albumentations"] = alb(A.Perspective(scale=(0.05, 0.1), p=1.0)) if A else None
    ops["RandomPerspective"] = row

    dst = (h // 2, w // 2)
    row = kornia_row("RandomResizedCrop", KA.RandomResizedCrop(dst))
    row["torchvision v2"] = tv(T2.RandomResizedCrop(dst, antialias=False)) if T2 else None
    row["albumentations"] = alb(A.RandomResizedCrop(size=dst, p=1.0)) if A else None
    ops["RandomResizedCrop"] = row

    row = kornia_row("ColorJiggle", KA.ColorJiggle(0.2, 0.2, 0.2, 0.1, p=1.0))
    row["torchvision v2"] = tv(T2.ColorJitter(0.2, 0.2, 0.2, 0.1)) if T2 else None
    row["albumentations"] = alb(A.ColorJitter(0.2, 0.2, 0.2, 0.1, p=1.0)) if A else None
    ops["ColorJiggle"] = row

    row = kornia_row("RandomGaussianBlur", KA.RandomGaussianBlur((5, 5), (0.1, 2.0), p=1.0))
    row["torchvision v2"] = tv(T2.GaussianBlur(5, sigma=(0.1, 2.0))) if T2 else None
    row["albumentations"] = alb(A.GaussianBlur(blur_limit=(5, 5), sigma_limit=(0.1, 2.0), p=1.0)) if A else None
    ops["RandomGaussianBlur"] = row

    row = kornia_row("RandomBrightness", KA.RandomBrightness(brightness=(0.8, 1.2), p=1.0))
    row["torchvision v2"] = tv(T2.ColorJitter(brightness=(0.8, 1.2))) if T2 else None
    row["albumentations"] = (
        alb(A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.0, p=1.0)) if A else None
    )
    ops["RandomBrightness"] = row

    row = kornia_row("RandomGrayscale", KA.RandomGrayscale(p=1.0))
    row["torchvision v2"] = tv(T2.RandomGrayscale(p=1.0)) if T2 else None
    row["albumentations"] = alb(A.ToGray(p=1.0)) if A else None
    row["opencv"] = (lambda: [cv2.cvtColor(im, cv2.COLOR_RGB2GRAY) for im in imgs_u8]) if cv2 else None
    row["PIL"] = (lambda: [pil.fromarray(im).convert("L") for im in imgs_u8]) if pil else None
    ops["RandomGrayscale"] = row

    return {k: v for k, v in ops.items() if selected is None or k in selected}, compiled_rows.compile_failures


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_flagship_args(parser)
    args = parser.parse_args()
    device, dtype, sync = setup_run(args)

    T2, t2_error = optional_import("torchvision.transforms.v2")
    A, a_error = optional_import("albumentations")
    cv2, cv2_error = optional_import("cv2")
    pil, pil_error = optional_import("PIL.Image")

    libs = [
        (T2, "torchvision", t2_error),
        (A, "albumentations", a_error),
        (cv2, "opencv", cv2_error),
        (pil, "PIL", pil_error),
    ]
    meta = start_run(
        "flagship augmentation",
        args,
        device,
        units="img/s",
        regimes=[
            "augmentation classes built once; timed region = parameter sampling + application per call",
            "kornia/torchvision: batched float BCHW; albumentations/opencv/PIL: uint8 HWC per-image loop (CPU)",
        ],
        missing=[(name, err) for lib, name, err in libs if lib is None],
    )
    backends = ["kornia (eager)", "kornia (compiled)", "torchvision v2", "albumentations", "opencv", "PIL"]
    results = run_batch_sweep(
        batch_list(args),
        lambda b: build_ops(
            b, args.size, args.size, device, dtype, args.compile, T2, A, cv2, pil, args.skip_compile_ops, args.ops
        ),
        backends,
        row_fields=image_row_fields(args),
        sync=sync,
        units="img/s",
        min_run_time=args.min_run_time,
    )
    finish_run(args, "augmentation", meta, results)


if __name__ == "__main__":
    main()
