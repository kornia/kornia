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

"""Flagship morphology benchmark: kornia.morphology vs torchmorph, OpenCV, albumentations and scikit-image.

Covers the two primitives and the two most used compound operators, all with a 5x5 square
structuring element:

========  ==========================  ================================  ==============  ==========================
kornia    torchmorph (CUDA only)      OpenCV (per-image loop)           albumentations  scikit-image (per channel)
========  ==========================  ================================  ==============  ==========================
dilation  ``grey_dilation``           ``cv2.dilate``                    ``"dilation"``  ``morphology.dilation``
erosion   ``grey_erosion``            ``cv2.erode``                     ``"erosion"``   ``morphology.erosion``
opening   ``grey_opening``            ``morphologyEx(MORPH_OPEN)``      —               ``morphology.opening``
gradient  ``morphological_gradient``  ``morphologyEx(MORPH_GRADIENT)``  —               ``dilation - erosion``
========  ==========================  ================================  ==============  ==========================

The kornia column names ``kornia.morphology`` functions and the albumentations column the
``operation`` of its ``Morphological`` transform.

Regimes (see ``benchmarks/README.md``): kornia runs a batched float BCHW tensor on CPU or GPU with
its default ``engine="auto"`` and is differentiable. ``auto`` picks ``unfold`` on CUDA and the exact
``shift`` engine elsewhere, except for a float32 or float64 CPU call that records a backward graph,
which keeps ``unfold``; this suite times the forward alone, so it measures ``unfold`` on CUDA and
``shift`` on CPU and MPS. `torchmorph <https://github.com/intcomp/torchmorph>`_
runs the same batched BCHW tensor through its custom CUDA kernels, which exist only for CUDA, so its
column is skipped on every other device; it computes in float32 (a float16 or bfloat16 input is
upcast inside the call) and is not differentiable. OpenCV runs single uint8 HWC images on CPU in
a Python loop, its native regime, and filters all channels in one call; albumentations'
``Morphological`` runs the same loop through its transform call, with its elliptical element
replaced by the 5x5 square (albumentations offers only dilation and erosion); scikit-image runs the
same uint8 images one channel at a time, because its grayscale morphology is 2-D. Borders are each
library's default: kornia's ``geodesic`` border and OpenCV's default border value both leave
out-of-image pixels out of the max/min; torchmorph and scikit-image follow SciPy and reflect at the
border (``mode="reflect"``). torchmorph's flat ``size=5`` element is the same 5x5 square.
scikit-image has no gradient operator, so its row is ``dilation - erosion`` in int16, which is the
definition. Throughput is img/s.

Usage:
    python benchmarks/morphology/flagship.py --batches 1,8,32 --size 256 --device cpu
    python benchmarks/morphology/flagship.py --device cuda --compile --json morphology_cuda.json
"""

from __future__ import annotations

import argparse
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

import kornia.morphology as KM

OPS = ("dilation", "erosion", "opening", "gradient")
KERNEL_SIZE = 5


def square_morphological(A: ModuleType, operation: str, element: np.ndarray) -> object:
    """albumentations' ``Morphological`` with a fixed element; its own draws an ellipse of size ``scale``."""

    class SquareMorphological(A.Morphological):
        def get_params(self) -> dict[str, np.ndarray]:
            return {"kernel": element}

    return SquareMorphological(operation=operation, p=1.0)


def build_ops(
    b: int,
    h: int,
    w: int,
    device: torch.device,
    dtype: torch.dtype,
    do_compile: bool,
    cv2: Optional[ModuleType],
    skm: Optional[ModuleType],
    A: Optional[ModuleType],
    tm: Optional[ModuleType],
    skip_compile: frozenset[str] = frozenset(),
    selected: Optional[frozenset[str]] = None,
) -> tuple[dict[str, dict[str, Backend]], dict[str, str]]:
    """Build {op: {backend: zero-arg callable}}; each callable filters the whole batch once."""
    imgs_u8, batch_f = image_batch(b, h, w, device, dtype)
    kernel = torch.ones(KERNEL_SIZE, KERNEL_SIZE, device=device, dtype=dtype)
    kernel_np = np.ones((KERNEL_SIZE, KERNEL_SIZE), dtype=np.uint8)
    footprint = np.ones((KERNEL_SIZE, KERNEL_SIZE), dtype=bool)

    kornia_row = KorniaRows(device, do_compile, skip_compile)
    ops: dict[str, dict[str, Backend]] = {}

    def include(name: str) -> bool:
        return selected is None or name in selected

    def per_channel(fn: Callable[[np.ndarray], np.ndarray]) -> Backend:
        return lambda: [np.stack([fn(im[..., c]) for c in range(im.shape[-1])], axis=-1) for im in imgs_u8]

    def sk_gradient(channel: np.ndarray) -> np.ndarray:
        dilated = skm.dilation(channel, footprint).astype(np.int16)
        return dilated - skm.erosion(channel, footprint).astype(np.int16)

    cv_ops = {
        "dilation": (lambda im: cv2.dilate(im, kernel_np)) if cv2 else None,
        "erosion": (lambda im: cv2.erode(im, kernel_np)) if cv2 else None,
        "opening": (lambda im: cv2.morphologyEx(im, cv2.MORPH_OPEN, kernel_np)) if cv2 else None,
        "gradient": (lambda im: cv2.morphologyEx(im, cv2.MORPH_GRADIENT, kernel_np)) if cv2 else None,
    }
    sk_ops = {
        "dilation": (lambda ch: skm.dilation(ch, footprint)) if skm else None,
        "erosion": (lambda ch: skm.erosion(ch, footprint)) if skm else None,
        "opening": (lambda ch: skm.opening(ch, footprint)) if skm else None,
        "gradient": sk_gradient if skm else None,
    }
    tm_ops = (
        {
            "dilation": tm.grey_dilation,
            "erosion": tm.grey_erosion,
            "opening": tm.grey_opening,
            "gradient": tm.morphological_gradient,
        }
        if tm
        else {}
    )
    alb_ops = {name: square_morphological(A, name, kernel_np) for name in ("dilation", "erosion")} if A else {}
    for name in OPS:
        if not include(name):
            continue
        row = kornia_row(name, getattr(KM, name), batch_f, kernel)
        cv_fn, sk_fn, tm_fn = cv_ops[name], sk_ops[name], tm_ops.get(name)
        row["torchmorph"] = (lambda tm_fn=tm_fn: tm_fn(batch_f, size=KERNEL_SIZE)) if tm_fn else None
        row["opencv"] = (lambda cv_fn=cv_fn: [cv_fn(im) for im in imgs_u8]) if cv_fn else None
        alb_t = alb_ops.get(name)
        row["albumentations"] = (lambda alb_t=alb_t: [alb_t(image=im)["image"] for im in imgs_u8]) if alb_t else None
        row["scikit-image"] = per_channel(sk_fn) if sk_fn else None
        ops[name] = row

    return ops, kornia_row.compile_failures


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_flagship_args(parser, ops=OPS)
    args = parser.parse_args()
    device, dtype, sync = setup_run(args)

    cv2, cv2_error = optional_import("cv2")
    skm, skm_error = optional_import("skimage.morphology")
    A, a_error = optional_import("albumentations")
    # torchmorph ships CUDA kernels only and rejects any other tensor, so it is not even imported off CUDA.
    tm, tm_error = optional_import("torchmorph") if device.type == "cuda" else (None, "CUDA only")

    libs = [
        (tm, "torchmorph", tm_error),
        (skm, "scikit-image", skm_error),
        (cv2, "opencv", cv2_error),
        (A, "albumentations", a_error),
    ]
    meta = start_run(
        "flagship morphology",
        args,
        device,
        units="img/s",
        regimes=[
            f"{KERNEL_SIZE}x{KERNEL_SIZE} square element; kornia: batched float BCHW; "
            "torchmorph: batched float32 BCHW (CUDA only); "
            "opencv/albumentations: uint8 HWC per-image loop (CPU); scikit-image: uint8 per-image, per-channel loop"
        ],
        missing=[(name, err) for lib, name, err in libs if lib is None],
    )
    backends = ["kornia (eager)", "kornia (compiled)", "torchmorph", "albumentations", "scikit-image", "opencv"]
    results = run_batch_sweep(
        batch_list(args),
        lambda b: build_ops(
            b, args.size, args.size, device, dtype, args.compile, cv2, skm, A, tm, args.skip_compile_ops, args.ops
        ),
        backends,
        torch_backends=("kornia (", "torchmorph"),
        row_fields=image_row_fields(args),
        sync=sync,
        units="img/s",
        min_run_time=args.min_run_time,
    )
    finish_run(args, "morphology", meta, results)


if __name__ == "__main__":
    main()
