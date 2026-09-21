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

"""Flagship metrics benchmark: kornia.metrics vs scikit-image.

Covers the two image-quality metrics and the segmentation metric an evaluation loop computes
most often:

============  =============================================================  ======================
kornia        scikit-image (float HWC per-image loop)                        notes
============  =============================================================  ======================
``psnr``      ``metrics.peak_signal_noise_ratio(data_range=1)``              —
``ssim``      ``metrics.structural_similarity`` (Gaussian 11x11, sigma 1.5)  kornia returns the map
``mean_iou``  —                                                              21 classes
============  =============================================================  ======================

Regimes (see ``benchmarks/README.md``): kornia runs batched float BCHW tensors on CPU or GPU and
is differentiable; scikit-image runs single normalized float HWC images on CPU in a Python loop.
The image pair is the shared seeded batch and a copy with clipped Gaussian noise (sigma 0.05).
SSIM parameters are the Wang et al. defaults in both: an 11x11 Gaussian window with sigma 1.5 and
population covariance (``gaussian_weights=True, use_sample_covariance=False``). kornia's ``ssim``
returns the per-pixel map and scikit-image the mean, which costs one reduction more on kornia's
side if the caller wants a scalar. ``mean_iou`` scores ``(B, H, W)`` label maps over 21 classes;
scikit-image has no counterpart. Throughput is image pairs per second (``img/s``).

Usage:
    python benchmarks/metrics/flagship.py --batches 1,8,32 --size 256 --device cpu
    python benchmarks/metrics/flagship.py --device cuda --compile --json metrics_cuda.json
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

import kornia.metrics as KMt

OPS = ("psnr", "ssim", "mean_iou")
WINDOW, SIGMA, NUM_CLASSES = 11, 1.5, 21


def build_ops(
    b: int,
    h: int,
    w: int,
    device: torch.device,
    dtype: torch.dtype,
    do_compile: bool,
    skm: Optional[ModuleType],
    skip_compile: frozenset[str] = frozenset(),
    selected: Optional[frozenset[str]] = None,
) -> tuple[dict[str, dict[str, Backend]], dict[str, str]]:
    """Build {op: {backend: zero-arg callable}}; each callable scores the whole batch once."""
    _, batch_f = image_batch(b, h, w, device, dtype)
    gen = torch.Generator().manual_seed(0)
    noisy = (batch_f.cpu().float() + 0.05 * torch.randn(batch_f.shape, generator=gen)).clamp(0, 1)
    noisy_f = noisy.to(device=device, dtype=dtype)
    refs_np = [im.permute(1, 2, 0).numpy() for im in batch_f.cpu().float()]
    tests_np = [im.permute(1, 2, 0).numpy() for im in noisy]
    labels_true = torch.randint(0, NUM_CLASSES, (b, h, w), generator=gen)
    # 80% of the predicted labels agree with the target, so every class has a non-trivial IoU.
    keep = torch.rand(b, h, w, generator=gen) < 0.8
    labels_pred = torch.where(keep, labels_true, torch.randint(0, NUM_CLASSES, (b, h, w), generator=gen))

    kornia_row = KorniaRows(device, do_compile, skip_compile)
    ops: dict[str, dict[str, Backend]] = {}

    def include(name: str) -> bool:
        return selected is None or name in selected

    if include("psnr"):
        row = kornia_row("psnr", KMt.psnr, noisy_f, batch_f, 1.0)
        row["scikit-image"] = (
            (lambda: [skm.peak_signal_noise_ratio(r, t, data_range=1.0) for r, t in zip(refs_np, tests_np)])
            if skm
            else None
        )
        ops["psnr"] = row

    if include("ssim"):
        row = kornia_row("ssim", KMt.ssim, noisy_f, batch_f, WINDOW)
        row["scikit-image"] = (
            (
                lambda: [
                    skm.structural_similarity(
                        r,
                        t,
                        data_range=1.0,
                        channel_axis=-1,
                        gaussian_weights=True,
                        sigma=SIGMA,
                        use_sample_covariance=False,
                    )
                    for r, t in zip(refs_np, tests_np)
                ]
            )
            if skm
            else None
        )
        ops["ssim"] = row

    if include("mean_iou"):
        pred, target = labels_pred.to(device), labels_true.to(device)
        ops["mean_iou"] = kornia_row("mean_iou", KMt.mean_iou, pred, target, NUM_CLASSES)

    return ops, kornia_row.compile_failures


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_flagship_args(parser, ops=OPS)
    args = parser.parse_args()
    device, dtype, sync = setup_run(args)

    skm, skm_error = optional_import("skimage.metrics")

    meta = start_run(
        "flagship metrics",
        args,
        device,
        units="img/s",
        regimes=["kornia: batched float BCHW; scikit-image: float HWC per-image loop (CPU); img/s = image pairs/s"],
        missing=[("scikit-image", skm_error)] if skm is None else [],
    )
    backends = ["kornia (eager)", "kornia (compiled)", "scikit-image"]
    results = run_batch_sweep(
        batch_list(args),
        lambda b: build_ops(b, args.size, args.size, device, dtype, args.compile, skm, args.skip_compile_ops, args.ops),
        backends,
        row_fields=image_row_fields(args),
        sync=sync,
        units="img/s",
        min_run_time=args.min_run_time,
    )
    finish_run(args, "metrics", meta, results)


if __name__ == "__main__":
    main()
