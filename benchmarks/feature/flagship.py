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

"""Flagship feature benchmark: kornia.feature vs OpenCV, scikit-image and kornia-rs.

Covers the local-feature pipeline end to end — a corner response, a full detector + descriptor,
and descriptor matching — rather than every detector and descriptor the module exports:

===================  ===================================================  ==============================
kornia.feature       OpenCV (per-image loop, CPU)                         scikit-image (per-image loop)
===================  ===================================================  ==============================
``harris_response``  ``cv2.cornerHarris(blockSize=7, ksize=3, k=0.04)``   ``corner_harris(sigma=1)``
``gftt_response``    ``cv2.cornerMinEigenVal(blockSize=7, ksize=3)``      ``corner_shi_tomasi(sigma=1)``
``SIFTFeature``      ``cv2.SIFT_create(nfeatures=512).detectAndCompute``  —
``match_snn``        ``BFMatcher(NORM_L2).knnMatch(k=2)`` + ratio test    —
``match_mnn``        ``BFMatcher(NORM_L2, crossCheck=True).match``        —
``match_smnn``       two ``knnMatch(k=2)`` calls + ratio/mutual filters   —
``match_smnn_batched`` same per-pair OpenCV loop as ``match_smnn``        —
===================  ===================================================  ==============================

Regimes (see ``benchmarks/README.md``): kornia runs a batched float B1HW tensor on CPU or GPU and
is differentiable; OpenCV and scikit-image run single grayscale images on CPU in a Python loop,
their native regime — float32 for the corner responses, uint8 for SIFT. The input is Gaussian-blurred
noise rather than white noise, so detectors see blob structure instead of a response at every
pixel. kornia's responses weight the structure tensor with a 7x7 Gaussian (sigma 1), which
scikit-image matches; OpenCV sums a 7x7 box. kornia's ``SIFTFeature`` and OpenCV's SIFT both keep
at most 512 features per image, but their scale spaces, thresholds and orientation handling differ;
the row compares complete detect+describe calls, not identical keypoints. kornia's scale-space
detector accepts one image per call, so its SIFT row loops over the batch like OpenCV's. For accuracy on real
images use ``local_features.py`` and ``sift_scale_space.py``; learned detectors and descriptors
(DISK, ALIKED, HardNet, ...) need downloaded weights and are measured there too.

kornia-rs 0.2+ adds a CPU SIFT detect+describe loop with a persistent detector and the same
512-feature budget. Its prebuilt float32 HWC1 inputs contain the same rounded 0..255 pixels
as OpenCV's uint8 inputs; normalizing them to 0..1 would change SIFT's contrast threshold.
It returns native host keypoints and float32 SIFT descriptors, like OpenCV; Kornia defaults to
RootSIFT normalization. The column stays on CPU when
Kornia runs on CUDA or MPS; GPU residency and descriptor matching are not measured here.

Matching rows share the same 2048-by-2048 unit-normalized 128-d descriptor pairs: the second
set is a seeded noisy permutation of the first, so the ratio tests accept and reject matches.
SNN uses an inclusive 0.8 ratio test, MNN uses mutual nearest neighbors, and SMNN requires both
mutuality and the ratio test in both directions. Kornia's unbatched functions and OpenCV loop
over pairs in Python; ``match_smnn_batched`` processes all pairs together (distance storage
scales as B*2048*2048). OpenCV includes Python ratio filtering and mutual-pair intersection;
its inputs are prebuilt float32 arrays. Matches have the same index semantics away from distance
ties; native output containers and quality scores differ. Throughput counts images for the
first three rows and descriptor-set pairs for every matching row; JSON units are ``items/s``.

Usage:
    python benchmarks/feature/flagship.py --batches 1,8,32 --size 256 --device cpu
    python benchmarks/feature/flagship.py --device cuda --compile --json feature_cuda.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, Optional

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
    kornia_rs_fn,
    optional_import,
    run_batch_sweep,
    setup_run,
    start_run,
)

import kornia.feature as KF
from kornia.filters import gaussian_blur2d

OPS = ("harris_response", "gftt_response", "SIFTFeature", "match_snn", "match_mnn", "match_smnn", "match_smnn_batched")
NUM_FEATURES, NUM_DESCRIPTORS, DESCRIPTOR_DIM, SNN_RATIO = 512, 2048, 128, 0.8


def blob_images(b: int, h: int, w: int) -> torch.Tensor:
    """Seeded Gaussian-blurred noise in [0, 1], ``(B, 1, H, W)`` float32 on CPU."""
    gen = torch.Generator().manual_seed(0)
    noise = torch.rand(b, 1, h, w, generator=gen)
    blurred = gaussian_blur2d(noise, (13, 13), (3.0, 3.0))
    lo = blurred.amin(dim=(-2, -1), keepdim=True)
    hi = blurred.amax(dim=(-2, -1), keepdim=True)
    return (blurred - lo) / (hi - lo)


def descriptor_pairs(b: int) -> tuple[torch.Tensor, torch.Tensor]:
    """``b`` pairs of unit descriptor sets; the second is a noisy permutation of the first, so matches exist."""
    gen = torch.Generator().manual_seed(0)
    d1 = torch.nn.functional.normalize(torch.randn(b, NUM_DESCRIPTORS, DESCRIPTOR_DIM, generator=gen), dim=-1)
    perm = torch.randperm(NUM_DESCRIPTORS, generator=gen)
    noisy = d1[:, perm] + 0.1 * torch.randn(b, NUM_DESCRIPTORS, DESCRIPTOR_DIM, generator=gen)
    return d1, torch.nn.functional.normalize(noisy, dim=-1)


def match_pairs(d1: torch.Tensor, d2: torch.Tensor, name: str = "match_snn") -> list[tuple[torch.Tensor, torch.Tensor]]:
    matcher = getattr(KF, name)
    return [
        matcher(d1[i], d2[i]) if name == "match_mnn" else matcher(d1[i], d2[i], SNN_RATIO) for i in range(d1.shape[0])
    ]


def opencv_matches(matcher: Any, d1: np.ndarray, d2: np.ndarray, name: str) -> list[Any]:
    """Native BF matching plus inclusive ratio filtering and, for SMNN, reverse consistency."""
    if name == "match_mnn":
        return list(matcher.match(d1, d2))
    forward = matcher.knnMatch(d1, d2, k=2)
    # Zero second-neighbor distance is an undefined ratio, rejected by Kornia too.
    accepted = [
        pair[0]
        for pair in forward
        if len(pair) == 2 and pair[1].distance > 0 and pair[0].distance <= SNN_RATIO * pair[1].distance
    ]
    if name == "match_snn":
        return accepted
    reverse = matcher.knnMatch(d2, d1, k=2)
    reverse_pairs = {
        (pair[0].trainIdx, pair[0].queryIdx)
        for pair in reverse
        if len(pair) == 2 and pair[1].distance > 0 and pair[0].distance <= SNN_RATIO * pair[1].distance
    }
    return [match for match in accepted if (match.queryIdx, match.trainIdx) in reverse_pairs]


def build_ops(
    b: int,
    h: int,
    w: int,
    device: torch.device,
    dtype: torch.dtype,
    do_compile: bool,
    cv2: Optional[ModuleType],
    skf: Optional[ModuleType],
    skip_compile: frozenset[str] = frozenset(),
    selected: Optional[frozenset[str]] = None,
) -> tuple[dict[str, dict[str, Backend]], dict[str, str]]:
    """Build {op: {backend: zero-arg callable}}; each callable processes the whole batch once."""
    gray = blob_images(b, h, w)
    gray_f = gray.to(device=device, dtype=dtype)
    imgs_f32 = [np.ascontiguousarray(g[0].numpy()) for g in gray]
    imgs_u8 = [(im * 255).round().astype(np.uint8) for im in imgs_f32]
    d1, d2 = descriptor_pairs(b)
    d1_dev, d2_dev = d1.to(device=device, dtype=dtype), d2.to(device=device, dtype=dtype)
    d1_np, d2_np = d1.numpy(), d2.numpy()

    kornia_row = KorniaRows(device, do_compile, skip_compile)
    ops: dict[str, dict[str, Backend]] = {}

    def include(name: str) -> bool:
        return selected is None or name in selected

    if include("harris_response"):
        row = kornia_row("harris_response", KF.harris_response, gray_f, 0.04)
        row["opencv"] = (lambda: [cv2.cornerHarris(im, 7, 3, 0.04) for im in imgs_f32]) if cv2 else None
        row["scikit-image"] = (
            (lambda: [skf.corner_harris(im, method="k", k=0.04, sigma=1) for im in imgs_f32]) if skf else None
        )
        ops["harris_response"] = row

    if include("gftt_response"):
        row = kornia_row("gftt_response", KF.gftt_response, gray_f)
        row["opencv"] = (lambda: [cv2.cornerMinEigenVal(im, 7, 3) for im in imgs_f32]) if cv2 else None
        row["scikit-image"] = (lambda: [skf.corner_shi_tomasi(im, sigma=1) for im in imgs_f32]) if skf else None
        ops["gftt_response"] = row

    if include("SIFTFeature"):
        sift = KF.SIFTFeature(num_features=NUM_FEATURES, device=device).to(dtype=dtype).eval()

        @torch.no_grad()
        def sift_each(images: torch.Tensor) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
            return [sift(images[i : i + 1]) for i in range(images.shape[0])]

        row = kornia_row("SIFTFeature", sift_each, gray_f)
        if cv2:
            cv_sift = cv2.SIFT_create(nfeatures=NUM_FEATURES)
            row["opencv"] = lambda: [cv_sift.detectAndCompute(im, None) for im in imgs_u8]
        rs_sift_type = kornia_rs_fn("Sift")
        row["kornia-rs"] = None
        if rs_sift_type is not None:
            rs_sift = rs_sift_type(n_features=NUM_FEATURES)
            rs_images = [np.ascontiguousarray(im[..., None], dtype=np.float32) for im in imgs_u8]
            row["kornia-rs"] = lambda: [rs_sift.detect_and_compute(im) for im in rs_images]
        ops["SIFTFeature"] = row

    for name in ("match_snn", "match_mnn", "match_smnn", "match_smnn_batched"):
        if not include(name):
            continue
        if name == "match_smnn_batched":
            row = kornia_row(name, KF.match_smnn_batched, d1_dev, d2_dev, SNN_RATIO)
        else:
            row = kornia_row(name, match_pairs, d1_dev, d2_dev, name)
        if cv2:
            matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=name == "match_mnn")

            def cv_match_pairs(matcher: Any = matcher, name: str = name) -> list[list[Any]]:
                return [opencv_matches(matcher, d1_np[i], d2_np[i], name) for i in range(b)]

            row["opencv"] = cv_match_pairs
        ops[name] = row

    return ops, kornia_row.compile_failures


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_flagship_args(parser, ops=OPS)
    args = parser.parse_args()
    device, dtype, sync = setup_run(args)

    cv2, cv2_error = optional_import("cv2")
    skf, skf_error = optional_import("skimage.feature")
    krs, krs_error = optional_import("kornia_rs")

    libs = [(skf, "scikit-image", skf_error), (cv2, "opencv", cv2_error), (krs, "kornia-rs", krs_error)]
    meta = start_run(
        "flagship feature",
        args,
        device,
        units="items/s",
        regimes=[
            (
                "kornia: batched float B1HW; opencv/scikit-image: grayscale per-image loop (CPU); "
                "items = images, except matching rows = descriptor-set pairs; descriptors float32 in OpenCV"
            ),
            "kornia-rs SIFT: CPU per-image loop, prebuilt float32 HWC1 0..255, persistent detector, 512 features",
            "matching: same 2048x2048 128-d pairs; inclusive 0.8 SNN/SMNN ratios; MNN uses cross-check",
            "match_smnn_batched batches pairs in Kornia; all other matching cells loop over pairs",
        ],
        missing=[(name, err) for lib, name, err in libs if lib is None]
        + (
            [("kornia-rs/SIFTFeature", "Sift unavailable")]
            if krs is not None and (args.ops is None or "SIFTFeature" in args.ops) and kornia_rs_fn("Sift") is None
            else []
        ),
    )
    backends = ["kornia (eager)", "kornia (compiled)", "scikit-image", "opencv", "kornia-rs"]
    results = run_batch_sweep(
        batch_list(args),
        lambda b: build_ops(
            b, args.size, args.size, device, dtype, args.compile, cv2, skf, args.skip_compile_ops, args.ops
        ),
        backends,
        row_fields=image_row_fields(args),
        sync=sync,
        units="items/s",
        min_run_time=args.min_run_time,
    )
    finish_run(args, "feature", meta, results)


if __name__ == "__main__":
    main()
