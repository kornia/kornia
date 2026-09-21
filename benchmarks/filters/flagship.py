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

"""Flagship filters benchmark: kornia.filters vs native CPU image libraries.

Covers the core image filters with fixed nominal parameters across backends
(5x5 kernels, sigma 1.5 where applicable; semantic differences are described below):

========================  =================================  ====================================================
kornia.filters            OpenCV (uint8 HWC per-image loop)  others
========================  =================================  ====================================================
gaussian_blur2d           cv2.GaussianBlur                   albumentations, torchvision, skimage, kornia-rs, PIL
sobel                     cv2.Sobel(dx/dy) -> cv2.magnitude  skimage, kornia-rs
laplacian                 cv2.Laplacian                      skimage
median_blur               cv2.medianBlur                     albumentations, skimage, kornia-rs, PIL
box_blur                  cv2.blur                           albumentations, skimage rank, kornia-rs, PIL
unsharp_mask              —                                  albumentations, skimage
bilateral_blur            —                                  skimage
bilateral_blur_grayscale  —                                  skimage, kornia-rs
motion_blur               —                                  albumentations
otsu_threshold            —                                  skimage
canny                     cv2.Canny (on grayscale)           —
========================  =================================  ====================================================

Regimes (see ``benchmarks/README.md``): kornia/torchvision run a batched float BCHW tensor on CPU
or GPU and kornia is differentiable; OpenCV/albumentations/kornia-rs/PIL run single
uint8 HWC images on CPU in a Python loop — their native regime. kornia-rs Sobel takes normalized
float32 HWC input, its public f32 API. albumentations wraps OpenCV in its
transform-class API (constructed once, called with fixed parameters). scikit-image runs normalized
float32 HWC images per-image, except rank mean which uses uint8. Its Sobel, Laplace, and rank mean
calls run independently per channel. PIL uses ``BoxBlur(2)``
(5x5 box) and ``MedianFilter(5)``; border handling differs from Kornia. Its
``GaussianBlur(radius=1.5)`` approximates a true Gaussian with repeated box passes, so the sigma is
matched in spirit only.
Canny thresholds are each library's standard
defaults — kornia 0.1/0.2 on normalized float gradients, OpenCV 100/200 on uint8 gradients — the
domains differ, so that row compares regimes, not identical outputs. kornia's canny converts to
grayscale internally per its definition; the OpenCV canny loop therefore includes ``cv2.cvtColor``
(sobel runs per-channel in both). Throughput is img/s.

The Laplacian columns are different operators: Kornia uses an L1-normalized all-ones window with
center ``1 - kernel_area``; OpenCV sums second Sobel derivatives and scikit-image's kernel differs
as well. Sobel normalization differs across libraries. Median uses zero padding in
Kornia/scikit-image and replicated borders in OpenCV. The scikit-image Gaussian uses ``mode='mirror'`` (the
SciPy mode matching PyTorch reflect padding) and ``truncate=2 / 1.5`` for a 5x5 support. Its rank
mean is a uint8 footprint-based box filter with truncated boundary footprints. These are
native-regime throughput comparisons, not output-equivalent comparisons.

kornia-rs 0.1.14 exposes Gaussian and box blur; 0.1.15rc5 adds median
blur and Sobel to the Python API. The median operator accepts only 3x3 or 5x5 uint8 images and
uses replicated borders. Its Sobel is f32 magnitude; its normalization and borders can differ
from Kornia. kornia-rs also exposes a bilateral filter only for single-channel uint8 images. The
grayscale bilateral row converts the common RGB image to precomputed mean grayscale before timing:
Kornia and scikit-image use normalized float, while kornia-rs receives the rounded uint8 value and
uses ``sigma_color=25.5`` to represent 0.1 in that domain. kornia-rs matches OpenCV's circular
diameter-5 neighborhood; Kornia's 5x5 filter has square support.

Albumentations ``UnsharpMask`` is an unsharp-mask algorithm but is not Kornia-equivalent: it clips
and blends through a blurred threshold mask, while Kornia is exactly ``2 * input - gaussian_blur``.
``MotionBlur`` has the same fixed nominal kernel, angle, and centered direction, but rasterization,
angle convention, and borders can differ. scikit-image's unsharp mask has ``amount=1`` but uses
its default 13x13 Gaussian support at sigma 1.5, so it is not a 5x5 counterpart. Its bilateral
filter uses an L2 color-distance lookup rather than Kornia's default L1 color distance. There is no
scikit-image counterpart for joint bilateral or guided filtering. The scikit-image Otsu adapter
computes both the threshold and full zeroed output, matching Kornia's default return form.

The ``box_blur`` row keeps ``separable=False`` to measure dense filtering.
``box_blur_separable`` measures the default separable implementation with the same kernel.
The remaining Kornia-only guided-blur row is useful for eager/compiled and revision-to-revision
comparisons, but does not claim output-equivalent performance against another library.

Usage:
    python benchmarks/filters/flagship.py --batches 1,8,32 --size 256 --device cpu
    python benchmarks/filters/flagship.py --device cuda --compile --json filters_cuda.json
    python benchmarks/filters/flagship.py --ops gaussian_blur2d,unsharp_mask --compile
"""

from __future__ import annotations

import argparse
import importlib
import platform
import sys
from pathlib import Path
from types import ModuleType
from typing import Callable, Optional

import numpy as np
import torch

# Direct script execution puts benchmarks/filters, not the checkout, on sys.path.
# Prefer this tree to a wheel or an editable installation of another checkout.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import (
    add_contribute_args,
    collect_load_metrics,
    contribute_result,
    print_preflight,
    run_batch_sweep,
    run_metadata,
    save_json,
    versions_line,
    warm_up_cpu,
)

import kornia.filters as KF

Backend = Optional[Callable[[], object]]

AVAILABLE_OPS = (
    "gaussian_blur2d",
    "sobel",
    "laplacian",
    "median_blur",
    "box_blur",
    "box_blur_separable",
    "canny",
    "unsharp_mask",
    "guided_blur_grayscale",
    "bilateral_blur",
    "bilateral_blur_grayscale",
    "motion_blur",
    "otsu_threshold",
)


def optional_import(name: str) -> tuple[Optional[ModuleType], Optional[str]]:
    """Import an optional benchmark dependency and retain a concise failure reason."""
    try:
        return importlib.import_module(name), None
    except Exception as error:
        return None, f"{type(error).__name__}: {error}"


def parse_ops(value: str) -> frozenset[str] | None:
    """Parse a comma-separated operation selection, validating benchmark row names."""
    selected = frozenset(name.strip() for name in value.split(",") if name.strip())
    unknown = selected.difference(AVAILABLE_OPS)
    if unknown:
        available = ", ".join(AVAILABLE_OPS)
        raise argparse.ArgumentTypeError(f"unknown operation(s): {', '.join(sorted(unknown))}. Available: {available}")
    return selected or None


def build_specialized_ops(
    batch_f: torch.Tensor,
    imgs_u8: list[np.ndarray],
    imgs_f: list[np.ndarray],
    imgs_gray_f: list[np.ndarray],
    imgs_gray_u8: list[np.ndarray],
    kornia_row: Callable[[str, Callable[[], object]], dict[str, Backend]],
    selected_ops: frozenset[str] | None,
    A: Optional[ModuleType],
    skf: Optional[ModuleType],
    skr: Optional[ModuleType],
) -> dict[str, dict[str, Backend]]:
    """Build specialized filters, including native-regime baselines where available."""

    def include(name: str) -> bool:
        return selected_ops is None or name in selected_ops

    def alb(t: object) -> Backend:
        return lambda: [t(image=im)["image"] for im in imgs_u8]

    ops: dict[str, dict[str, Backend]] = {}

    if include("unsharp_mask"):
        row = kornia_row("unsharp_mask", lambda: KF.unsharp_mask(batch_f, (5, 5), (1.5, 1.5)))
        row["albumentations"] = (
            alb(A.UnsharpMask(blur_limit=(5, 5), sigma_limit=(1.5, 1.5), alpha=(1.0, 1.0), threshold=0, p=1.0))
            if A
            else None
        )
        # Use a positive axis: skimage 0.26's unsharp slicing mishandles -1.
        row["scikit-image"] = (
            (
                lambda: [
                    skf.unsharp_mask(im, radius=1.5, amount=1.0, preserve_range=True, channel_axis=2) for im in imgs_f
                ]
            )
            if skf
            else None
        )
        ops["unsharp_mask"] = row
    if include("guided_blur_grayscale"):
        gray_guidance = batch_f.mean(dim=1, keepdim=True)
        ops["guided_blur_grayscale"] = kornia_row(
            "guided_blur_grayscale", lambda: KF.guided_blur(gray_guidance, batch_f, (5, 5), 0.1)
        )
    if include("bilateral_blur"):
        row = kornia_row("bilateral_blur", lambda: KF.bilateral_blur(batch_f, (5, 5), 0.1, (1.5, 1.5)))
        row["scikit-image"] = (
            (
                lambda: [
                    skr.denoise_bilateral(
                        im, win_size=5, sigma_color=0.1, sigma_spatial=1.5, mode="reflect", channel_axis=-1
                    )
                    for im in imgs_f
                ]
            )
            if skr
            else None
        )
        ops["bilateral_blur"] = row
    if include("bilateral_blur_grayscale"):
        krs_bilateral = krs_fn("bilateral_filter")
        gray_batch = batch_f.mean(dim=1, keepdim=True)
        row = kornia_row(
            "bilateral_blur_grayscale",
            lambda: KF.bilateral_blur(gray_batch, (5, 5), 0.1, (1.5, 1.5)),
        )
        row["kornia-rs"] = (lambda: [krs_bilateral(im, 5, 25.5, 1.5) for im in imgs_gray_u8]) if krs_bilateral else None
        row["scikit-image"] = (
            (
                lambda: [
                    skr.denoise_bilateral(
                        im, win_size=5, sigma_color=0.1, sigma_spatial=1.5, mode="reflect", channel_axis=None
                    )
                    for im in imgs_gray_f
                ]
            )
            if skr
            else None
        )
        ops["bilateral_blur_grayscale"] = row
    if include("motion_blur"):
        row = kornia_row("motion_blur", lambda: KF.motion_blur(batch_f, 5, 45.0, 0.0))
        row["albumentations"] = (
            alb(
                A.MotionBlur(
                    blur_limit=(5, 5),
                    allow_shifted=False,
                    angle_range=(45.0, 45.0),
                    direction_range=(0.0, 0.0),
                    p=1.0,
                )
            )
            if A
            else None
        )
        ops["motion_blur"] = row
    if include("otsu_threshold"):
        gray_batch = batch_f.mean(dim=1, keepdim=True)
        row = kornia_row("otsu_threshold", lambda: KF.otsu_threshold(gray_batch))

        def skimage_otsu(gray: np.ndarray) -> tuple[np.ndarray, float]:
            threshold = skf.threshold_otsu(gray)
            return np.where(gray > threshold, gray, 0), threshold

        row["scikit-image"] = (lambda: [skimage_otsu(gray) for gray in imgs_gray_f]) if skf else None
        ops["otsu_threshold"] = row

    return ops


def krs_fn(name: str) -> Optional[Callable[..., object]]:
    """Resolve a kornia-rs function across wheel layouts (imgproc submodule vs top-level)."""
    try:
        import kornia_rs
    except Exception:
        return None
    ns = getattr(kornia_rs, "imgproc", kornia_rs)
    return getattr(ns, name, getattr(kornia_rs, name, None))


def build_ops(
    b: int,
    h: int,
    w: int,
    device: torch.device,
    dtype: torch.dtype,
    do_compile: bool,
    cv2: Optional[ModuleType],
    A: Optional[ModuleType],
    tvf: Optional[ModuleType],
    pil: Optional[ModuleType],
    pilf: Optional[ModuleType],
    skip_compile: frozenset[str] = frozenset(),
    selected_ops: frozenset[str] | None = None,
    skf: Optional[ModuleType] = None,
    skr: Optional[ModuleType] = None,
) -> tuple[dict[str, dict[str, Backend]], dict[str, str]]:
    """Build {op: {backend: zero-arg callable}} with the documented per-backend regimes."""
    rng = np.random.default_rng(0)
    imgs_u8 = [(rng.random((h, w, 3)) * 255).astype(np.uint8) for _ in range(b)]
    imgs_f = [im.astype(np.float32) / 255.0 for im in imgs_u8]
    imgs_gray_f = [im.mean(axis=-1) for im in imgs_f]
    imgs_gray_u8 = [np.rint(im.mean(axis=-1)).astype(np.uint8)[..., None] for im in imgs_u8]
    batch_f = (
        torch.stack([torch.from_numpy(im).permute(2, 0, 1) for im in imgs_u8]).to(device=device, dtype=dtype).div(255)
    )

    compile_failures: dict[str, str] = {}

    def kornia_row(label: str, fn: Callable[[], object]) -> dict[str, Backend]:
        row: dict[str, Backend] = {"kornia (eager)": fn}
        if do_compile and label not in skip_compile:
            torch._dynamo.reset()
            compiled = torch.compile(fn)
            try:
                compiled()  # warmup: compile + autotune before the timed region
                if device.type == "cuda":
                    torch.cuda.synchronize()  # surface async kernel faults HERE, not at the next op
                row["kornia (compiled)"] = compiled
            except Exception as e:
                errors = str(e)
                if device.type == "cuda":
                    try:
                        torch.cuda.synchronize()  # a FAILED warmup may still have launched kernels
                    except Exception as sync_err:
                        errors += " | " + str(sync_err)
                if "illegal memory access" in errors:
                    raise SystemExit(
                        f"FATAL: CUDA context poisoned during torch.compile warmup of '{label}' "
                        "(illegal memory access); no later measurement would be trustworthy. "
                        f"Rerun with --skip-compile-ops {label} to keep it eager-only, or without "
                        "--compile; CUDA_LAUNCH_BLOCKING=1 localizes the kernel."
                    ) from e
                row["kornia (compiled)"] = None
                compile_failures[label] = type(e).__name__
        return row

    def alb(t: object) -> Backend:
        return lambda: [t(image=im)["image"] for im in imgs_u8]

    def skimage_per_channel(fn: Callable[[np.ndarray], np.ndarray], images: list[np.ndarray]) -> Backend:
        return lambda: [np.stack([fn(im[..., channel]) for channel in range(im.shape[-1])], axis=-1) for im in images]

    footprint_5 = np.ones((5, 5), dtype=bool)
    footprint_5_hwc = footprint_5[..., None]
    ops: dict[str, dict[str, Backend]] = {}

    def include(name: str) -> bool:
        return selected_ops is None or name in selected_ops

    if include("gaussian_blur2d"):
        krs_gaussian = krs_fn("gaussian_blur")
        row = kornia_row("gaussian_blur2d", lambda: KF.gaussian_blur2d(batch_f, (5, 5), (1.5, 1.5)))
        row["opencv"] = (lambda: [cv2.GaussianBlur(im, (5, 5), 1.5) for im in imgs_u8]) if cv2 else None
        row["albumentations"] = alb(A.GaussianBlur(blur_limit=(5, 5), sigma_limit=(1.5, 1.5), p=1.0)) if A else None
        row["torchvision v2"] = (lambda: tvf.gaussian_blur(batch_f, [5, 5], [1.5, 1.5])) if tvf else None
        row["kornia-rs"] = (lambda: [krs_gaussian(im, (5, 5), (1.5, 1.5)) for im in imgs_u8]) if krs_gaussian else None
        row["PIL"] = (
            (lambda: [pil.fromarray(im).filter(pilf.GaussianBlur(radius=1.5)) for im in imgs_u8])
            if pil is not None and pilf is not None
            else None
        )
        row["scikit-image"] = (
            (
                lambda: [
                    skf.gaussian(im, sigma=1.5, mode="mirror", truncate=2.0 / 1.5, preserve_range=True, channel_axis=-1)
                    for im in imgs_f
                ]
            )
            if skf
            else None
        )
        ops["gaussian_blur2d"] = row

    if include("sobel"):
        krs_sobel = krs_fn("sobel")
        row = kornia_row("sobel", lambda: KF.sobel(batch_f))
        if cv2:

            def cv_sobel_mag(im: np.ndarray) -> np.ndarray:
                dx = cv2.Sobel(im, cv2.CV_32F, 1, 0, ksize=3)
                dy = cv2.Sobel(im, cv2.CV_32F, 0, 1, ksize=3)
                return cv2.magnitude(dx, dy)

            row["opencv"] = lambda: [cv_sobel_mag(im) for im in imgs_u8]
        else:
            row["opencv"] = None
        row["albumentations"] = None
        row["torchvision v2"] = None
        row["kornia-rs"] = (lambda: [krs_sobel(im, 3) for im in imgs_f]) if krs_sobel else None
        row["scikit-image"] = skimage_per_channel(skf.sobel, imgs_f) if skf else None
        ops["sobel"] = row

    if include("laplacian"):
        row = kornia_row("laplacian", lambda: KF.laplacian(batch_f, 5))
        row["opencv"] = (lambda: [cv2.Laplacian(im, cv2.CV_32F, ksize=5) for im in imgs_u8]) if cv2 else None
        row["albumentations"] = None
        row["torchvision v2"] = None
        row["kornia-rs"] = None
        row["scikit-image"] = skimage_per_channel(lambda image: skf.laplace(image, ksize=5), imgs_f) if skf else None
        ops["laplacian"] = row

    if include("median_blur"):
        krs_median = krs_fn("median_blur")
        row = kornia_row("median_blur", lambda: KF.median_blur(batch_f, (5, 5)))
        row["opencv"] = (lambda: [cv2.medianBlur(im, 5) for im in imgs_u8]) if cv2 else None
        row["albumentations"] = alb(A.MedianBlur(blur_limit=(5, 5), p=1.0)) if A else None
        row["torchvision v2"] = None
        row["kornia-rs"] = (lambda: [krs_median(im, 5) for im in imgs_u8]) if krs_median else None
        row["PIL"] = (
            (lambda: [pil.fromarray(im).filter(pilf.MedianFilter(5)) for im in imgs_u8])
            if pil is not None and pilf is not None
            else None
        )
        row["scikit-image"] = (
            (lambda: [skf.median(im, footprint=footprint_5_hwc, mode="constant", cval=0) for im in imgs_f])
            if skf
            else None
        )
        ops["median_blur"] = row

    if include("box_blur"):
        krs_box = krs_fn("box_blur")
        row = kornia_row("box_blur", lambda: KF.box_blur(batch_f, (5, 5), separable=False))
        row["opencv"] = (lambda: [cv2.blur(im, (5, 5)) for im in imgs_u8]) if cv2 else None
        row["albumentations"] = alb(A.Blur(blur_limit=(5, 5), p=1.0)) if A else None
        row["torchvision v2"] = None
        row["kornia-rs"] = (lambda: [krs_box(im, (5, 5)) for im in imgs_u8]) if krs_box else None
        row["PIL"] = (
            (lambda: [pil.fromarray(im).filter(pilf.BoxBlur(2)) for im in imgs_u8])
            if pil is not None and pilf is not None
            else None
        )
        row["scikit-image"] = (
            # The 2D rank implementation is faster than a 3D HWC footprint.
            skimage_per_channel(lambda image: skf.rank.mean(image, footprint=footprint_5), imgs_u8) if skf else None
        )
        ops["box_blur"] = row

    if include("box_blur_separable"):
        ops["box_blur_separable"] = kornia_row(
            "box_blur_separable", lambda: KF.box_blur(batch_f, (5, 5), separable=True)
        )

    if include("canny"):
        row = kornia_row("canny", lambda: KF.canny(batch_f))
        row["opencv"] = (
            (lambda: [cv2.Canny(cv2.cvtColor(im, cv2.COLOR_RGB2GRAY), 100, 200) for im in imgs_u8]) if cv2 else None
        )
        row["albumentations"] = None
        row["torchvision v2"] = None
        row["kornia-rs"] = None
        ops["canny"] = row

    ops.update(
        build_specialized_ops(
            batch_f, imgs_u8, imgs_f, imgs_gray_f, imgs_gray_u8, kornia_row, selected_ops, A, skf, skr
        )
    )

    return ops, compile_failures


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--batches", type=str, default="1,8,32", help="comma-separated batch sizes to sweep")
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--dtype", type=str, default="float32", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--compile", action="store_true", help="also time torch.compile'd kornia")
    parser.add_argument(
        "--ops",
        type=parse_ops,
        default=None,
        help="comma-separated benchmark rows to run (default: all)",
    )
    parser.add_argument(
        "--skip-compile-ops",
        type=str,
        default="",
        help="comma-separated op names to keep eager-only (workaround for faulting compiled kernels)",
    )
    parser.add_argument("--json", type=str, default=None, help="write machine-readable results to this path")
    add_contribute_args(parser)
    args = parser.parse_args()
    skip_compile = frozenset(s.strip() for s in args.skip_compile_ops.split(",") if s.strip())

    torch.set_num_threads(args.threads)
    warm_up_cpu()
    torch.manual_seed(0)
    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)
    sync = torch.mps.synchronize if device.type == "mps" else None  # Timer only syncs CUDA

    cv2, cv2_error = optional_import("cv2")
    A, albumentations_error = optional_import("albumentations")
    tvf, torchvision_error = optional_import("torchvision.transforms.v2.functional")
    pil, pil_error = optional_import("PIL.Image")
    pilf, pilf_error = optional_import("PIL.ImageFilter")
    skf, skimage_filters_error = optional_import("skimage.filters")
    skr, skimage_restoration_error = optional_import("skimage.restoration")

    meta = run_metadata(device)
    meta["load"] = collect_load_metrics()
    if args.contribute:
        print_preflight(meta["load"])
    print(f"# flagship filters benchmark — commit {meta['git_commit']} — {platform.platform()}")
    print(versions_line(meta))
    print(f"# kornia source: {Path(KF.__file__).resolve()}")
    if device.type == "cuda":
        print(f"# CUDA device: {meta['cuda_device']} (CUDA {meta['cuda_version']})")
    print(f"# device={device}, dtype={args.dtype}, threads={args.threads}, size={args.size} — throughput img/s")
    print(
        "# kornia/torchvision: batched float BCHW; "
        "albumentations/opencv/kornia-rs/PIL: uint8 HWC per-image loop (CPU), except kornia-rs Sobel float32; "
        "scikit-image: normalized float HWC per-image loop except uint8 rank mean"
    )
    unavailable = [
        (cv2, "opencv", cv2_error),
        (A, "albumentations", albumentations_error),
        (tvf, "torchvision", torchvision_error),
        (pil if pilf is not None else None, "PIL", pil_error or pilf_error),
    ]
    for present, name, error in unavailable:
        if present is None:
            print(f"# NOTE: {name} not available ({error}) — its column is skipped")
    if skf is None and skr is None:
        error = skimage_filters_error or skimage_restoration_error
        print(f"# NOTE: scikit-image not available ({error}) — its column is skipped")
    else:
        if skf is None:
            print(f"# NOTE: scikit-image filters not available ({skimage_filters_error}) — affected rows are skipped")
        if skr is None:
            print(
                f"# NOTE: scikit-image restoration not available ({skimage_restoration_error}) "
                "— bilateral rows are skipped"
            )
    if krs_fn("gaussian_blur") is None:
        print("# NOTE: kornia-rs filter APIs not available — its column is skipped")
    if skip_compile:
        print(f"# NOTE: --skip-compile-ops keeps eager-only: {', '.join(sorted(skip_compile))}")
    if args.ops:
        print(f"# selected rows: {', '.join(sorted(args.ops))}")

    backends = [
        "kornia (eager)",
        "kornia (compiled)",
        "torchvision v2",
        "albumentations",
        "scikit-image",
        "opencv",
        "kornia-rs",
        "PIL",
    ]
    results = run_batch_sweep(
        [int(x) for x in args.batches.split(",") if x.strip()],
        lambda b: build_ops(
            b,
            args.size,
            args.size,
            device,
            dtype,
            args.compile,
            cv2,
            A,
            tvf,
            pil,
            pilf,
            skip_compile,
            args.ops,
            skf,
            skr,
        ),
        backends,
        row_fields=lambda b: {"height": args.size, "width": args.size, "dtype": args.dtype},
        sync=sync,
    )
    if args.json:
        out = save_json(args.json, meta, results)
        print(f"# results written to {out}")
    if args.contribute:
        contribute_result(args.contribute, "filters", meta, results, slug_override=args.machine_slug)


if __name__ == "__main__":
    main()
