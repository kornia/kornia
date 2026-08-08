"""Flagship filters benchmark: kornia.filters vs OpenCV, albumentations, torchvision v2, kornia-rs.

Covers the core image filters with fixed, identical parameters across backends (equal footing;
5×5 kernels, sigma 1.5 where applicable):

================  ===============================================  ==============================
kornia.filters    OpenCV (uint8 HWC, per-image Python loop)        others
================  ===============================================  ==============================
gaussian_blur2d   ``cv2.GaussianBlur``                             albumentations, tvf, kornia-rs
sobel             ``cv2.magnitude(Sobel(dx), Sobel(dy))``          —
laplacian         ``cv2.Laplacian``                                —
median_blur       ``cv2.medianBlur``                               albumentations
box_blur          ``cv2.blur``                                     albumentations
canny             ``cv2.Canny`` (on grayscale)                     —
================  ===============================================  ==============================

Regimes (see ``benchmarks/README.md``): kornia/torchvision run a batched float BCHW tensor on CPU
or GPU and kornia is differentiable; OpenCV/albumentations/kornia-rs run single uint8 HWC images
on CPU in a Python loop — their native regime. albumentations wraps OpenCV in its transform-class
API (constructed once, called with fixed parameters). Canny thresholds are each library's standard
defaults — kornia 0.1/0.2 on normalized float gradients, OpenCV 100/200 on uint8 gradients — the
domains differ, so that row compares regimes, not identical outputs. kornia's canny converts to
grayscale internally per its definition; the OpenCV canny loop therefore includes ``cv2.cvtColor``
(sobel runs per-channel in both). Throughput is img/s.

Usage:
    python benchmarks/filters/flagship.py --batches 1,8,32 --size 256 --device cpu
    python benchmarks/filters/flagship.py --device cuda --compile --json filters_cuda.json
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

import kornia.filters as KF

Backend = Optional[Callable[[], object]]


def krs_fn(name: str) -> Optional[Callable[..., object]]:
    """Resolve a kornia-rs function across wheel layouts (imgproc submodule vs top-level)."""
    try:
        import kornia_rs
    except ImportError:
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
) -> tuple[dict[str, dict[str, Backend]], dict[str, str]]:
    """Build {op: {backend: zero-arg callable}} with identical filter parameters per backend."""
    rng = np.random.default_rng(0)
    imgs_u8 = [(rng.random((h, w, 3)) * 255).astype(np.uint8) for _ in range(b)]
    batch_f = (
        torch.stack([torch.from_numpy(im).permute(2, 0, 1) for im in imgs_u8]).to(device=device, dtype=dtype).div(255)
    )

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

    def alb(t: object) -> Backend:
        return lambda: [t(image=im)["image"] for im in imgs_u8]

    ops: dict[str, dict[str, Backend]] = {}

    krs_gaussian = krs_fn("gaussian_blur")
    row = kornia_row("gaussian_blur2d", lambda: KF.gaussian_blur2d(batch_f, (5, 5), (1.5, 1.5)))
    row["opencv"] = (lambda: [cv2.GaussianBlur(im, (5, 5), 1.5) for im in imgs_u8]) if cv2 else None
    row["albumentations"] = alb(A.GaussianBlur(blur_limit=(5, 5), sigma_limit=(1.5, 1.5), p=1.0)) if A else None
    row["torchvision v2"] = (lambda: tvf.gaussian_blur(batch_f, [5, 5], [1.5, 1.5])) if tvf else None
    row["kornia-rs"] = (lambda: [krs_gaussian(im, (5, 5), (1.5, 1.5)) for im in imgs_u8]) if krs_gaussian else None
    ops["gaussian_blur2d"] = row

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
    row["kornia-rs"] = None
    ops["sobel"] = row

    row = kornia_row("laplacian", lambda: KF.laplacian(batch_f, 5))
    row["opencv"] = (lambda: [cv2.Laplacian(im, cv2.CV_32F, ksize=5) for im in imgs_u8]) if cv2 else None
    row["albumentations"] = None
    row["torchvision v2"] = None
    row["kornia-rs"] = None
    ops["laplacian"] = row

    row = kornia_row("median_blur", lambda: KF.median_blur(batch_f, (5, 5)))
    row["opencv"] = (lambda: [cv2.medianBlur(im, 5) for im in imgs_u8]) if cv2 else None
    row["albumentations"] = alb(A.MedianBlur(blur_limit=(5, 5), p=1.0)) if A else None
    row["torchvision v2"] = None
    row["kornia-rs"] = None
    ops["median_blur"] = row

    row = kornia_row("box_blur", lambda: KF.box_blur(batch_f, (5, 5)))
    row["opencv"] = (lambda: [cv2.blur(im, (5, 5)) for im in imgs_u8]) if cv2 else None
    row["albumentations"] = alb(A.Blur(blur_limit=(5, 5), p=1.0)) if A else None
    row["torchvision v2"] = None
    row["kornia-rs"] = None
    ops["box_blur"] = row

    row = kornia_row("canny", lambda: KF.canny(batch_f))
    row["opencv"] = (
        (lambda: [cv2.Canny(cv2.cvtColor(im, cv2.COLOR_RGB2GRAY), 100, 200) for im in imgs_u8]) if cv2 else None
    )
    row["albumentations"] = None
    row["torchvision v2"] = None
    row["kornia-rs"] = None
    ops["canny"] = row

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
    except ImportError:
        cv2 = None
    try:
        import albumentations as A
    except ImportError:
        A = None
    try:
        import torchvision.transforms.v2.functional as tvf
    except ImportError:
        tvf = None

    meta = run_metadata(device)
    print(f"# flagship filters benchmark — commit {meta['git_commit']} — {platform.platform()}")
    if device.type == "cuda":
        print(f"# CUDA device: {meta['cuda_device']}")
    print(f"# device={device}, dtype={args.dtype}, threads={args.threads}, size={args.size} — throughput img/s")
    print("# kornia/torchvision: batched float BCHW; albumentations/opencv/kornia-rs: uint8 HWC per-image loop (CPU)")
    skips = [(cv2, "opencv"), (A, "albumentations"), (tvf, "torchvision"), (krs_fn("gaussian_blur"), "kornia-rs filters")]
    for present, name in skips:
        if present is None:
            print(f"# NOTE: {name} not available — its column is skipped")

    backends = ["kornia (eager)", "kornia (compiled)", "torchvision v2", "albumentations", "opencv", "kornia-rs"]
    results = run_batch_sweep(
        [int(x) for x in args.batches.split(",")],
        lambda b: build_ops(b, args.size, args.size, device, dtype, args.compile, cv2, A, tvf),
        backends,
        row_fields=lambda b: {"height": args.size, "width": args.size, "dtype": args.dtype},
        sync=sync,
    )
    if args.json:
        out = save_json(args.json, meta, results)
        print(f"# results written to {out}")


if __name__ == "__main__":
    main()
