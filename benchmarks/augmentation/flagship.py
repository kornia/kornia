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
import platform
import random
import sys
from pathlib import Path
from types import ModuleType
from typing import Callable, Optional

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import run_batch_sweep, run_metadata, save_json

import kornia.augmentation as KA

Backend = Optional[Callable[[], object]]


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
) -> tuple[dict[str, dict[str, Backend]], dict[str, str]]:
    """Build {op: {backend: zero-arg callable}}; each callable transforms the whole batch once."""
    rng = np.random.default_rng(0)
    imgs_u8 = [(rng.random((h, w, 3)) * 255).astype(np.uint8) for _ in range(b)]
    batch_f = (
        torch.stack([torch.from_numpy(im).permute(2, 0, 1) for im in imgs_u8]).to(device=device, dtype=dtype).div(255)
    )

    compile_failures: dict[str, str] = {}

    def kornia_row(label: str, aug: torch.nn.Module) -> dict[str, Backend]:
        aug = aug.to(device)
        row: dict[str, Backend] = {"kornia (eager)": lambda: aug(batch_f)}
        if do_compile:
            torch._dynamo.reset()
            compiled = torch.compile(aug)
            try:
                compiled(batch_f)  # warmup: compile + autotune before the timed region
                row["kornia (compiled)"] = lambda: compiled(batch_f)
            except Exception as e:
                row["kornia (compiled)"] = None
                compile_failures[label] = type(e).__name__
        return row

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
    row["opencv"] = None
    ops["RandomAffine"] = row

    row = kornia_row("RandomPerspective", KA.RandomPerspective(0.5, p=1.0))
    row["torchvision v2"] = tv(T2.RandomPerspective(distortion_scale=0.5, p=1.0)) if T2 else None
    row["albumentations"] = alb(A.Perspective(scale=(0.05, 0.1), p=1.0)) if A else None
    row["opencv"] = None
    ops["RandomPerspective"] = row

    dst = (h // 2, w // 2)
    row = kornia_row("RandomResizedCrop", KA.RandomResizedCrop(dst))
    row["torchvision v2"] = tv(T2.RandomResizedCrop(dst, antialias=False)) if T2 else None
    row["albumentations"] = alb(A.RandomResizedCrop(size=dst, p=1.0)) if A else None
    row["opencv"] = None
    ops["RandomResizedCrop"] = row

    row = kornia_row("ColorJiggle", KA.ColorJiggle(0.2, 0.2, 0.2, 0.1, p=1.0))
    row["torchvision v2"] = tv(T2.ColorJitter(0.2, 0.2, 0.2, 0.1)) if T2 else None
    row["albumentations"] = alb(A.ColorJitter(0.2, 0.2, 0.2, 0.1, p=1.0)) if A else None
    row["opencv"] = None
    ops["ColorJiggle"] = row

    row = kornia_row("RandomGaussianBlur", KA.RandomGaussianBlur((5, 5), (0.1, 2.0), p=1.0))
    row["torchvision v2"] = tv(T2.GaussianBlur(5, sigma=(0.1, 2.0))) if T2 else None
    row["albumentations"] = alb(A.GaussianBlur(blur_limit=(5, 5), sigma_limit=(0.1, 2.0), p=1.0)) if A else None
    row["opencv"] = None
    ops["RandomGaussianBlur"] = row

    row = kornia_row("RandomBrightness", KA.RandomBrightness(brightness=(0.8, 1.2), p=1.0))
    row["torchvision v2"] = tv(T2.ColorJitter(brightness=(0.8, 1.2))) if T2 else None
    row["albumentations"] = (
        alb(A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.0, p=1.0)) if A else None
    )
    row["opencv"] = None
    ops["RandomBrightness"] = row

    row = kornia_row("RandomGrayscale", KA.RandomGrayscale(p=1.0))
    row["torchvision v2"] = tv(T2.RandomGrayscale(p=1.0)) if T2 else None
    row["albumentations"] = alb(A.ToGray(p=1.0)) if A else None
    row["opencv"] = (lambda: [cv2.cvtColor(im, cv2.COLOR_RGB2GRAY) for im in imgs_u8]) if cv2 else None
    row["PIL"] = (lambda: [pil.fromarray(im).convert("L") for im in imgs_u8]) if pil else None
    ops["RandomGrayscale"] = row

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
    np.random.seed(0)  # noqa: NPY002 — albumentations samples from the legacy global RNG
    random.seed(0)
    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)
    sync = torch.mps.synchronize if device.type == "mps" else None  # Timer only syncs CUDA

    try:
        import torchvision.transforms.v2 as T2
    except ImportError:
        T2 = None
    try:
        import albumentations as A
    except ImportError:
        A = None
    try:
        import cv2
    except ImportError:
        cv2 = None
    try:
        from PIL import Image as pil
    except ImportError:
        pil = None

    meta = run_metadata(device)
    print(f"# flagship augmentation benchmark — commit {meta['git_commit']} — {platform.platform()}")
    if device.type == "cuda":
        print(f"# CUDA device: {meta['cuda_device']}")
    print(f"# device={device}, dtype={args.dtype}, threads={args.threads}, size={args.size} — throughput img/s")
    print("# augmentation classes built once; timed region = parameter sampling + application per call")
    print(
        "# kornia/torchvision: batched float BCHW; albumentations/opencv/PIL: uint8 HWC per-image loop (CPU); "
        "'-' = skipped"
    )
    for lib, name in [(T2, "torchvision"), (A, "albumentations"), (cv2, "opencv"), (pil, "PIL")]:
        if lib is None:
            print(f"# NOTE: {name} not installed — its column is skipped")

    backends = ["kornia (eager)", "kornia (compiled)", "torchvision v2", "albumentations", "opencv", "PIL"]
    results = run_batch_sweep(
        [int(x) for x in args.batches.split(",")],
        lambda b: build_ops(b, args.size, args.size, device, dtype, args.compile, T2, A, cv2, pil),
        backends,
        row_fields=lambda b: {"height": args.size, "width": args.size, "dtype": args.dtype},
        sync=sync,
    )
    if args.json:
        out = save_json(args.json, meta, results)
        print(f"# results written to {out}")


if __name__ == "__main__":
    main()
