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

"""Comprehensive per-op image-augmentation throughput benchmark across every common library.

Compares, for a set of representative ops, the throughput of:

- **kornia** (eager)      — float ``BCHW`` batch, CPU or GPU, differentiable
- **kornia** (compiled)   — the same under ``torch.compile``
- **torchvision v2**      — float ``BCHW`` batch, CPU or GPU
- **albumentations**      — ``uint8`` HWC, per-image Python loop (CPU, OpenCV backend)
- **opencv** (cv2)        — ``uint8`` HWC, per-image Python loop (CPU)
- **PIL** (Pillow)        — ``uint8`` HWC, per-image Python loop (CPU)
- **kornia-rs**           — ``uint8`` HWC, per-image, native Rust (CPU)

Read the numbers with the regimes in mind — they are **not** apples-to-apples, and that is the
point of showing them together:

- albumentations / opencv / PIL / kornia-rs operate on a **single ``uint8`` image** at a time
  (a batch is a Python loop). This is the CPU/integer/single-image regime, and it is where the
  OpenCV-backed libraries and the native ``kornia-rs`` win.
- kornia and torchvision v2 operate on a **batched float tensor**, and on the **GPU**.
- Only **kornia** is differentiable end-to-end.

So: expect albumentations/opencv/kornia-rs to lead the CPU/uint8 rows, and kornia (compiled,
GPU-batched) to lead the throughput that matters for on-device differentiable training. The
purpose of this script is to make each regime measurable and honest, and to give a durable
baseline to improve against. See ``README.md`` in this directory for methodology and the list
of known improvement opportunities.

Usage:
    python benchmarks/augmentation/all_libraries.py [--batch 32] [--size 256] [--device cpu]
    python benchmarks/augmentation/all_libraries.py --device cuda --compile
"""

from __future__ import annotations

import argparse
import math
import platform
import subprocess
from typing import Callable, Optional

import numpy as np
import torch
import torch.utils.benchmark as bench


def _us(fn: Callable[[], object], min_run_time: float = 1.5) -> float:
    """Median wall-clock of ``fn`` in microseconds (CUDA-synced by torch.utils.benchmark)."""
    try:
        return bench.Timer(stmt="fn()", globals={"fn": fn}).blocked_autorange(min_run_time=min_run_time).median * 1e6
    except Exception:
        return float("nan")


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()  # noqa: S607
    except Exception:
        return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--compile", action="store_true", help="also time torch.compile'd kornia")
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    b, h, w = args.batch, args.size, args.size

    # Two views of the same random data: a batched float tensor (kornia/torchvision) and a list
    # of single uint8 HWC images (albumentations/opencv/PIL/kornia-rs).
    rng = np.random.default_rng(0)
    imgs_u8 = [(rng.random((h, w, 3)) * 255).astype(np.uint8) for _ in range(b)]
    batch_f = torch.stack([torch.from_numpy(im).permute(2, 0, 1).float().div(255) for im in imgs_u8]).to(device)

    # Lazily import the optional libraries; missing ones are simply reported as skipped.
    libs: dict[str, object] = {}
    for name, mod in [("cv2", "cv2"), ("PIL", "PIL.Image"), ("albumentations", "albumentations")]:
        try:
            libs[name] = __import__(mod, fromlist=["x"]) if "." in mod else __import__(mod)
        except Exception:
            libs[name] = None
    try:
        import torchvision.transforms.v2.functional as tvf

        libs["tv"] = tvf
    except Exception:
        libs["tv"] = None
    try:
        import kornia_rs

        libs["krs"] = kornia_rs
    except Exception:
        libs["krs"] = None

    import kornia

    def thr(t: float) -> float:
        return b / (t / 1e6) if t and not math.isnan(t) else float("nan")

    # Each op: name -> dict of {backend: zero-arg callable that applies it to the whole batch}.
    # `None` for a backend means "not benchmarked for this op".
    cv2 = libs["cv2"]
    pil = libs["PIL"]
    A = libs["albumentations"]
    tvf = libs["tv"]
    krs = libs["krs"]

    def kornia_run(op_eager: torch.nn.Module) -> dict[str, Optional[Callable[[], object]]]:
        op_eager = op_eager.to(device)
        out: dict[str, Optional[Callable[[], object]]] = {"kornia (eager)": lambda: op_eager(batch_f)}
        if args.compile:
            torch._dynamo.reset()
            compiled = torch.compile(op_eager)
            try:
                compiled(batch_f)  # warmup / compile
                out["kornia (compiled)"] = lambda: compiled(batch_f)
            except Exception:
                out["kornia (compiled)"] = None
        return out

    ops: dict[str, dict[str, Optional[Callable[[], object]]]] = {}

    # kornia-rs operates on single uint8 HWC numpy images (float input is unsupported).
    krs_ip = krs.imgproc if krs is not None else None

    # --- Horizontal flip -------------------------------------------------------------------
    row = kornia_run(kornia.augmentation.RandomHorizontalFlip(p=1.0))
    row["torchvision v2"] = (lambda: tvf.horizontal_flip(batch_f)) if tvf else None
    row["albumentations"] = (lambda: [A.HorizontalFlip(p=1.0)(image=im)["image"] for im in imgs_u8]) if A else None
    row["opencv"] = (lambda: [cv2.flip(im, 1) for im in imgs_u8]) if cv2 else None
    row["PIL"] = (lambda: [pil.fromarray(im).transpose(pil.FLIP_LEFT_RIGHT) for im in imgs_u8]) if pil else None
    row["kornia-rs"] = (lambda: [krs_ip.horizontal_flip(im) for im in imgs_u8]) if krs_ip else None
    ops["HorizontalFlip"] = row

    # --- Vertical flip ---------------------------------------------------------------------
    row = kornia_run(kornia.augmentation.RandomVerticalFlip(p=1.0))
    row["torchvision v2"] = (lambda: tvf.vertical_flip(batch_f)) if tvf else None
    row["albumentations"] = (lambda: [A.VerticalFlip(p=1.0)(image=im)["image"] for im in imgs_u8]) if A else None
    row["opencv"] = (lambda: [cv2.flip(im, 0) for im in imgs_u8]) if cv2 else None
    row["PIL"] = (lambda: [pil.fromarray(im).transpose(pil.FLIP_TOP_BOTTOM) for im in imgs_u8]) if pil else None
    row["kornia-rs"] = (lambda: [krs_ip.vertical_flip(im) for im in imgs_u8]) if krs_ip else None
    ops["VerticalFlip"] = row

    # --- Resize (downscale to half) --------------------------------------------------------
    dst = (h // 2, w // 2)
    row = kornia_run(kornia.augmentation.Resize(dst))
    row["torchvision v2"] = (lambda: tvf.resize(batch_f, list(dst), antialias=True)) if tvf else None
    row["albumentations"] = (lambda: [A.Resize(dst[0], dst[1])(image=im)["image"] for im in imgs_u8]) if A else None
    row["opencv"] = (lambda: [cv2.resize(im, (dst[1], dst[0])) for im in imgs_u8]) if cv2 else None
    row["PIL"] = (lambda: [pil.fromarray(im).resize((dst[1], dst[0])) for im in imgs_u8]) if pil else None
    row["kornia-rs"] = (lambda: [krs_ip.resize(im, dst, "bilinear") for im in imgs_u8]) if krs_ip else None
    ops["Resize"] = row

    # --- Gaussian blur ---------------------------------------------------------------------
    row = kornia_run(kornia.augmentation.RandomGaussianBlur((5, 5), (1.0, 1.0), p=1.0))
    row["torchvision v2"] = (lambda: tvf.gaussian_blur(batch_f, [5, 5], [1.0, 1.0])) if tvf else None
    row["albumentations"] = (
        (lambda: [A.GaussianBlur((5, 5), (1.0, 1.0), p=1.0)(image=im)["image"] for im in imgs_u8]) if A else None
    )
    row["opencv"] = (lambda: [cv2.GaussianBlur(im, (5, 5), 1.0) for im in imgs_u8]) if cv2 else None
    row["kornia-rs"] = (lambda: [krs_ip.gaussian_blur(im, (5, 5), (1.0, 1.0)) for im in imgs_u8]) if krs_ip else None
    ops["GaussianBlur"] = row

    # --- Brightness ------------------------------------------------------------------------
    row = kornia_run(kornia.augmentation.RandomBrightness((1.3, 1.3), p=1.0))
    row["torchvision v2"] = (lambda: tvf.adjust_brightness(batch_f, 1.3)) if tvf else None
    row["albumentations"] = (
        (lambda: [A.RandomBrightnessContrast(0.3, 0, p=1.0)(image=im)["image"] for im in imgs_u8]) if A else None
    )
    row["opencv"] = (lambda: [cv2.convertScaleAbs(im, alpha=1.3, beta=0) for im in imgs_u8]) if cv2 else None
    row["kornia-rs"] = (lambda: [krs_ip.adjust_brightness(im, 1.3) for im in imgs_u8]) if krs_ip else None
    ops["Brightness"] = row

    # --- Grayscale -------------------------------------------------------------------------
    row = kornia_run(kornia.augmentation.RandomGrayscale(p=1.0))
    row["torchvision v2"] = (lambda: tvf.rgb_to_grayscale(batch_f, num_output_channels=3)) if tvf else None
    row["albumentations"] = (lambda: [A.ToGray(p=1.0)(image=im)["image"] for im in imgs_u8]) if A else None
    row["opencv"] = (lambda: [cv2.cvtColor(im, cv2.COLOR_RGB2GRAY) for im in imgs_u8]) if cv2 else None
    row["PIL"] = (lambda: [pil.fromarray(im).convert("L") for im in imgs_u8]) if pil else None
    row["kornia-rs"] = (lambda: [krs_ip.gray_from_rgb(im) for im in imgs_u8]) if krs_ip else None
    ops["Grayscale"] = row

    # ---------------------------------------------------------------------------------------
    backends = [
        "kornia (eager)",
        "kornia (compiled)",
        "torchvision v2",
        "albumentations",
        "opencv",
        "PIL",
        "kornia-rs",
    ]

    print(f"# all-library augmentation benchmark — commit {_git_commit()} — {platform.platform()}")
    if device.type == "cuda":
        print(f"# CUDA device: {torch.cuda.get_device_name(0)}")
    print(
        f"# batch={b}, {h}x{w}, device={device}, threads={args.threads} — throughput img/s (higher is better)\n"
        f"# kornia/torchvision: float BCHW batch{' on ' + device.type if device.type == 'cuda' else ''}; "
        f"albumentations/opencv/PIL/kornia-rs: uint8 HWC single-image loop (CPU)"
    )
    col_w = 13
    header = f"{'op':<15}" + "".join(f"{n[:col_w]:>{col_w + 1}}" for n in backends)
    print("-" * len(header))
    print(header)
    print("-" * len(header))
    for op_name, row in ops.items():
        cells = []
        for backend in backends:
            fn = row.get(backend)
            cells.append(f"{thr(_us(fn)):>{col_w + 1}.0f}" if fn is not None else f"{'-':>{col_w + 1}}")
        print(f"{op_name:<15}" + "".join(cells))
    print("-" * len(header))
    print("# '-' = op not benchmarked for that backend (unsupported or not mapped).")


if __name__ == "__main__":
    main()
