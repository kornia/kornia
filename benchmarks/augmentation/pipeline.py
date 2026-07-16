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

"""End-to-end augmentation *pipeline* throughput: kornia vs torchvision vs albumentations.

Where ``cross_library.py`` times single ops, this times a realistic multi-op pipeline applied
to a whole batch — the shape training loops actually run. It exists to measure the regime
kornia is built to lead: **GPU-batched, differentiable augmentation compiled end-to-end**.

Reading the results honestly:

- **albumentations** applies the pipeline to single ``uint8`` HWC numpy images in a Python loop
  (its home turf: CPU, integer, per-image). It has no batch or GPU or gradient path.
- **torchvision v2** and **kornia** apply the pipeline to a batched ``float`` ``BCHW`` tensor,
  and both run on GPU. Neither albumentations nor torchvision v2's default transforms are
  differentiable end-to-end; kornia's are.
- **kornia (compiled)** wraps the same pipeline in ``torch.compile``. On GPU the pointwise ops
  fuse; this is where kornia is designed to win. The pipeline is chosen from the compile-clean
  op set (see the per-op ``test_dynamo`` tests), and ``ColorJitter`` is given a fixed ``order``
  so the whole pipeline is fullgraph-compilable.

The point is not to win the CPU/uint8 single-image race (albumentations owns that — that is what
the planned ``kornia-rs`` backend targets). It is to quantify the GPU-batched, differentiable,
compiled pipeline where kornia leads. Record the CUDA row on a datacenter GPU for headline
numbers; Jetson/CPU rows are directional.

Usage:
    python benchmarks/augmentation/pipeline.py [--batch 32] [--size 224] [--device cpu] [--compile]
"""

from __future__ import annotations

import argparse
import math
import platform
import subprocess

import numpy as np
import torch
import torch.utils.benchmark as bench
from torch import nn


def _throughput_us(fn, min_run_time: float = 2.0) -> float:
    return bench.Timer(stmt="fn()", globals={"fn": fn}).blocked_autorange(min_run_time=min_run_time).median * 1e6


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()  # noqa: S607
    except Exception:
        return "unknown"


def build_kornia(device: torch.device) -> nn.Module:
    import kornia.augmentation as K

    # p=1.0 everywhere for a deterministic, apples-to-apples timing of the applied work.
    # A fixed `order` keeps ColorJitter (and hence the whole pipeline) fullgraph-compilable.
    return nn.Sequential(
        K.RandomHorizontalFlip(p=1.0),
        K.RandomVerticalFlip(p=1.0),
        K.ColorJitter(0.2, 0.2, 0.2, 0.1, p=1.0, order=[0, 1, 2, 3]),
        K.RandomBrightness((1.2, 1.2), p=1.0),
        K.RandomContrast((1.2, 1.2), p=1.0),
        K.RandomGaussianNoise(mean=0.0, std=0.05, p=1.0),
    ).to(device)


def build_torchvision() -> object:
    import torchvision.transforms.v2 as T2

    return T2.Compose(
        [
            T2.RandomHorizontalFlip(p=1.0),
            T2.RandomVerticalFlip(p=1.0),
            T2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            T2.GaussianNoise(mean=0.0, sigma=0.05),
        ]
    )


def build_albumentations() -> object:
    import albumentations as A

    return A.Compose(
        [
            A.HorizontalFlip(p=1.0),
            A.VerticalFlip(p=1.0),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=1.0),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1.0),
            A.GaussNoise(p=1.0),
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--size", type=int, default=224)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--compile", action="store_true", help="also time the torch.compile'd kornia pipeline")
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    b, h, w = args.batch, args.size, args.size

    rng = np.random.default_rng(0)
    batch_f = torch.rand(b, 3, h, w, device=device)
    batch_u8 = [(rng.random((h, w, 3)) * 255).astype(np.uint8) for _ in range(b)]

    def thr(t: float) -> float:
        return b / (t / 1e6) if not math.isnan(t) else float("nan")

    print(f"# pipeline benchmark — commit {_git_commit()} — {platform.platform()}")
    if device.type == "cuda":
        print(f"# CUDA device: {torch.cuda.get_device_name(0)}")
    print(f"batch={b}, {h}x{w}, device={device}, threads={args.threads} — throughput img/s (higher is better)")
    print("-" * 84)

    results: dict[str, float] = {}

    kornia_pipe = build_kornia(device)
    results["kornia (eager)"] = thr(_throughput_us(lambda: kornia_pipe(batch_f)))

    if args.compile:
        torch._dynamo.reset()
        compiled = torch.compile(kornia_pipe)
        compiled(batch_f)  # warmup / trigger compilation
        results["kornia (compiled)"] = thr(_throughput_us(lambda: compiled(batch_f)))

    try:
        tv = build_torchvision()
        results["torchvision v2"] = thr(_throughput_us(lambda: tv(batch_f)))
    except Exception as e:  # pragma: no cover - optional dependency / version drift
        print(f"# torchvision skipped: {e}")

    try:
        albu = build_albumentations()
        results["albumentations"] = thr(_throughput_us(lambda: [albu(image=im)["image"] for im in batch_u8]))
    except Exception as e:  # pragma: no cover - optional dependency
        print(f"# albumentations skipped: {e}")

    width = max(len(k) for k in results)
    for name, value in results.items():
        print(f"{name:<{width}} | {value:9.0f} img/s")


if __name__ == "__main__":
    main()
