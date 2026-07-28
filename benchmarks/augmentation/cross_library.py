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

"""Cross-library augmentation throughput benchmark: kornia vs torchvision vs albumentations.

Compares like-for-like augmentations across the three libraries so kornia's numbers are
grounded against real alternatives instead of only against itself.

Reading the results honestly:

- **albumentations** operates on single ``uint8`` HWC numpy arrays via OpenCV; a batch is a
  Python loop. This is its home turf (CPU, integer, single-image) and it wins there.
- **torchvision v2** and **kornia** operate on batched ``float`` ``BCHW`` tensors.
- **kornia** additionally runs the same op under ``torch.compile`` and (when a GPU is
  present) on-device — its home turf is GPU-batched, differentiable augmentation.

On CPU, ``torch.compile`` is kornia's lever to close the gap for pointwise ops (it can beat
albumentations on e.g. brightness). Geometric ops that still graph-break (a ``.item()`` on
the input-shape tensor) do not yet benefit — those are the compile targets that matter most
competitively. Run on a CUDA box to see the GPU-batched regime where kornia leads.

Usage:
    python benchmarks/augmentation/cross_library.py [--batch 32] [--size 256] [--device cpu]
"""

from __future__ import annotations

import argparse
import math

import numpy as np
import torch
import torch.utils.benchmark as bench


def _throughput_us(fn, min_run_time: float = 1.5) -> float:
    return bench.Timer(stmt="fn()", globals={"fn": fn}).blocked_autorange(min_run_time=min_run_time).median * 1e6


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()

    import albumentations as A
    import torchvision.transforms.v2 as T2

    import kornia.augmentation as K

    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    b, h, w = args.batch, args.size, args.size

    rng = np.random.default_rng(0)
    batch_f = torch.rand(b, 3, h, w, device=device)
    batch_u8 = [(rng.random((h, w, 3)) * 255).astype(np.uint8) for _ in range(b)]

    def bench_row(name, kornia_op, tv_op, albu_op):
        kornia_op = kornia_op.to(device)
        eager = _throughput_us(lambda: kornia_op(batch_f))
        torch._dynamo.reset()
        compiled_op = torch.compile(kornia_op)
        compiled_op(batch_f)  # warmup
        compiled = _throughput_us(lambda: compiled_op(batch_f))
        tv = _throughput_us(lambda: tv_op(batch_f)) if tv_op is not None else float("nan")
        albu = _throughput_us(lambda: [albu_op(image=im)["image"] for im in batch_u8]) if albu_op else float("nan")

        def thr(t: float) -> float:
            return b / (t / 1e6) if not math.isnan(t) else float("nan")

        print(
            f"{name:16} | kornia {thr(eager):9.0f} | k-compiled {thr(compiled):9.0f} "
            f"| torchvision {thr(tv):9.0f} | albumentations {thr(albu):9.0f}  img/s"
        )

    print(f"batch={b}, {h}x{w}, device={device}, threads={args.threads} — throughput (higher is better)")
    print("-" * 104)
    bench_row(
        "HorizontalFlip",
        K.RandomHorizontalFlip(p=1.0),
        T2.RandomHorizontalFlip(p=1.0),
        A.HorizontalFlip(p=1.0),
    )
    bench_row(
        "GaussianBlur",
        K.RandomGaussianBlur((5, 5), (1.0, 1.0), p=1.0),
        T2.GaussianBlur(5, (1.0, 1.0)),
        A.GaussianBlur((5, 5), (1.0, 1.0), p=1.0),
    )
    bench_row(
        "Brightness",
        K.RandomBrightness((1.3, 1.3), p=1.0),
        T2.ColorJitter(brightness=(1.3, 1.3)),
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0, p=1.0),
    )
    bench_row(
        "Rotate90",
        K.RandomRotation((90.0, 90.0), p=1.0),
        T2.RandomRotation((90, 90)),
        A.Rotate((90, 90), p=1.0),
    )


if __name__ == "__main__":
    main()
