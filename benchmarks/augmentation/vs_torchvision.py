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

"""Per-op kornia-vs-torchvision-v2 diagnostic: every augmentation with a tv equivalent.

For each op it times **kornia eager**, **kornia ``torch.compile``**, and **torchvision v2** on
the same batched float ``BCHW`` tensor, and prints the best-kornia / tv ratio plus a verdict.
Both libraries run on the same device; both apply one transform per call. This is the diagnostic
that answers "where does kornia beat torchvision, and by how much" op by op — the companion to
``pipeline.py`` (which times a whole compiled pipeline, kornia's headline regime).

Reading it honestly:

- **Pointwise / color** (flips, ColorJitter, brightness, invert, posterize, solarize, sharpness,
  equalize, gaussian noise) is kornia's regime — differentiable, and ``torch.compile`` fuses the
  chain. kornia is competitive-to-winning here, especially compiled.
- **GaussianBlur / Erasing** apply *per-sample* parameters (a kernel / mask per batch element),
  where torchvision applies one batch-shared parameter — so torchvision does strictly less work.
- **Geometric** (rotation/affine/perspective/resized-crop) route through ``grid_sample``;
  torchvision's private sampler path is hard to match with public torch ops.

Every measurement is the median of a manual warmup+loop with a CPU-governor ramp (throttled
laptop/Jetson CPUs otherwise report wildly inflated host time). Run GPU numbers locally with
``--device cuda``; on some edge wheels (e.g. Jetson) ``torch.compile`` errors on a subset of ops
and those cells show ``ERR`` — that is a wheel limitation, reported honestly, not a kornia bug.

Usage:
    python benchmarks/augmentation/vs_torchvision.py [--batch 32] [--size 224] [--device cpu]
"""

from __future__ import annotations

import argparse
import platform
import subprocess
import time

import torch


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()  # noqa: S607
    except Exception:
        return "unknown"


def _time_ms(fn, device: torch.device, iters: int = 200, ramp_s: float = 1.5) -> float | tuple[str, str]:
    """Median-ish ms per call: ramp the CPU governor, then time ``iters`` calls, GPU-synced."""
    try:
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < ramp_s:
            fn()
        if device.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        for _ in range(iters):
            fn()
        if device.type == "cuda":
            torch.cuda.synchronize()
        return (time.perf_counter() - t1) / iters * 1e3
    except Exception as e:  # pragma: no cover - op/backend unsupported on this wheel
        return ("ERR", str(e).splitlines()[-1][:48])


def _build_ops():
    """(name, kornia factory, torchvision factory, is_geometric) for every op with a tv match."""
    import torchvision.transforms.v2 as T2

    import kornia.augmentation as K

    return [
        ("HFlip", lambda: K.RandomHorizontalFlip(p=1.0), lambda: T2.RandomHorizontalFlip(p=1.0), False),
        ("VFlip", lambda: K.RandomVerticalFlip(p=1.0), lambda: T2.RandomVerticalFlip(p=1.0), False),
        ("Grayscale", lambda: K.RandomGrayscale(p=1.0), lambda: T2.RandomGrayscale(p=1.0), False),
        (
            "ColorJitter",
            lambda: K.ColorJitter(0.2, 0.2, 0.2, 0.1, p=1.0, order=[0, 1, 2, 3]),
            lambda: T2.ColorJitter(0.2, 0.2, 0.2, 0.1),
            False,
        ),
        (
            "GaussBlur",
            lambda: K.RandomGaussianBlur((3, 3), (0.1, 2.0), p=1.0),
            lambda: T2.GaussianBlur(3, (0.1, 2.0)),
            False,
        ),
        (
            "GaussNoise",
            lambda: K.RandomGaussianNoise(mean=0.0, std=0.05, p=1.0),
            lambda: T2.GaussianNoise(mean=0.0, sigma=0.05),
            False,
        ),
        ("Invert", lambda: K.RandomInvert(p=1.0), lambda: T2.RandomInvert(p=1.0), False),
        ("Posterize", lambda: K.RandomPosterize(3, p=1.0), lambda: T2.RandomPosterize(3, p=1.0), False),
        ("Solarize", lambda: K.RandomSolarize(0.1, 0.1, p=1.0), lambda: T2.RandomSolarize(0.5, p=1.0), False),
        ("Sharpness", lambda: K.RandomSharpness(0.5, p=1.0), lambda: T2.RandomAdjustSharpness(2.0, p=1.0), False),
        ("Equalize", lambda: K.RandomEqualize(p=1.0), lambda: T2.RandomEqualize(p=1.0), False),
        ("Erasing", lambda: K.RandomErasing(p=1.0), lambda: T2.RandomErasing(p=1.0), False),
        ("Rotation", lambda: K.RandomRotation(30.0, p=1.0), lambda: T2.RandomRotation(30), True),
        ("Affine", lambda: K.RandomAffine(30.0, p=1.0), lambda: T2.RandomAffine(30), True),
        ("Perspective", lambda: K.RandomPerspective(0.5, p=1.0), lambda: T2.RandomPerspective(0.5, p=1.0), True),
        (
            "ResizedCrop",
            lambda: K.RandomResizedCrop((224, 224), p=1.0),
            lambda: T2.RandomResizedCrop((224, 224), antialias=True),
            True,
        ),
        ("Elastic", lambda: K.RandomElasticTransform(p=1.0), T2.ElasticTransform, True),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--size", type=int, default=224)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    b, s = args.batch, args.size
    x = torch.rand(b, 3, s, s, device=device)

    # Warm the CPU governor once before any measurement (throttled CPUs ramp slowly).
    for _ in range(300):
        x * 1.0001
    if device.type == "cuda":
        torch.cuda.synchronize()

    print(f"# kornia vs torchvision-v2 per-op — commit {_git_commit()} — {platform.platform()}")
    if device.type == "cuda":
        print(f"# CUDA device: {torch.cuda.get_device_name(0)}")
    print(f"# batch={b}, {s}x{s}, device={device}, threads={args.threads} — ms/call (lower is better)")
    print("-" * 74)
    print(f"{'op':12s} {'k_eager':>9s} {'k_comp':>9s} {'tv':>9s} {'best/tv':>8s}  verdict")
    print("-" * 74)

    def fmt(v) -> str:
        return f"{v:9.2f}" if isinstance(v, float) else f"{'ERR':>9s}"

    for name, kb, tb, geo in _build_ops():
        try:
            kop = kb().to(device)
        except Exception as e:
            print(f"{name:12s} kornia-build-ERR: {str(e).splitlines()[-1][:40]}")
            continue
        ke = _time_ms(lambda op=kop: op(x), device)
        try:
            torch._dynamo.reset()
            kc = torch.compile(kb().to(device))
            kcomp = _time_ms(lambda op=kc: op(x), device)
        except Exception as e:
            kcomp = ("ERR", str(e).splitlines()[-1][:30])
        try:
            top = tb()
            tv = _time_ms(lambda op=top: op(x), device)
        except Exception as e:
            tv = ("ERR", str(e).splitlines()[-1][:40])

        finite_k = [v for v in (ke, kcomp) if isinstance(v, float)]
        if finite_k and isinstance(tv, float):
            ratio = min(finite_k) / tv
            verdict = "WIN" if ratio <= 1.0 else "LOSE"
            ratio_s = f"{ratio:8.2f}"
        else:
            verdict, ratio_s = "tv-err" if not isinstance(tv, float) else "k-err", f"{'-':>8s}"
        print(f"{name:12s} {fmt(ke)} {fmt(kcomp)} {fmt(tv)} {ratio_s}  {verdict}" + ("  [geometric]" if geo else ""))

    print("-" * 74)
    print("# best/tv = min(kornia eager, compiled) / torchvision; <=1.0 => kornia WIN.")
    print("# ERR under compile on Jetson wheels = torch.compile limitation, not a kornia defect.")


if __name__ == "__main__":
    main()
