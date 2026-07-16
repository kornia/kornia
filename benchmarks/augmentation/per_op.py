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

"""Per-op augmentation throughput sweep — kornia eager vs ``torch.compile``, every op.

Benchmarks each augmentation individually (eager and compiled) so slow ops and compile
regressions are visible *by fact*, not by guess. It is the tool that surfaced, e.g., that
``RandomElasticTransform`` was ~1000x slower than a flip (a full 2-D Gaussian conv that should
be separable), and that a few conv-bound ops *regress* under ``torch.compile`` on CPU.

For each op it reports eager and compiled throughput (img/s), the speedup, and which compile
mode succeeded (``fg`` = fullgraph, ``gb`` = graph-breaks allowed, ``—`` = compile failed). The
tail summarises the slowest ops (improvement targets) and the ops where compile hurts
(<=1.05x — wrapper/kernel-bound, not helped by fusion).

Run all GPU numbers locally (`--device cuda`); on some edge wheels (e.g. Jetson) a subset of
ops can't ``torch.compile`` on GPU — those show ``—`` in the compiled column, which is a wheel
limitation, not an op bug.

Usage:
    python benchmarks/augmentation/per_op.py [--device cpu] [--batch 16] [--size 64]
"""

from __future__ import annotations

import argparse
import math

import torch
import torch.utils.benchmark as bench


def _us(fn, min_run_time: float = 1.0) -> float:
    try:
        return bench.Timer(stmt="fn()", globals={"fn": fn}).blocked_autorange(min_run_time=min_run_time).median * 1e6
    except Exception:
        return float("nan")


def _registry(K):
    """name -> factory. p=1.0 so the op always applies; args chosen to be representative."""
    return {
        "HFlip": lambda: K.RandomHorizontalFlip(p=1.0),
        "VFlip": lambda: K.RandomVerticalFlip(p=1.0),
        "Rotation": lambda: K.RandomRotation(30.0, p=1.0),
        "Affine": lambda: K.RandomAffine(30.0, p=1.0),
        "Shear": lambda: K.RandomShear((10.0, 10.0), p=1.0),
        "Translate": lambda: K.RandomTranslate((0.1, 0.1), p=1.0),
        "CenterCrop": lambda: K.CenterCrop((48, 48)),
        "RandomCrop": lambda: K.RandomCrop((48, 48), p=1.0),
        "Resize": lambda: K.Resize((48, 48)),
        "ResizedCrop": lambda: K.RandomResizedCrop((48, 48), p=1.0),
        "Elastic": lambda: K.RandomElasticTransform(p=1.0),
        "ThinPlate": lambda: K.RandomThinPlateSpline(p=1.0),
        "PadTo": lambda: K.PadTo((80, 80)),
        "Brightness": lambda: K.RandomBrightness((1.3, 1.3), p=1.0),
        "Contrast": lambda: K.RandomContrast((1.3, 1.3), p=1.0),
        "Saturation": lambda: K.RandomSaturation((1.3, 1.3), p=1.0),
        "Hue": lambda: K.RandomHue((0.1, 0.1), p=1.0),
        "Gamma": lambda: K.RandomGamma((0.8, 1.2), (1.0, 1.0), p=1.0),
        "ColorJitter": lambda: K.ColorJitter(0.2, 0.2, 0.2, 0.1, p=1.0),
        "ColorJiggle": lambda: K.ColorJiggle(0.2, 0.2, 0.2, 0.1, p=1.0),
        "Grayscale": lambda: K.RandomGrayscale(p=1.0),
        "Invert": lambda: K.RandomInvert(p=1.0),
        "Solarize": lambda: K.RandomSolarize(0.1, 0.1, p=1.0),
        "Sharpness": lambda: K.RandomSharpness(0.5, p=1.0),
        "Equalize": lambda: K.RandomEqualize(p=1.0),
        "AutoContrast": lambda: K.RandomAutoContrast(p=1.0),
        "RGBShift": lambda: K.RandomRGBShift(p=1.0),
        "PlanckianJitter": lambda: K.RandomPlanckianJitter(p=1.0),
        "ChannelShuffle": lambda: K.RandomChannelShuffle(p=1.0),
        "ChannelDropout": lambda: K.RandomChannelDropout(p=1.0),
        "Normalize": lambda: K.Normalize(mean=torch.tensor([0.5, 0.5, 0.5]), std=torch.tensor([0.2, 0.2, 0.2]), p=1.0),
        "GaussianBlur": lambda: K.RandomGaussianBlur((5, 5), (1.0, 1.0), p=1.0),
        "BoxBlur": lambda: K.RandomBoxBlur((5, 5), p=1.0),
        "MedianBlur": lambda: K.RandomMedianBlur((5, 5), p=1.0),
        "MotionBlur": lambda: K.RandomMotionBlur(5, 35.0, 0.5, p=1.0),
        "GaussianNoise": lambda: K.RandomGaussianNoise(0.0, 0.05, p=1.0),
        "SaltPepper": lambda: K.RandomSaltAndPepperNoise(p=1.0),
        "Erasing": lambda: K.RandomErasing(p=1.0),
        "GaussianIllum": lambda: K.RandomGaussianIllumination(p=1.0),
        "LinearIllum": lambda: K.RandomLinearIllumination(p=1.0),
        "PlasmaBright": lambda: K.RandomPlasmaBrightness(p=1.0),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--size", type=int, default=64)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    import kornia.augmentation as K

    dev = torch.device(args.device)
    b = args.batch
    x = torch.rand(b, 3, args.size, args.size, device=dev)

    def thr(t):
        return b / (t / 1e6) if t and not math.isnan(t) else float("nan")

    print(f"device={dev}  batch={b} {args.size}x{args.size}  (img/s, higher is better)")
    print(f"{'op':<16}{'eager':>10}{'compiled':>10}{'speedup':>9}  mode")
    print("-" * 56)
    rows = []
    for name, factory in _registry(K).items():
        try:
            op = factory().to(dev)
            params = op.forward_parameters(x.shape)
            op(x, params=params)  # warmup
            te = _us(lambda op=op, params=params: op(x, params=params))
        except Exception as e:
            print(f"{name:<16}{'—':>10}{'—':>10}{'—':>9}  skip: {str(e).splitlines()[0][:24]}")
            continue
        tc, mode = float("nan"), "—"
        for fullgraph in (True, False):
            try:
                torch._dynamo.reset()
                compiled = torch.compile(op, fullgraph=fullgraph)
                compiled(x, params=params)
                tc = _us(lambda compiled=compiled, params=params: compiled(x, params=params))
                mode = "fg" if fullgraph else "gb"
                break
            except Exception:  # noqa: S112 - benchmark: an op that won't compile is just skipped
                continue
        sp = te / tc if te and tc and not math.isnan(te) and not math.isnan(tc) else float("nan")
        rows.append((name, thr(te), thr(tc), sp, mode))
        print(f"{name:<16}{thr(te):>10.0f}{thr(tc):>10.0f}{sp:>8.2f}x  {mode}")

    print("-" * 56)
    print("\nSlowest eager (improvement targets):")
    for r in sorted(rows, key=lambda r: r[1])[:8]:
        print(f"  {r[0]:<16} {r[1]:>8.0f} img/s  (compiled {r[2]:.0f}, {r[3]:.2f}x {r[4]})")
    print("\nCompile does not help (<=1.05x — wrapper/kernel-bound):")
    for r in sorted((r for r in rows if not math.isnan(r[3])), key=lambda r: r[3]):
        if r[3] <= 1.05:
            print(f"  {r[0]:<16} {r[3]:.2f}x  (eager {r[1]:.0f} -> compiled {r[2]:.0f})")


if __name__ == "__main__":
    main()
