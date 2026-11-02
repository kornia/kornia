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
"""Patch descriptor and shape networks: channels-last versus the default memory layout.

Two sections, both on synthetic ``(N, 1, 32, 32)`` patches and untrained weights (``pretrained=False``; the
weights change neither the layout nor the arithmetic):

- ``layout``: the convolution body of HardNet, HardNet8, SOSNet, TFeat, HyNet, AffNet and OriNet -- the public
  submodules up to the last convolution -- timed with input and weights in the default layout and again in
  channels-last. Independent of the layout kornia picks, so it answers which layout a device should use. Each
  row records the largest absolute difference between the two layouts' outputs.
- ``forward``: the public forward of each network on ready patches, except AffNet (``LAFAffNetShapeEstimator``),
  whose forward extracts its own patches from a 480x640 image, plus ``extract_patches_from_pyramid`` followed by
  ``HardNet`` on that image, as users call them. Run it from the base and the changed checkout to compare the
  layout kornia picks.

Inputs live on the measured device; no downloads, transfers or file IO are timed.
"""

from __future__ import annotations

# Imports follow path setup so direct execution always measures this checkout.
# ruff: noqa: E402
import argparse
import copy
import hashlib
import sys
from functools import partial
from pathlib import Path
from typing import Callable

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "benchmarks"))

import torch
from common import kornia_provenance, run_metadata, save_json, setup_run, time_us
from torch import nn

import kornia
from kornia import feature

NETWORKS: dict[str, Callable[[], nn.Module]] = {
    "HardNet": lambda: feature.HardNet(pretrained=False),
    "HardNet8": lambda: feature.HardNet8(pretrained=False),
    "SOSNet": lambda: feature.SOSNet(pretrained=False),
    "TFeat": lambda: feature.TFeat(pretrained=False),
    "HyNet": lambda: feature.HyNet(pretrained=False),
    "AffNet": lambda: feature.LAFAffNetShapeEstimator(pretrained=False),
    "OriNet": lambda: feature.OriNet(pretrained=False),
}


def conv_body(name: str, net: nn.Module) -> nn.Sequential:
    """The network's layers in call order, up to and including its last convolution, as one plain Sequential.

    Stopping at the last convolution leaves out the trailing BatchNorm that the descriptors keep in float32 for
    half inputs, and the descriptor normalisation; both act on ``(N, D, 1, 1)`` and cost next to nothing.
    """
    if name == "HyNet":
        layers = [
            m
            for block in (net.layer1, net.layer2, net.layer3, net.layer4, net.layer5, net.layer6, net.layer7)
            for m in block
        ]
    else:
        layers = list(net.layers if name == "SOSNet" else net.features)
    last = max(i for i, m in enumerate(layers) if isinstance(m, nn.Conv2d))
    return nn.Sequential(*layers[: last + 1])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="float32", choices=["float32", "float16", "bfloat16", "float64"])
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--counts", default="512,2048", help="comma-separated patch counts N")
    parser.add_argument(
        "--networks", default=",".join(NETWORKS), help="comma-separated subset of " + ",".join(NETWORKS)
    )
    parser.add_argument("--sections", default="layout,forward")
    parser.add_argument("--cudnn-benchmark", action="store_true", help="set torch.backends.cudnn.benchmark")
    parser.add_argument("--min-run-time", type=float, default=1.0)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    names = args.networks.split(",")
    unknown = sorted(set(names) - set(NETWORKS))
    if unknown:
        parser.error(f"unknown networks: {unknown}")
    sections = set(args.sections.split(","))
    device, dtype, sync = setup_run(args, opencv=False)
    torch.backends.cudnn.benchmark = args.cudnn_benchmark
    source, module = kornia_provenance(kornia.__file__)
    print(f"# kornia source: {source}\n# interpreter: {sys.executable}", flush=True)
    if module == "outside-checkout":
        raise RuntimeError("kornia was imported from a different checkout")
    metadata = {
        **run_metadata(device),
        "kornia_module": module,
        "dtype": args.dtype,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "mkldnn_available": torch.backends.mkldnn.is_available(),
        "benchmark_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "units": "patches/s",
        "regime": __doc__,
    }
    print(
        f"# {metadata['platform']} | torch {torch.__version__} | device={device} dtype={args.dtype} "
        f"threads={args.threads} | oneDNN={metadata['mkldnn_available']} | commit {metadata['git_commit']}"
    )
    rows: list[dict[str, object]] = []

    def record(section: str, net: str, count: int, variant: str, fn: Callable[[], object], **extra: object) -> float:
        median, iqr = time_us(fn, min_run_time=args.min_run_time, sync=sync)
        rows.append(
            {
                "section": section,
                "network": net,
                "count": count,
                "variant": variant,
                "dtype": args.dtype,
                "median_us": median,
                "iqr_us": iqr,
                "throughput_per_s": count * 1e6 / median,
                **extra,
            }
        )
        if args.json:
            save_json(args.json, metadata, rows)
        return median

    counts = [int(n) for n in args.counts.split(",")]
    with torch.inference_mode():
        if "layout" in sections:
            print("\n# layout: convolution body, ms (median)   default | channels-last | speed-up | max|diff|")
            for name in names:
                net = NETWORKS[name]().eval().to(device, dtype)
                body = conv_body(name, net)
                body_cl = copy.deepcopy(body).to(memory_format=torch.channels_last)
                for count in counts:
                    x = torch.randn(count, 1, 32, 32, device=device, dtype=dtype)
                    x_cl = x.to(memory_format=torch.channels_last)
                    diff = (body(x).float() - body_cl(x_cl).float()).abs().max().item()
                    default = record("layout", name, count, "default", partial(body, x), max_abs_diff=diff)
                    cl = record("layout", name, count, "channels_last", partial(body_cl, x_cl), max_abs_diff=diff)
                    print(
                        f"{name:9s} N={count:5d}  {default / 1e3:9.2f} | {cl / 1e3:9.2f} | x{default / cl:5.2f}"
                        f" | {diff:.1e}",
                        flush=True,
                    )
        if "forward" in sections:
            print("\n# forward: public call as kornia runs it, ms (median)")
            generator = torch.Generator().manual_seed(0)
            height, width = 480, 640
            image = torch.rand(1, 1, height, width, generator=generator).to(device, dtype)
            lafs = {}
            for count in counts:
                xy = torch.rand(1, count, 2, generator=generator) * torch.tensor([width - 1.0, height - 1.0])
                scale = 4.0 + 60.0 * torch.rand(1, count, 1, 1, generator=generator)
                angle = 360.0 * torch.rand(1, count, 1, generator=generator)
                lafs[count] = feature.laf_from_center_scale_ori(xy, scale, angle).to(device)
            for name in names:
                net = NETWORKS[name]().eval().to(device, dtype)
                for count in counts:
                    patches = torch.rand(count, 1, 32, 32, device=device, dtype=dtype)
                    call = partial(net, lafs[count], image) if name == "AffNet" else partial(net, patches)
                    median = record("forward", name, count, "kornia", call)
                    print(f"{name:15s} N={count:5d}  {median / 1e3:9.2f}", flush=True)
            hardnet = feature.HardNet(pretrained=False).eval().to(device, dtype)
            for count in counts:

                def pipeline(laf: torch.Tensor = lafs[count]) -> torch.Tensor:
                    return hardnet(feature.extract_patches_from_pyramid(image, laf, 32).view(-1, 1, 32, 32))

                median = record("forward", "extract+HardNet", count, "kornia", pipeline)
                print(f"{'extract+HardNet':15s} N={count:5d}  {median / 1e3:9.2f}", flush=True)
    if args.json:
        save_json(args.json, metadata, rows)
        print(f"# results written to {args.json}")


if __name__ == "__main__":
    main()
