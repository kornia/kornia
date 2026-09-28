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

"""Compare public Gaussian blur and scale-pyramid APIs across revisions.

Run identical commands from each checkout as a module (``python -m
benchmarks.filters.gaussian_cpu``), and verify the printed Kornia import path.
The baseline is the same public API on the base revision, not a replacement
implementation. CPU float tensors, fixed seed, eager execution, and one timed
thread; allocation, Gaussian kernel generation, and optional backward are timed.
CUDA/MPS runs check the unchanged accelerator path, with device synchronization.
Reports include medians/IQRs and losses alongside wins. Save reference tensors
outside the repository with --save-reference, then use --reference to quantify
output and gradient differences. Reference files must be locally generated.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path
from typing import Any, Callable

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import collect_load_metrics, run_metadata, save_json, time_us, versions_line

import kornia
from kornia.filters import gaussian_blur2d
from kornia.geometry.transform import ScalePyramid


def cases() -> list[dict[str, Any]]:
    """Fixed workloads spanning accelerated and deliberately unchanged regimes."""
    rows = []
    for size, channels in ((128, 1), (256, 1), (256, 3), (512, 1)):
        for kernel in (7,) if size == 128 or (channels == 1 and size == 256) else (3, 7, 15, 31):
            rows.append(
                {"name": f"blur-{size}-c{channels}-k{kernel}", "size": size, "channels": channels, "kernel": kernel}
            )
    rows.extend(
        [
            {"name": "blur-batch", "batch": 4, "size": 256, "channels": 3, "kernel": 15},
            {"name": "blur-batched-sigma", "batch": 2, "size": 256, "channels": 3, "kernel": 7, "batched_sigma": True},
            {"name": "blur-float64", "size": 512, "channels": 1, "kernel": 15, "dtype": "float64"},
            {"name": "blur-channels-last", "size": 256, "channels": 3, "kernel": 7, "layout": "channels_last"},
            {"name": "blur-strided", "size": 256, "channels": 3, "kernel": 7, "layout": "strided"},
            {"name": "blur-input-backward", "size": 256, "channels": 3, "kernel": 7, "backward": "input"},
            {"name": "blur-sigma-backward", "size": 256, "channels": 3, "kernel": 7, "backward": "sigma"},
            {"name": "blur-signed", "size": 256, "channels": 3, "kernel": 15, "signal": "signed"},
            {"name": "blur-constant", "size": 256, "channels": 3, "kernel": 15, "signal": "constant"},
            {"name": "blur-impulse", "size": 256, "channels": 3, "kernel": 15, "signal": "impulse"},
            {"name": "pyramid-gray", "size": 512, "channels": 1, "op": "ScalePyramid"},
            {"name": "pyramid-rgb", "size": 256, "channels": 3, "op": "ScalePyramid"},
            {"name": "pyramid-double", "size": 256, "channels": 1, "op": "ScalePyramid", "double": True},
            {"name": "pyramid-float64", "size": 512, "channels": 1, "op": "ScalePyramid", "dtype": "float64"},
        ]
    )
    return rows


def make_operation(case: dict[str, Any], device: torch.device) -> Callable[[], tuple[torch.Tensor, ...]]:
    """Construct inputs once; keep all public forward/backward work inside the callable."""
    torch.manual_seed(0)
    size = case["size"]
    shape = (case.get("batch", 1), case["channels"], size, size)
    dtype = getattr(torch, case.get("dtype", "float32"))
    value = torch.rand(shape, dtype=dtype, device=device)
    signal = case.get("signal")
    if signal == "signed":
        value = 2 * value - 1
    elif signal == "constant":
        value.fill_(1)
    elif signal == "impulse":
        value.zero_()
        value[..., size // 2, size // 2] = 1
    if case.get("layout") == "channels_last":
        value = value.contiguous(memory_format=torch.channels_last)
    elif case.get("layout") == "strided":
        value = torch.rand((*shape[:-1], 2 * size), dtype=dtype, device=device)[..., ::2]
    if case.get("op") == "ScalePyramid":
        pyramid = ScalePyramid(double_image=case.get("double", False)).to(device=device, dtype=dtype)

        def run_pyramid() -> tuple[torch.Tensor, ...]:
            levels, sigmas, distances = pyramid(value)
            return tuple(levels + sigmas + distances)

        return run_pyramid
    sigma = torch.tensor([[1.5, 2.1]], dtype=dtype, device=device)
    if case.get("batched_sigma"):
        sigma = torch.tensor([[1.5, 2.1], [0.8, 3.2]], dtype=dtype, device=device)
    backward = case.get("backward")
    value.requires_grad_(backward is not None)
    sigma.requires_grad_(backward == "sigma")
    loss_weights = torch.randn_like(value) if backward else None

    def run_blur() -> tuple[torch.Tensor, ...]:
        result = gaussian_blur2d(value, case["kernel"], sigma)
        if backward:
            arguments = (value, sigma) if backward == "sigma" else (value,)
            gradients = torch.autograd.grad(result, arguments, grad_outputs=loss_weights)
            return (result, *gradients)
        return (result,)

    return run_blur


def differences(actual: tuple[torch.Tensor, ...], reference: tuple[torch.Tensor, ...]) -> dict[str, Any]:
    """Report absolute error and scale-normalized error, avoiding relative error near zero."""
    assert len(actual) == len(reference)
    stats = []
    for current, baseline in zip(actual, reference):
        assert current.shape == baseline.shape and current.dtype == baseline.dtype
        delta = (current.double() - baseline.double()).abs()
        scale = baseline.double().abs().max().item()
        norm = baseline.double().norm().item()
        stats.append(
            {
                "max_abs": delta.max().item(),
                "mean_abs": delta.mean().item(),
                "max_abs_over_reference_max": delta.max().item() / max(scale, 1e-300),
                "relative_l2": delta.norm().item() / max(norm, 1e-300),
            }
        )
    return {"tensor_errors": stats}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default="cpu")
    parser.add_argument("--min-run-time", type=float, default=1.0)
    parser.add_argument("--save-reference", type=Path)
    parser.add_argument("--reference", type=Path)
    args = parser.parse_args()
    device = torch.device(args.device)
    torch.set_num_threads(1)
    # Keep the CUDA comparison independent of reduced-precision convolution defaults.
    torch.backends.cudnn.allow_tf32 = False
    root = Path.cwd().resolve()
    assert Path(kornia.__file__).resolve().is_relative_to(root), "Kornia must import from the measured checkout"
    print(f"Kornia: {kornia.__file__}; Python: {sys.executable}", flush=True)
    metadata = run_metadata(device)
    sources = ("kornia/filters/gaussian.py", "kornia/filters/filter.py", "kornia/geometry/transform/pyramid.py")
    metadata.update(
        seed=0,
        load=collect_load_metrics(),
        min_run_time=args.min_run_time,
        mkldnn_available=torch.backends.mkldnn.is_available(),
        cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
        implementation_sha256={name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in sources},
        harness_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        timing="public forward including allocation/kernel generation; labelled backward; one CPU thread",
    )
    print(versions_line(metadata), flush=True)
    print(f"# commit {metadata['git_commit']}; {metadata['platform']}; {device}; one CPU thread", flush=True)
    reference = torch.load(args.reference, weights_only=True, map_location="cpu") if args.reference else {}
    saved = {}
    results = []
    for case in cases():
        if device.type == "mps" and case.get("dtype") == "float64":
            continue
        operation = make_operation(case, device)
        outputs = tuple(tensor.detach().cpu().clone() for tensor in operation())
        sync = torch.mps.synchronize if device.type == "mps" else None
        median, spread = time_us(operation, min_run_time=args.min_run_time, sync=sync)
        batch = case.get("batch", 1)
        row = {
            "op": case.get("op", "gaussian_blur2d"),
            "backend": "kornia (eager)",
            "case": case["name"],
            "batch": batch,
            "channels": case["channels"],
            "height": case["size"],
            "width": case["size"],
            "dtype": case.get("dtype", "float32"),
            "median_us": median,
            "iqr_us": spread,
            "throughput_per_s": batch * 1e6 / median,
            "parameters": case,
        }
        if reference:
            row.update(differences(outputs, reference[case["name"]]))
        results.append(row)
        saved[case["name"]] = outputs
        print(f"{case['name']:25s} {median:12.1f} us +/- IQR {spread:10.1f}", flush=True)
    save_json(args.json, metadata, results)
    if args.save_reference:
        torch.save(saved, args.save_reference)


if __name__ == "__main__":
    main()
