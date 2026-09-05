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
"""Historical public scale-space SIFT runtime, including true CUDA image batches.

Run one configuration per process from the checkout being measured. For old
worktrees, invoke this file with runpy from stdin (see graf_benchmark.md), so the
editable install cannot silently replace the historical Kornia import.

Uses each release's native SIFTFeatureScaleSpace preset with 4096 features,
orientation and RootSIFT enabled. Historical refinement algorithms differ; this
is a release comparison, not an isolated ablation or a quality-equivalence claim.
The input is Oxford graf img1 repeated into a contiguous batch. CPU uses one
thread. Transfers, initialization, compilation, matching, and RANSAC are outside
steady-state timing. Raw median/IQR are BATCH latency; divide by B for ms/image.
OpenCV is a one-thread uint8 CPU detectAndCompute call plus RootSIFT normalization,
using its native contrast/edge filtering. It is not a GPU or batched baseline.
The features_per_image field counts nonzero LAF output slots for Kornia, not
verified valid detections: historical releases could retain rejected LAFs.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

import kornia
from kornia.feature import SIFTFeatureScaleSpace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import run_metadata, save_json, time_us, versions_line


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--expected-checkout", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--batch", type=int, choices=(1, 4, 8), default=1)
    parser.add_argument("--series", choices=("0.8.2", "0.8.3", "current", "current + compile", "OpenCV"), required=True)
    parser.add_argument("--nf", type=int, default=4096)
    parser.add_argument("--min-run-time", type=float, default=5.0)
    parser.add_argument("--json", type=Path, required=True)
    args = parser.parse_args()
    if importlib.util.find_spec("cv2") is None:
        raise SystemExit("SKIP: this benchmark requires the optional opencv-python package")
    import cv2

    imported_root = Path(kornia.__file__).resolve().parents[1]
    print(f"# interpreter: {sys.executable}\n# kornia: {kornia.__file__}", flush=True)
    if imported_root != args.expected_checkout.resolve():
        raise RuntimeError(f"Wrong Kornia checkout: {imported_root}")
    if args.device == "cpu" and args.batch != 1:
        parser.error("CPU comparisons use batch one")
    if args.series == "OpenCV" and (args.device != "cpu" or args.batch != 1):
        parser.error("OpenCV is a CPU batch-one reference")
    torch.manual_seed(0)
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device(args.device)
    pixels = cv2.imread(str(args.image), cv2.IMREAD_GRAYSCALE)
    if pixels is None:
        raise FileNotFoundError(args.image)
    meta = run_metadata(device)
    compiled = args.series == "current + compile"
    meta.update(
        series=args.series,
        seed=0,
        image=args.image.name,
        input_sha256=hashlib.sha256(args.image.read_bytes()).hexdigest(),
        num_features=args.nf,
        compile=compiled,
        compile_scope=["scale_pyr", "resp", "subpix"] if compiled else [],
        compile_dynamic=True if compiled else None,
        timing="feature extraction only; raw batch latency; transfers/compilation/matching/RANSAC excluded",
        batch_regime="same graf image repeated as a true tensor batch",
        opencv_num_threads=cv2.getNumThreads(),
        cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
        cudnn_benchmark=torch.backends.cudnn.benchmark,
        historical_presets="native release defaults; refinement algorithms and quality differ",
        min_run_time=args.min_run_time,
    )
    print(f"# {meta['git_commit']} | {args.series} | {device} | B={args.batch}", flush=True)
    print(versions_line(meta), flush=True)
    if args.series == "OpenCV":
        model = cv2.SIFT_create(nfeatures=args.nf, nOctaveLayers=3, contrastThreshold=0.04, edgeThreshold=10, sigma=1.6)

        def run():
            keypoints, descriptors = model.detectAndCompute(pixels, None)
            if descriptors is not None:
                descriptors = np.sqrt(descriptors / (np.abs(descriptors).sum(axis=1, keepdims=True) + 1e-8))
            return keypoints, descriptors

    else:
        kwargs = {"compile_modules": ["scale_pyr", "resp", "subpix"]} if compiled else {}
        model = SIFTFeatureScaleSpace(
            num_features=args.nf, upright=False, rootsift=True, device=device, **kwargs
        ).eval()
        inputs = torch.from_numpy(pixels).to(device=device, dtype=torch.float32)[None, None].div(255)
        inputs = inputs.repeat(args.batch, 1, 1, 1)
        meta["model_repr"] = repr(model)

        def run():
            return model(inputs)

    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    run()
    if device.type == "cuda":
        torch.cuda.synchronize()
    initial_seconds = time.perf_counter() - start
    print(f"# Initial call (including any compilation): {initial_seconds:.3f} s", flush=True)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        memory_before = torch.cuda.memory_allocated(device)
    start = time.perf_counter()
    output = run()
    if device.type == "cuda":
        torch.cuda.synchronize()
    warm_seconds = time.perf_counter() - start
    peak_bytes = torch.cuda.max_memory_allocated(device) - memory_before if device.type == "cuda" else None
    features = (
        [len(output[0])] if args.series == "OpenCV" else output[0].ne(0).any(dim=-1).any(dim=-1).sum(dim=-1).tolist()
    )
    del output
    median, iqr = time_us(run, min_run_time=max(args.min_run_time, 5 * warm_seconds))
    if not math.isfinite(median):
        raise RuntimeError(f"Timing failed: {args.series}, {device}, B={args.batch}")
    row = {
        "op": "OpenCV SIFT" if args.series == "OpenCV" else "SIFTFeatureScaleSpace",
        "series": args.series,
        "backend": "opencv"
        if args.series == "OpenCV"
        else ("kornia (selectively compiled)" if compiled else "kornia (eager)"),
        "device": args.device,
        "batch": args.batch,
        "height": pixels.shape[0],
        "width": pixels.shape[1],
        "dtype": "uint8 input/float32 descriptor" if args.series == "OpenCV" else "float32",
        "median_us": median,
        "iqr_us": iqr,
        "throughput_per_s": args.batch * 1e6 / median,
        "features_per_image": features,
        "peak_extra_cuda_bytes": peak_bytes,
        "initial_call_seconds": initial_seconds,
    }
    save_json(args.json, meta, [row])
    print(f"# {median / args.batch / 1000:.3f} ms/image, IQR {iqr / args.batch / 1000:.3f} ms/image", flush=True)
    print(f"# features/image: {features}", flush=True)


if __name__ == "__main__":
    main()
