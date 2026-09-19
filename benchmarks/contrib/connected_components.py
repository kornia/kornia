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

"""Compare exact block CCL and pooling, validating partitions with optional SciPy.

Run from the repository root: python -m benchmarks.contrib.connected_components
--device cuda --sizes 256 1024 --calibrate --json results.json

Both backends receive identical resident float32 masks. Reference validation and
transfers to CPU are outside timing. --calibrate finds the minimum *correct*
pooling budget by doubling then binary search against SciPy, outside timing.
This oracle-tuned baseline favors pooling: users normally do not know this
budget. The default 100-iteration baseline is also measured, but incorrect
partitions must never be treated as equivalent-work speedups. SciPy itself is
not timed because it labels host arrays, whereas these APIs label device tensors.
CUDA rows include peak allocated bytes above the resident input, including the
output and temporary tensors but excluding the allocator's reserved cache.
"""

from __future__ import annotations

import argparse
import sys
from functools import partial
from pathlib import Path

import torch

import kornia

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import run_metadata, save_json, time_us


def reference_labels(image):
    try:
        import numpy as np
        from scipy.ndimage import label
    except ImportError:
        return None
    return [label(mask, structure=np.ones((3, 3)))[0] for mask in image[:, 0].cpu().numpy()]


def same_partition(labels, references):
    if references is None:
        return None
    import numpy as np

    for actual, expected in zip(labels[:, 0].cpu().numpy(), references):
        if not np.array_equal(actual != 0, expected != 0):
            return False
        # A bijection between actual and reference IDs establishes equality of
        # partitions, independent of their particular numerical label values.
        pairs = np.unique(np.stack((actual.ravel(), expected.ravel()), axis=1), axis=0)
        if len(pairs) != len(np.unique(actual)) or len(pairs) != len(np.unique(expected)):
            return False
    return True


def minimum_iterations(image, reference):
    low, high = 0, 1
    maximum = image.shape[-2] * image.shape[-1]
    while not same_partition(kornia.contrib.connected_components(image, high), reference):
        low, high = high, min(high * 2, maximum)
        if low == high:
            raise RuntimeError("Pooling did not match the reference even at the pixel-count budget")
    while high - low > 1:
        middle = (high + low) // 2
        if same_partition(kornia.contrib.connected_components(image, middle), reference):
            high = middle
        else:
            low = middle
    return high


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--sizes", type=int, nargs="+", default=[256, 1024])
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--calibrate", action="store_true")
    parser.add_argument("--min-run-time", type=float, default=1.0)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    device = torch.device(args.device)
    torch.set_num_threads(1)
    print(f"Python: {sys.executable}\nKornia: {kornia.__file__}", flush=True)
    metadata = run_metadata(device)
    print(metadata, flush=True)
    results = []
    for size in args.sizes:
        for case in ("dense", "random", "isolated", "snake"):
            generator = torch.Generator().manual_seed(0)
            height, width = (33, 65) if case == "snake" else (size, size)
            if case == "snake" and size != args.sizes[0]:
                continue
            image = torch.zeros(args.batch, 1, height, width)
            if case == "dense":
                image.fill_(1)
            elif case == "random":
                image = (torch.rand(image.shape, generator=generator) < 0.35).float()
            elif case == "isolated":
                image[..., ::2, ::2] = 1
            else:
                image[..., ::2, :] = 1
                image[..., 1::4, -1] = 1
                image[..., 3::4, 0] = 1
            image = image.to(device)
            reference = reference_labels(image)
            if reference is None:
                print("SciPy unavailable: skipping partition validation and calibration", flush=True)
            methods = [("pool100", 100)]
            if args.calibrate and reference is not None:
                methods.append(("pool_calibrated", minimum_iterations(image, reference)))
            if hasattr(kornia.contrib, "connected_components_union_find"):
                methods.append(("block_union_find", None))
            for backend, iterations in methods:
                if iterations is None:
                    operation = partial(kornia.contrib.connected_components_union_find, image)
                else:
                    operation = partial(kornia.contrib.connected_components, image, iterations)
                correct = same_partition(operation(), reference)
                if backend == "block_union_find" and correct is False:
                    raise AssertionError(f"Incorrect partition: {case}, {size}")
                sync = torch.mps.synchronize if device.type == "mps" else None
                median, iqr = time_us(operation, min_run_time=args.min_run_time, sync=sync)
                peak_extra_bytes = None
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                    allocated = torch.cuda.memory_allocated(device)
                    torch.cuda.reset_peak_memory_stats(device)
                    output = operation()
                    torch.cuda.synchronize(device)
                    peak_extra_bytes = torch.cuda.max_memory_allocated(device) - allocated
                    del output
                row = {
                    "op": "connected_components",
                    "backend": backend,
                    "case": case,
                    "batch": args.batch,
                    "height": height,
                    "width": width,
                    "dtype": "float32",
                    "iterations": iterations,
                    "correct": correct,
                    "median_us": median,
                    "iqr_us": iqr,
                    "peak_extra_bytes": peak_extra_bytes,
                    "throughput_per_s": args.batch * 1e6 / median,
                }
                results.append(row)
                print(row, flush=True)
    if args.json:
        save_json(args.json, metadata, results)


if __name__ == "__main__":
    main()
