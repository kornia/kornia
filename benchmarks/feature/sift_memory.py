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
"""Public SIFT CPU allocation and runtime probes for texture and batch edge cases.

Run as a module from each measured checkout with the same explicit interpreter::

    python -m benchmarks.feature.sift_memory --seq /data/graf \
        --expected-checkout "$PWD" --json sift-memory.json

The 512x512 checkerboard stresses candidate rejection. Graf images 1 and 2 are
resized to 192x240 to check small images and batch two. Inputs are prepared before
timing. Memory is profiled in a separate inference forward: peak live and total
allocated PyTorch CPU tensor bytes, not process RSS or measured memory bandwidth.
Matching quality is measured separately by ``sift_scale_space.py`` on full Graf.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

import kornia
from benchmarks.feature.sift_scale_space import _build, _image_path, _load_image, _profile_cpu_allocations

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import collect_load_metrics, run_metadata, save_json, time_us


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seq", type=Path, required=True)
    parser.add_argument("--expected-checkout", type=Path, required=True)
    parser.add_argument("--json", type=Path, required=True)
    args = parser.parse_args()
    print(f"# interpreter: {sys.executable}\n# kornia: {kornia.__file__}", flush=True)
    root = Path(kornia.__file__).resolve().parents[1]
    if root != args.expected_checkout.resolve():
        raise RuntimeError(f"Wrong Kornia checkout: {root}")
    torch.manual_seed(0)
    torch.set_num_threads(1)
    device = torch.device("cpu")
    meta = run_metadata(device)
    meta.update(
        load=collect_load_metrics(),
        seed=0,
        inference_mode=True,
        timing="entire public SIFTFeatureScaleSpace forward; input preparation excluded",
        memory="separate forward, PyTorch CPU allocator events; excludes native-library workspaces and RSS",
        implementation_sha256={
            str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (
                Path(__file__),
                Path(__file__).with_name("sift_scale_space.py"),
                root / "kornia/feature/sift/scale_space.py",
                root / "kornia/feature/integrated.py",
            )
        },
    )
    axis = torch.arange(512)
    checker = ((axis[:, None] + axis[None, :]) % 2).float()[None, None]
    small = [
        F.interpolate(
            _load_image(_image_path(args.seq, index), device), size=(192, 240), mode="bilinear", align_corners=False
        )
        for index in (1, 2)
    ]
    rows = []
    for name, image, count in (
        ("checkerboard-512", checker, 4096),
        ("graf-small", small[0], 512),
        ("graf-small-batch2", torch.cat(small), 512),
    ):
        model = _build("pyramid", count, device)
        start = time.perf_counter()
        output = model(image)
        warm_seconds = time.perf_counter() - start
        filled = kornia.feature.laf_is_filled(output[0]).sum(1).tolist()
        if not all(torch.isfinite(value).all() for value in output):
            raise RuntimeError(f"Nonfinite output: {name}")
        del output

        def run(model=model, image=image):
            return model(image)

        median, iqr = time_us(run, min_run_time=max(2.0, 5 * warm_seconds))
        row = dict(
            op="SIFTFeatureScaleSpace",
            backend="pyramid",
            case=name,
            batch=image.shape[0],
            height=image.shape[-2],
            width=image.shape[-1],
            num_features=count,
            features=filled,
            dtype="float32",
            median_us=median,
            iqr_us=iqr,
            throughput_per_s=image.shape[0] * 1e6 / median,
            input_sha256=hashlib.sha256(image.numpy().tobytes()).hexdigest(),
            **_profile_cpu_allocations(run),
        )
        rows.append(row)
        print(row, flush=True)
        save_json(args.json, meta, rows)


if __name__ == "__main__":
    main()
