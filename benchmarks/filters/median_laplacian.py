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

"""Public median/Laplacian A/B benchmark with matched float OpenCV references.

Run as ``python -m benchmarks.filters.median_laplacian`` from each checkout,
using the same harness and common.py. Native uint8 OpenCV is additionally
reported, labelled separately. Compile warmup is excluded; allocations included.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import collect_load_metrics, run_metadata, save_json, time_us

import kornia


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--device", default="cpu", choices=("cpu", "mps", "cuda"))
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--batches", default="1,8")
    parser.add_argument("--kernels", default="3,5,7")
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--min-run-time", type=float, default=0.5)
    args = parser.parse_args()
    root = Path.cwd().resolve()
    assert Path(kornia.__file__).resolve().is_relative_to(root), kornia.__file__
    print(f"Kornia: {kornia.__file__}; Python: {sys.executable}", flush=True)
    torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    device = torch.device(args.device)
    metadata = run_metadata(device)
    metadata.update(
        load=collect_load_metrics(),
        seed=0,
        min_run_time=args.min_run_time,
        mkldnn_available=torch.backends.mkldnn.is_available(),
        implementation_sha256={
            name: hashlib.sha256((root / name).read_bytes()).hexdigest()
            for name in ("kornia/filters/median.py", "kornia/filters/laplacian.py", "benchmarks/common.py")
        },
        harness_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    )
    try:
        import cv2

        cv2.setNumThreads(args.threads)
    except ImportError:
        cv2 = None
    sync = torch.mps.synchronize if device.type == "mps" else None
    rows = []
    for batch in map(int, args.batches.split(",")):
        images_u8 = np.random.default_rng(0).integers(0, 256, (batch, args.size, args.size, 3), dtype=np.uint8)
        images = images_u8.astype(np.float32) / 255
        value = torch.from_numpy(images).permute(0, 3, 1, 2).contiguous().to(device)
        for kernel in map(int, args.kernels.split(",")):
            for op in ("median_blur", "laplacian"):
                fn = getattr(kornia.filters, op)

                def eager(fn=fn, value=value, kernel=kernel):
                    return fn(value, kernel)

                backends = {"kornia (eager)": eager}
                if args.compile:
                    torch._dynamo.reset()
                    compiled = torch.compile(eager)
                    torch.testing.assert_close(compiled(), eager())
                    backends["kornia (compiled)"] = compiled
                if cv2 is not None:
                    if op == "laplacian":
                        weights = np.ones((kernel, kernel), dtype=np.float32)
                        weights[kernel // 2, kernel // 2] -= kernel * kernel
                        weights /= 2 * (kernel * kernel - 1)
                        backends["opencv (matched float32)"] = lambda weights=weights, images=images: [
                            cv2.filter2D(im, -1, weights) for im in images
                        ]
                        backends["opencv (native uint8 Laplacian)"] = lambda kernel=kernel, images_u8=images_u8: [
                            cv2.Laplacian(im, cv2.CV_32F, ksize=kernel) for im in images_u8
                        ]
                    elif kernel <= 5:
                        pad = kernel // 2

                        def cv_median(pad=pad, kernel=kernel, images=images):
                            return [
                                cv2.medianBlur(cv2.copyMakeBorder(im, pad, pad, pad, pad, cv2.BORDER_CONSTANT), kernel)[
                                    pad:-pad, pad:-pad
                                ]
                                for im in images
                            ]

                        backends["opencv (matched float32)"] = cv_median
                        backends["opencv (native uint8 median)"] = lambda kernel=kernel, images_u8=images_u8: [
                            cv2.medianBlur(im, kernel) for im in images_u8
                        ]
                    if "opencv (matched float32)" in backends:
                        expected = np.stack(backends["opencv (matched float32)"]())
                        np.testing.assert_allclose(
                            eager().permute(0, 2, 3, 1).cpu().numpy(), expected, atol=2e-6, rtol=1e-5
                        )
                for backend, operation in backends.items():
                    median, spread = time_us(
                        operation, min_run_time=args.min_run_time, sync=sync if backend.startswith("kornia") else None
                    )
                    row = {
                        "op": op,
                        "backend": backend,
                        "batch": batch,
                        "height": args.size,
                        "width": args.size,
                        "kernel_size": kernel,
                        "dtype": "uint8" if "uint8" in backend else "float32",
                        "median_us": median,
                        "iqr_us": spread,
                        "throughput_per_s": batch * 1e6 / median,
                    }
                    rows.append(row)
                    print(f"{op:12} k={kernel} b={batch} {backend:36} {median:10.1f} us", flush=True)
    save_json(args.json, metadata, rows)


if __name__ == "__main__":
    main()
