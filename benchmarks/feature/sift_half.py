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
"""CPU/CUDA half-precision public quadratic interpolation and SIFT forwards.

Run from each compared checkout root, one process per device/dtype::

    python -m benchmarks.feature.sift_half --seq /data/graf --device cuda \
        --dtype float16 --threads 1 --json half.json

Includes native half inputs and outputs (no autocast). SIFT uses Graf image 1,
4096 features, both descriptor backends, and includes detection/orientation/description.
Matching quality is not measured. Unsupported combinations are recorded as failures.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import torch

import kornia
from benchmarks.feature.sift_scale_space import _build, _image_path, _load_image
from kornia.geometry.subpix import conv_quad_interp3d, iterative_quad_interp3d

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import collect_load_metrics, run_metadata, save_json, time_us, versions_line


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seq", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), required=True)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--min-run-time", type=float, default=1.0)
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--outputs", type=Path, help="Optional local output tensors for A/B checks")
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    root = Path(kornia.__file__).resolve().parents[1]
    if root != Path.cwd().resolve():
        raise RuntimeError("Run from the measured checkout root; wrong editable import")
    print(f"kornia.__file__ = {kornia.__file__}", flush=True)
    print(f"sys.executable = {sys.executable}", flush=True)
    torch.manual_seed(0)
    torch.set_num_threads(args.threads)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device, dtype = torch.device(args.device), getattr(torch, args.dtype)
    image_path = _image_path(args.seq, 1)
    image = _load_image(image_path, device).to(dtype)
    volume = (torch.randn(1, 1, 5, 128, 128) * 0.01).to(device=device, dtype=dtype)
    meta = run_metadata(device)
    sources = ("geometry/subpix/spatial_soft_argmax.py", "feature/sift/scale_space.py", "feature/siftdesc.py")
    meta.update(
        load=collect_load_metrics(),
        seed=0,
        timer_num_threads=args.threads,
        min_run_time=args.min_run_time,
        dtype=args.dtype,
        autocast=False,
        compile=False,
        input_sha256=hashlib.sha256(image_path.read_bytes()).hexdigest(),
        implementation_sha256={
            name: hashlib.sha256((root / "kornia" / name).read_bytes()).hexdigest() for name in sources
        },
        timing="public eager inference forward; no matching or I/O",
        memory="CUDA peak allocated tensor bytes above resident inputs/model; includes returned outputs",
    )
    print(f"# {meta['git_commit']} | {meta['platform']} | {meta.get('cuda_device', device)}", flush=True)
    print(versions_line(meta), flush=True)
    rows, outputs = [], {}
    with torch.inference_mode():
        for name in ("conv_quad_interp3d", "iterative_quad_interp3d", "sift-patch", "sift-pyramid"):
            row = {
                "op": "SIFTFeatureScaleSpace" if name.startswith("sift") else name,
                "backend": name,
                "batch": 1,
                "dtype": args.dtype,
                "median_us": None,
                "iqr_us": None,
                "throughput_per_s": None,
            }
            try:
                if name.startswith("sift"):
                    model = _build(name.split("-")[1], 4096, device)
                    data = image
                else:
                    model = conv_quad_interp3d if name == "conv_quad_interp3d" else iterative_quad_interp3d
                    data = volume
                row.update(height=data.shape[-2], width=data.shape[-1])

                def run(model=model, data=data, name=name):
                    if name.startswith("sift"):
                        return model(data)
                    return model(data, n_iters=5, strict_maxima_bonus=0.0)

                for _ in range(3):
                    run()
                med, iqr = time_us(run, min_run_time=args.min_run_time, num_threads=args.threads)
                peak = None
                if device.type == "cuda":
                    torch.cuda.synchronize()
                    resident = torch.cuda.memory_allocated()
                    torch.cuda.reset_peak_memory_stats()
                    result = run()
                    torch.cuda.synchronize()
                    peak = torch.cuda.max_memory_allocated() - resident
                else:
                    result = run()
                if name.startswith("sift"):
                    lafs, responses, descriptors = result
                    filled = kornia.feature.laf_is_filled(lafs)
                    finite = bool(
                        torch.isfinite(lafs).all()
                        & torch.isfinite(descriptors).all()
                        & torch.isfinite(responses[filled]).all()
                    )
                    row["features"] = int(filled.sum())
                else:
                    finite = all(bool(torch.isfinite(t).all()) for t in result)
                row.update(
                    median_us=med,
                    iqr_us=iqr,
                    throughput_per_s=1e6 / med,
                    finite=finite,
                    output_dtypes=[str(t.dtype) for t in result],
                    peak_extra_bytes=peak,
                )
                if not finite:
                    row["error"] = "Nonfinite public outputs"
                outputs[name] = tuple(t.cpu() for t in result)
                del result
            except (RuntimeError, NotImplementedError, ValueError) as exc:
                row["error"] = f"{type(exc).__name__}: {exc}"
            rows.append(row)
            print(row, flush=True)
            save_json(args.json, meta, rows)
    if args.outputs:
        torch.save(outputs, args.outputs)


if __name__ == "__main__":
    main()
