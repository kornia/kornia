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
"""Public quadratic-interpolation A/B benchmark on seeded response volumes.

Run from each measured checkout root with the same harness::

    python -m benchmarks.geometry.quad_interp --device cuda --threads 1 --json results.json

Eager forward under inference_mode, including NMS, interpolation and output construction.
Random volumes stress dense candidate sets; smooth DoG-like blobs and flat inputs cover
sparse and empty sets. No cross-library equivalent shares this function's contract.
CUDA peak allocated tensor memory is measured separately from synchronized timing.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import torch

import kornia
from kornia.geometry.subpix import conv_quad_interp3d, iterative_quad_interp3d

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import collect_load_metrics, run_metadata, save_json, time_us, versions_line


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--min-run-time", type=float, default=1.0)
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--outputs", type=Path, help="Optional local tensors for exact A/B comparisons")
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    if Path(kornia.__file__).resolve().parents[1] != Path.cwd().resolve():
        raise RuntimeError("Run from the measured checkout root; wrong editable import")
    print(f"kornia.__file__ = {kornia.__file__}", flush=True)
    print(f"sys.executable = {sys.executable}", flush=True)
    torch.manual_seed(0)
    torch.set_num_threads(args.threads)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(args.device)
    meta = run_metadata(device)
    source = Path(kornia.__file__).parent / "geometry/subpix/spatial_soft_argmax.py"
    meta.update(
        load=collect_load_metrics(),
        seed=0,
        timer_num_threads=args.threads,
        min_run_time=args.min_run_time,
        compile=False,
        implementation_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        harness_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        timing="entire public forward, eager inference; NMS and output allocation included",
        memory="CUDA peak allocated bytes above resident inputs, separate forward; includes returned outputs",
    )
    print(f"# {meta['git_commit']} | {meta['platform']} | {meta.get('cuda_device', device)}", flush=True)
    print(versions_line(meta), flush=True)
    cases = [(f"dense-{h}", torch.randn(1, 1, 5, h, h) * 0.01) for h in (32, 128, 320)]
    dd, yy, xx = torch.meshgrid(torch.arange(5), torch.arange(128), torch.arange(128), indexing="ij")
    smooth = torch.zeros(5, 128, 128)
    for y, x in ((20, 30), (60, 50), (90, 95)):
        smooth += 0.1 * torch.exp(
            -((dd - 2.2).square() + ((yy - y - 0.15) / 3).square() + ((xx - x + 0.2) / 3).square())
        )
    cases += [("sparse", smooth[None, None]), ("flat", torch.zeros(1, 1, 5, 128, 128))]
    rows, outputs = [], {}
    with torch.inference_mode():
        for case, host in cases:
            volume = host.to(device)
            for op in (conv_quad_interp3d, iterative_quad_interp3d):

                def run(op=op, volume=volume):
                    return op(volume, n_iters=5, strict_maxima_bonus=0.0)

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
                outputs[f"{case}/{op.__name__}"] = tuple(t.cpu() for t in result)
                del result
                row = {
                    "op": op.__name__,
                    "backend": "kornia (eager)",
                    "case": case,
                    "batch": host.shape[0],
                    "depth": host.shape[2],
                    "height": host.shape[3],
                    "width": host.shape[4],
                    "dtype": "float32",
                    "median_us": med,
                    "iqr_us": iqr,
                    "throughput_per_s": 1e6 / med,
                    "peak_extra_bytes": peak,
                }
                rows.append(row)
                print(row, flush=True)
                save_json(args.json, meta, rows)
    if args.outputs:
        torch.save(outputs, args.outputs)


if __name__ == "__main__":
    main()
