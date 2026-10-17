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

"""Synthetic RANSAC latency and inlier recovery, eager versus RANSAC(compile=True).

Uses preloaded float inputs, fixed planar/two-view scenes, 0.5-pixel noise, and a 1.5-pixel
threshold (camera-normalized for essential matrices). Public forward includes staging and
refinement; construction and first-call compilation/loading are excluded. Every seed has
its own repeated median/IQR and recall/false-positive count, so stopping variance is visible.
Run the identical harness from each checkout; the source path is printed and hashes recorded.
These controlled scenes diagnose latency and recovery, not real-world pose accuracy.

    python benchmarks/geometry/ransac_compile_synthetic.py --compile --json /tmp/ransac.json
    python benchmarks/geometry/ransac_compile_synthetic.py --compile --confidence 1 --max-samples 2048
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import os
import sys
from functools import partial
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(1, str(ROOT / "benchmarks"))

from common import add_flagship_args, finish_run, run_batch_sweep, setup_run, start_run  # noqa: E402

import kornia  # noqa: E402
from kornia.geometry import RANSAC, transform_points  # noqa: E402
from kornia.geometry.conversions import axis_angle_to_rotation_matrix, convert_points_from_homogeneous  # noqa: E402

MODELS = {"H": "homography", "F7": "fundamental", "F8": "fundamental_8pt", "E": "essential"}


def scene(model: str, n: int, ratio: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fixed scenes matching the RANSAC test geometry; independent of the estimator seed."""
    g = torch.Generator().manual_seed(n)
    f64 = torch.float64
    outliers = round(n * (1 - ratio))
    if model == "H":
        h = torch.tensor([[1.1, 0.05, 20.0], [0.02, 0.95, -10.0], [1e-4, 2e-4, 1.0]], dtype=f64)
        p = torch.rand(n, 2, generator=g, dtype=f64) * 600
        q = transform_points(h[None], p[None])[0]
        extent = torch.tensor([600.0, 600.0], dtype=f64)
    else:
        camera = torch.tensor([[800.0, 0.0, 320.0], [0.0, 800.0, 240.0], [0.0, 0.0, 1.0]], dtype=f64)
        rotation = axis_angle_to_rotation_matrix(torch.tensor([[0.05, -0.1, 0.02]], dtype=f64))[0]
        translation = torch.tensor([1.0, 0.1, 0.2], dtype=f64)
        xyz = torch.randn(n, 3, generator=g, dtype=f64) * torch.tensor([2.0, 2.0, 1.0], dtype=f64)
        xyz += torch.tensor([0.0, 0.0, 8.0], dtype=f64)
        p = convert_points_from_homogeneous(xyz @ camera.T)
        q = convert_points_from_homogeneous((xyz @ rotation.T + translation) @ camera.T)
        extent = torch.tensor([640.0, 480.0], dtype=f64)
    q = q + 0.5 * torch.randn(n, 2, generator=g, dtype=f64)
    q[:outliers] = torch.rand(outliers, 2, generator=g, dtype=f64) * extent
    if model == "E":
        center = torch.tensor([320.0, 240.0], dtype=f64)
        p, q = (p - center) / 800, (q - center) / 800
    return p, q, torch.arange(n) >= outliers


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_flagship_args(parser, ops=tuple(MODELS), batches="", size=0)
    parser.add_argument("--sizes", default="500,2000,5000")
    parser.add_argument("--ratios", default="0.2,0.5")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--confidence", type=float, default=0.999)
    parser.add_argument("--max-samples", type=int, default=20480)
    parser.add_argument("--sample-batch", default="auto", help="auto or a fixed number of minimal samples")
    args = parser.parse_args()
    sizes = [int(v) for v in args.sizes.split(",")]
    ratios = [float(v) for v in args.ratios.split(",")]
    seeds = [int(v) for v in args.seeds.split(",")]
    if min(sizes) < 9 or any(not 0 < r <= 1 for r in ratios) or min(seeds) < 0:
        parser.error("sizes must be >=9, ratios in (0,1], and seeds nonnegative")
    if args.device not in ("cpu", "cuda") or args.dtype not in ("float32", "float64"):
        parser.error("use CPU/CUDA float32/float64")
    batch = args.sample_batch if args.sample_batch == "auto" else int(args.sample_batch)
    device, dtype, sync = setup_run(args, opencv=False)
    meta = start_run(
        "Synthetic RANSAC",
        args,
        device,
        units="pairs/s",
        regimes=[
            "Public RANSAC.forward; first call excluded; independent timing per RANSAC seed",
            "Eager and compiled streams differ; confidence=1 compares a fixed sample budget",
        ],
    )
    print(f"# interpreter: {sys.executable}", flush=True)
    assert Path(kornia.__file__).resolve().is_relative_to(ROOT)
    meta.update(
        sizes=sizes,
        inlier_ratios=ratios,
        seeds=seeds,
        compile=args.compile,
        artifact_enabled=os.environ.get("KORNIA_RANSAC_AOT", "1") != "0",
        confidence=args.confidence,
        max_samples=args.max_samples,
        sample_batch=batch,
        noise_px=0.5,
        inlier_threshold_px=1.5,
        min_run_time=args.min_run_time,
        source_sha256={
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [
                Path(__file__).resolve(),
                ROOT / "kornia/geometry/ransac.py",
                ROOT / "kornia/geometry/_ransac_program.py",
                ROOT / "kornia/geometry/homography.py",
                ROOT / "kornia/geometry/epipolar/fundamental.py",
                ROOT / "kornia/geometry/epipolar/essential.py",
            ]
        },
    )
    quality = {}
    backends = ["kornia (eager)"] + (["kornia (compiled)"] if args.compile else [])

    def build(config):
        n, ratio, seed = config
        ops = {}
        for name, model_type in MODELS.items():
            if args.ops and name not in args.ops:
                continue
            p, q, truth = scene(name, n, ratio)
            p, q, truth = p.to(device, dtype), q.to(device, dtype), truth.to(device)
            cells = {}
            for backend in backends:
                estimator = RANSAC(
                    model_type,
                    inl_th=1.5 / (800 if name == "E" else 1),
                    seed=seed,
                    confidence=args.confidence,
                    max_samples=args.max_samples,
                    batch_size=batch,
                    compile=backend.endswith("(compiled)"),
                )
                fn = partial(estimator, p, q)
                model, mask = fn()  # Warm first call before the timed sweep, including compilation.
                quality[name, backend, config] = {
                    "recall": float((mask & truth).sum()) / int(truth.sum()),
                    "false_positives": int((mask & ~truth).sum()),
                    "support": int(mask.sum()),
                    "failed": bool((model == 0).all()),
                }
                cells[backend] = fn
            ops[name] = cells
        return ops, {}

    configs = list(itertools.product(sizes, ratios, seeds))
    results = run_batch_sweep(
        configs,
        build,
        backends,
        sync=sync,
        units="pairs/s",
        row_fields=lambda c: {"batch": 1, "n": c[0], "inlier_ratio": c[1], "seed": c[2]},
        label_fn=lambda c: f"N={c[0]} inliers={c[1]:.0%} seed={c[2]}",
        items_fn=lambda c: 1,
        min_run_time=args.min_run_time,
    )
    for row in results:
        config = row["n"], row["inlier_ratio"], row["seed"]
        row.update(quality[row["op"], row["backend"], config])
    finish_run(args, "geometry-ransac-compile-synthetic", meta, results)


if __name__ == "__main__":
    main()
