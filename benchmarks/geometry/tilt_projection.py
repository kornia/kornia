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

"""Public sensor-tilt and camera-distortion latency for paired base/branch runs.

Angles are 0.05 and -0.08 radians. Distortion calls operate on 64 image points per camera,
with all inputs prepared outside timing. Shared helpers provide warmup, median/IQR and
checkout metadata. This compares the same Kornia APIs, without external substitutes.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import KorniaRows, add_flagship_args, batch_list, finish_run, run_batch_sweep, setup_run, start_run

from kornia.geometry.calibration import distort_points, undistort_points
from kornia.geometry.calibration.distort import tilt_projection


def build_ops(batch, device, dtype, do_compile, skip_compile=frozenset(), selected_ops=None):
    taux = torch.full((batch, 1), 0.05, device=device, dtype=dtype)
    tauy = torch.full((batch, 1), -0.08, device=device, dtype=dtype)
    K = torch.tensor([[800.0, 0.0, 320.0], [0.0, 760.0, 240.0], [0.0, 0.0, 1.0]])
    K = K.expand(batch, 3, 3).to(device=device, dtype=dtype)
    points = torch.linspace(20.0, 400.0, 128, device=device, dtype=dtype).reshape(1, 64, 2).expand(batch, 64, 2)
    dist = torch.zeros(batch, 14, device=device, dtype=dtype)
    dist[:, 12] = 0.05
    dist[:, 13] = -0.08
    rows = KorniaRows(device, do_compile, skip_compile=skip_compile)
    operations = {
        "tilt_projection": (tilt_projection, (taux, tauy)),
        "tilt_projection_inverse": (tilt_projection, (taux, tauy, True)),
        "distort_points_14": (distort_points, (points, K, dist)),
        "undistort_points_14": (undistort_points, (points, K, dist)),
    }
    return {
        name: rows(name, target, *inputs)
        for name, (target, inputs) in operations.items()
        if selected_ops is None or name in selected_ops
    }, rows.compile_failures


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_flagship_args(
        parser,
        ops=("tilt_projection", "tilt_projection_inverse", "distort_points_14", "undistort_points_14"),
        batches="1,32,256,1024",
    )
    args = parser.parse_args()
    device, dtype, sync = setup_run(args, opencv=False)
    meta = start_run("tilt_projection", args, device, units="cameras/s", regimes=("public eager camera API",))
    backends = ["kornia (eager)"] + (["kornia (compiled)"] if args.compile else [])
    results = run_batch_sweep(
        batch_list(args),
        lambda batch: build_ops(batch, device, dtype, args.compile, args.skip_compile_ops, args.ops),
        backends,
        row_fields=lambda batch: {"batch": batch, "dtype": args.dtype},
        sync=sync,
        min_run_time=args.min_run_time,
    )
    finish_run(args, "tilt_projection", meta, results)


if __name__ == "__main__":
    main()
