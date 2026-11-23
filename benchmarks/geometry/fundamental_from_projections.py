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

"""Public camera-operation latency for a base/branch comparison.

Run this identical harness from both checkouts. The seeded camera pairs use focal lengths near
800 pixels, a 0.2-radian rotation and translation of 0.2 to 1.2 world units. Inputs are prepared
outside timing. Each item is one camera pair/composition; no unrelated library substitutes for it.
Uses common setup, sustained CPU warmup, repeated median/IQR, and checkout provenance.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import KorniaRows, add_flagship_args, batch_list, finish_run, run_batch_sweep, setup_run, start_run

from kornia.geometry.conversions import axis_angle_to_rotation_matrix
from kornia.geometry.epipolar import fundamental_from_projections


def build_ops(batch, device, dtype, do_compile, skip_compile=frozenset()):
    K = torch.tensor([[800.0, 0.0, 320.0], [0.0, 760.0, 240.0], [0.0, 0.0, 1.0]])
    K = K.expand(batch, 3, 3).to(device=device, dtype=dtype)
    R = axis_angle_to_rotation_matrix(torch.tensor([[0.1, -0.2, 0.05]]))
    R = R.expand(batch, 3, 3).to(device=device, dtype=dtype)
    t = torch.stack((torch.linspace(0.2, 1.2, batch), torch.full((batch,), 0.1), torch.full((batch,), 0.02)), -1)
    t = t[..., None].to(device=device, dtype=dtype)
    eye = torch.eye(3, device=device, dtype=dtype).expand(batch, 3, 3)
    P1 = K @ torch.cat((eye, torch.zeros_like(t)), -1)
    P2 = K @ torch.cat((R, t), -1)
    rows = KorniaRows(device, do_compile, skip_compile=skip_compile)
    return {
        "fundamental_from_projections": rows("fundamental_from_projections", fundamental_from_projections, P1, P2)
    }, rows.compile_failures


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_flagship_args(parser, ops=("fundamental_from_projections",), batches="1,32,256,1024")
    args = parser.parse_args()
    device, dtype, sync = setup_run(args, opencv=False)
    meta = start_run(
        "fundamental_from_projections", args, device, units="camera pairs/s", regimes=("public eager camera API",)
    )
    backends = ["kornia (eager)"] + (["kornia (compiled)"] if args.compile else [])
    results = run_batch_sweep(
        batch_list(args),
        lambda batch: build_ops(batch, device, dtype, args.compile, args.skip_compile_ops),
        backends,
        row_fields=lambda batch: {"batch": batch, "dtype": args.dtype},
        sync=sync,
        min_run_time=args.min_run_time,
    )
    finish_run(args, "fundamental_from_projections", meta, results)


if __name__ == "__main__":
    main()
