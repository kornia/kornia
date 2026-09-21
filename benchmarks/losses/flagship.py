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

"""Flagship losses benchmark: kornia.losses forward + backward, vs torchvision where it has one.

Losses run inside training steps, so every row times a forward **and** the backward pass to the
prediction (``torch.autograd.grad``), not the forward alone. One representative per family:

=================================  ======================================  ============================
kornia.losses                      torchvision                             task
=================================  ======================================  ============================
``binary_focal_loss_with_logits``  ``ops.sigmoid_focal_loss``              dense binary detection
``focal_loss``                     —                                       21-class segmentation
``dice_loss``                      —                                       21-class segmentation
``ssim_loss``                      —                                       image reconstruction
``total_variation``                —                                       image regularization
=================================  ======================================  ============================

Regime (see ``benchmarks/README.md``): everything runs as batched float tensors on the selected
device, so every column is the same regime. Classification losses score ``(B, 21, H, W)`` logits
against ``(B, H, W)`` labels; the binary focal loss scores ``(B, 1, H, W)`` logits against
``{0, 1}`` targets with ``alpha=0.25, gamma=2`` and mean reduction in both libraries. ``ssim_loss``
uses an 11x11 window; ``total_variation`` sums over each image and the batch is summed before the
backward. ``--compile`` compiles the loss function; the backward then runs through AOTAutograd,
and both are compiled during the untimed warmup. Other libraries' losses are thin wrappers over the
same PyTorch ops and are left out rather than timed as look-alikes. Throughput is img/s.

Usage:
    python benchmarks/losses/flagship.py --batches 1,8,32 --size 256 --device cpu
    python benchmarks/losses/flagship.py --device cuda --compile --json losses_cuda.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import ModuleType
from typing import Callable, Optional

import torch

# Prefer this checkout to an installed wheel or another editable checkout.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import (
    Backend,
    KorniaRows,
    add_flagship_args,
    batch_list,
    finish_run,
    image_batch,
    image_row_fields,
    optional_import,
    run_batch_sweep,
    setup_run,
    start_run,
)

import kornia.losses as KL

OPS = ("binary_focal_loss_with_logits", "focal_loss", "dice_loss", "ssim_loss", "total_variation")
NUM_CLASSES, ALPHA, GAMMA, WINDOW = 21, 0.25, 2.0, 11


def build_ops(
    b: int,
    h: int,
    w: int,
    device: torch.device,
    dtype: torch.dtype,
    do_compile: bool,
    tvops: Optional[ModuleType],
    skip_compile: frozenset[str] = frozenset(),
    selected: Optional[frozenset[str]] = None,
) -> tuple[dict[str, dict[str, Backend]], dict[str, str]]:
    """Build {op: {backend: zero-arg callable}}; each callable runs one forward + backward on the batch."""
    gen = torch.Generator().manual_seed(0)
    _, batch_f = image_batch(b, h, w, device, dtype)
    logits = torch.randn(b, NUM_CLASSES, h, w, generator=gen).to(device=device, dtype=dtype).requires_grad_()
    labels = torch.randint(0, NUM_CLASSES, (b, h, w), generator=gen).to(device)
    bin_logits = torch.randn(b, 1, h, w, generator=gen).to(device=device, dtype=dtype).requires_grad_()
    bin_target = (torch.rand(b, 1, h, w, generator=gen) < 0.1).to(device=device, dtype=dtype)
    recon = (batch_f + 0.05 * torch.randn(batch_f.shape, generator=gen).to(device=device, dtype=dtype)).clamp(0, 1)
    recon = recon.detach().requires_grad_()

    def backward_to(pred: torch.Tensor) -> Callable[[Callable[[], object]], Callable[[], object]]:
        return lambda loss: lambda: torch.autograd.grad(loss(), pred)

    ops: dict[str, dict[str, Backend]] = {}
    failures: dict[str, str] = {}

    def include(name: str) -> bool:
        return selected is None or name in selected

    def kornia_row(label: str, pred: torch.Tensor, target: Callable[..., object], *args: object) -> dict[str, Backend]:
        rows = KorniaRows(device, do_compile, skip_compile, step=backward_to(pred))
        row = rows(label, target, *args)
        failures.update(rows.compile_failures)
        return row

    if include("binary_focal_loss_with_logits"):
        row = kornia_row(
            "binary_focal_loss_with_logits",
            bin_logits,
            KL.binary_focal_loss_with_logits,
            bin_logits,
            bin_target,
            ALPHA,
            GAMMA,
            "mean",
        )
        if tvops:
            row["torchvision"] = backward_to(bin_logits)(
                lambda: tvops.sigmoid_focal_loss(bin_logits, bin_target, alpha=ALPHA, gamma=GAMMA, reduction="mean")
            )
        ops["binary_focal_loss_with_logits"] = row

    if include("focal_loss"):
        ops["focal_loss"] = kornia_row("focal_loss", logits, KL.focal_loss, logits, labels, ALPHA, GAMMA, "mean")

    if include("dice_loss"):
        ops["dice_loss"] = kornia_row("dice_loss", logits, KL.dice_loss, logits, labels)

    if include("ssim_loss"):
        ops["ssim_loss"] = kornia_row("ssim_loss", recon, KL.ssim_loss, recon, batch_f, WINDOW)

    if include("total_variation"):

        def tv_sum(img: torch.Tensor) -> torch.Tensor:
            return KL.total_variation(img).sum()

        ops["total_variation"] = kornia_row("total_variation", recon, tv_sum, recon)

    return ops, failures


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_flagship_args(parser, ops=OPS)
    args = parser.parse_args()
    device, dtype, sync = setup_run(args, opencv=False)

    tvops, tvops_error = optional_import("torchvision.ops")

    meta = start_run(
        "flagship losses",
        args,
        device,
        units="img/s",
        regimes=["all backends: batched float tensors on the device; timed region = forward + backward"],
        missing=[("torchvision", tvops_error)] if tvops is None else [],
    )
    backends = ["kornia (eager)", "kornia (compiled)", "torchvision"]
    results = run_batch_sweep(
        batch_list(args),
        lambda b: build_ops(
            b, args.size, args.size, device, dtype, args.compile, tvops, args.skip_compile_ops, args.ops
        ),
        backends,
        row_fields=image_row_fields(args),
        sync=sync,
        torch_backends=("kornia (", "torchvision"),
        units="img/s",
        min_run_time=args.min_run_time,
    )
    finish_run(args, "losses", meta, results)


if __name__ == "__main__":
    main()
