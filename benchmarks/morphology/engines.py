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

"""Compare the three ``kornia.morphology`` engines, ``unfold``, ``convolution`` and ``shift``, on one device.

The ``engine="auto"`` default picks one engine per device (see ``kornia.morphology.dilation``).
This script measures the evidence behind that choice: for each dtype, kernel size and batch it
times ``dilation`` with each engine, reports the fastest one and the one ``auto`` selects, and
checks that the engines agree. ``conv-unfold`` and ``shift-unfold`` are the largest absolute
differences from the ``unfold`` output. The engines compute the same max-plus expression, so a
non-zero value is backend rounding: ``convolution`` inherits the precision of ``conv2d``, and
``shift`` takes the same max or min in a different order, which should always give 0. On CUDA the
TF32 column repeats the ``convolution`` comparison with ``torch.backends.cudnn.allow_tf32``
(PyTorch's default ``True``) switched on, which rounds float32 convolution inputs to a 10-bit
mantissa on Ampere and newer GPUs; every other column runs with the flag left as the process
found it.

``--backward`` times forward + backward of ``dilation(x).sum()`` instead of the forward alone.
``--compile`` adds a ``torch.compile(fullgraph=True)`` timing of each engine and the wall clock of
its first (compiling) call: inductor fuses the ``unfold`` window reduction and the ``shift`` loop,
so the ranking under compile can differ from the eager one, and ``shift``'s compile time grows
with the kernel area. ``--engines`` restricts the sweep, e.g. to skip ``convolution`` in CPU half
precision, which runs for minutes at large kernels.

Usage:
    python benchmarks/morphology/engines.py --device cpu
    python benchmarks/morphology/engines.py --device cuda --dtypes float32,float16,bfloat16 --json engines_cuda.json
    python benchmarks/morphology/engines.py --device mps --kernels 3,7 --batches 1,32 --size 512
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Callable, Optional

import torch

# Prefer this checkout to an installed wheel or another editable checkout.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import add_contribute_args, finish_run, parse_names, start_run, time_us, warm_up_cpu

import kornia.morphology as KM
from kornia.morphology.morphology import _resolve_engine

ENGINES = ("unfold", "convolution", "shift")
SHORT = {"unfold": "unfold", "convolution": "conv", "shift": "shift"}


def max_abs_diff(x: torch.Tensor, kernel: torch.Tensor, engine: str, tf32: Optional[bool]) -> float:
    """Largest absolute difference between ``engine`` and ``unfold``, optionally with cuDNN TF32 forced."""
    previous = torch.backends.cudnn.allow_tf32
    if tf32 is not None:
        torch.backends.cudnn.allow_tf32 = tf32
    try:
        out = KM.dilation(x, kernel, engine=engine)
    finally:
        torch.backends.cudnn.allow_tf32 = previous
    ref = KM.dilation(x, kernel, engine="unfold")
    return (out.cpu().double() - ref.cpu().double()).abs().max().item()


def bench_fn(
    x: torch.Tensor, kernel: torch.Tensor, engine: str, backward: bool, compiled: bool = False
) -> Callable[[], object]:
    def op(t: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        return KM.dilation(t, k, engine=engine)

    if compiled:
        # Every config compiles its own graph; without a reset the sweep hits dynamo's recompile limit.
        torch._dynamo.reset()
    fn = torch.compile(op, fullgraph=True) if compiled else op
    if not backward:
        return lambda: fn(x, kernel)
    xg = x.detach().clone().requires_grad_(True)

    def step() -> None:
        xg.grad = None
        fn(xg, kernel).sum().backward()

    return step


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", type=str, default="cpu", help="cpu, cuda, or mps")
    parser.add_argument(
        "--engines", type=parse_names, default=frozenset(ENGINES), help="comma-separated engines (default: all)"
    )
    parser.add_argument("--dtypes", type=parse_names, default=frozenset({"float32", "float16"}))
    parser.add_argument("--kernels", type=str, default="3,5,7,15", help="comma-separated square kernel sizes")
    parser.add_argument("--batches", type=str, default="1,8", help="comma-separated batch sizes")
    parser.add_argument("--channels", type=int, default=3)
    parser.add_argument("--size", type=int, default=256, help="square image side")
    parser.add_argument("--threads", type=int, default=4, help="torch intra-op threads")
    parser.add_argument("--backward", action="store_true", help="time forward + backward")
    parser.add_argument("--compile", action="store_true", help="also time torch.compile(fullgraph=True) engines")
    parser.add_argument("--min-run-time", type=float, default=0.5, help="seconds of repeats per measurement")
    parser.add_argument("--json", type=str, default=None, help="write machine-readable results to this path")
    add_contribute_args(parser)
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    warm_up_cpu()
    torch.manual_seed(0)
    device = torch.device(args.device)
    sync = torch.mps.synchronize if device.type == "mps" else None
    dtypes = [d for d in ("float32", "float16", "bfloat16", "float64") if d in args.dtypes]
    args.dtype = ",".join(dtypes)  # start_run prints it
    meta = start_run(
        "morphology engines",
        args,
        device,
        units="ms",
        regimes=[
            f"dilation, square kernel, B x {args.channels} x {args.size} x {args.size} float input"
            + (", forward + backward" if args.backward else ", forward only"),
        ],
    )
    if device.type == "cuda":
        meta["cudnn_allow_tf32"] = torch.backends.cudnn.allow_tf32
        print(f"# cudnn.allow_tf32={torch.backends.cudnn.allow_tf32}")
    if device.type == "cpu":
        meta["cpu_capability"] = torch.backends.cpu.get_cpu_capability()
        print(f"# cpu capability: {meta['cpu_capability']}")

    engines = [e for e in ENGINES if e in args.engines]
    diffs = [e for e in engines if e != "unfold"] if "unfold" in engines else []
    tf32_col = device.type == "cuda" and "convolution" in diffs
    header = f"{'dtype':9} {'B':>3} {'k':>3}" + "".join(f" {SHORT[e] + ' ms':>11}" for e in engines)
    header += f" {'fastest':>12} {'auto':>12}" + "".join(f" {SHORT[e] + '-unfold':>13}" for e in diffs)
    header += f" {'conv TF32 on':>13}" if tf32_col else ""
    if args.compile:
        header += "".join(f" {'compiled ' + SHORT[e] + ' ms':>18} {'compile s':>9}" for e in engines)
    print(header)
    results: list[dict[str, Any]] = []
    for dtype_name in dtypes:
        dtype = getattr(torch, dtype_name)
        for b in (int(v) for v in args.batches.split(",")):
            for k in (int(v) for v in args.kernels.split(",")):
                x = torch.rand(b, args.channels, args.size, args.size, device=device, dtype=dtype)
                kernel = torch.ones(k, k, device=device, dtype=dtype)
                row: dict[str, Any] = {"dtype": dtype_name, "batch": b, "kernel": k, "auto": _resolve_engine("auto", x)}
                for engine in engines:
                    median, iqr = time_us(bench_fn(x, kernel, engine, args.backward), args.min_run_time, sync)
                    row[f"{engine}_ms"] = median / 1e3
                    row[f"{engine}_iqr_ms"] = iqr / 1e3
                row["fastest"] = min(engines, key=lambda e: row[f"{e}_ms"])
                if args.compile:
                    for engine in engines:
                        fn = bench_fn(x, kernel, engine, args.backward, compiled=True)
                        start = time.perf_counter()
                        fn()  # compile outside the timed region
                        if sync is not None:
                            sync()
                        elif device.type == "cuda":
                            torch.cuda.synchronize(device)
                        row[f"compile_{engine}_s"] = time.perf_counter() - start
                        median, iqr = time_us(fn, args.min_run_time, sync)
                        row[f"compiled_{engine}_ms"] = median / 1e3
                        row[f"compiled_{engine}_iqr_ms"] = iqr / 1e3
                for engine in diffs:
                    row[f"max_abs_diff_{engine}"] = max_abs_diff(x, kernel, engine, None)
                if tf32_col:
                    row["max_abs_diff_convolution_tf32"] = max_abs_diff(x, kernel, "convolution", True)
                line = f"{dtype_name:9} {b:>3} {k:>3}" + "".join(f" {row[e + '_ms']:>11.3f}" for e in engines)
                line += f" {row['fastest']:>12} {row['auto']:>12}"
                line += "".join(f" {row['max_abs_diff_' + e]:>13.2e}" for e in diffs)
                if tf32_col:
                    line += f" {row['max_abs_diff_convolution_tf32']:>13.2e}"
                if args.compile:
                    line += "".join(
                        f" {row['compiled_' + e + '_ms']:>18.3f} {row['compile_' + e + '_s']:>9.1f}" for e in engines
                    )
                print(line, flush=True)
                results.append(row)
    finish_run(args, "morphology-engines", meta, results)


if __name__ == "__main__":
    main()
