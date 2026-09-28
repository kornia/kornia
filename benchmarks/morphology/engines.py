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

For each dtype, kernel size and batch it times ``dilation`` with each explicit public engine,
reports the fastest successful timing, and checks that the engines agree. ``conv-unfold`` and
``shift-unfold`` are the largest absolute differences from the ``unfold`` output. The engines
compute the same max-plus expression, so a non-zero value is backend rounding: ``convolution``
inherits the precision of ``conv2d``, and ``shift`` takes the same max or min sequentially in the
same order, which should always give 0. On CUDA the TF32 column repeats the ``convolution`` comparison
with ``torch.backends.cudnn.allow_tf32`` (PyTorch's default ``True``) switched on, which rounds
float32 convolution inputs to a 10-bit mantissa on Ampere and newer GPUs; every other column runs
with the flag left as the process found it.

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
import math
import sys
import time
from pathlib import Path
from typing import Any, Callable, Optional

import torch

# Prefer this checkout to an installed wheel or another editable checkout.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import add_contribute_args, finish_run, parse_names, start_run, time_us_or_error, warm_up_cpu

import kornia.morphology as KM

ENGINES = ("unfold", "convolution", "shift")
SHORT = {"unfold": "unfold", "convolution": "conv", "shift": "shift"}


def parse_selected_names(value: str, *, kind: str, choices: tuple[str, ...]) -> frozenset[str]:
    """Parse a non-empty comma-separated subset and make invalid names an argparse error."""
    selected = parse_names(value)
    unknown = selected.difference(choices)
    if unknown:
        raise argparse.ArgumentTypeError(
            f"unknown {kind}(s): {', '.join(sorted(unknown))}. Available: {', '.join(choices)}"
        )
    if not selected:
        raise argparse.ArgumentTypeError(f"at least one {kind} is required. Available: {', '.join(choices)}")
    return selected


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


def display_timing(value: float) -> str:
    """Render a failed timing as a visible table marker instead of a plausible number."""
    return f"{value:.3f}" if math.isfinite(value) else "-"


def display_difference(value: float) -> str:
    """Render a failed numerical comparison as visibly unavailable."""
    return f"{value:.2e}" if math.isfinite(value) else "-"


def abort_if_cuda_context_poisoned(
    device: torch.device, *, context: str, error_text: str = "", cause: Optional[BaseException] = None
) -> None:
    """Abort rather than publish later timings after an illegal CUDA memory access."""
    if device.type != "cuda":
        return
    try:
        torch.cuda.synchronize(device)
    except Exception as sync_error:
        error_text = f"{error_text} | {sync_error}"
    if "illegal memory access" in error_text.lower():
        raise SystemExit(
            f"FATAL: CUDA context poisoned during {context} (illegal memory access); "
            "no later measurement or contributed result would be trustworthy. "
            "Rerun without the failing configuration; CUDA_LAUNCH_BLOCKING=1 localizes the kernel."
        ) from cause


def time_us_checked(
    fn: Callable[[], None], min_run_time: float, sync: Optional[Callable[[], None]], device: torch.device, context: str
) -> tuple[float, float, Optional[str]]:
    """Time a call and ensure a reported CUDA failure did not poison the context."""
    median, iqr, error = time_us_or_error(fn, min_run_time, sync)
    if error is not None:
        abort_if_cuda_context_poisoned(device, context=context)
    return median, iqr, error


def result_row(
    *, dtype: str, batch: int, kernel: int, backend: str, median_us: float, iqr_us: float, error: Optional[str]
) -> dict[str, Any]:
    """Return one schema-valid long-form timing row."""
    valid = error is None and math.isfinite(median_us) and median_us > 0
    row: dict[str, Any] = {
        "op": "dilation",
        "backend": backend,
        "batch": batch,
        "dtype": dtype,
        "kernel": kernel,
        "median_us": median_us if valid else None,
        "iqr_us": iqr_us if valid and math.isfinite(iqr_us) else None,
        "throughput_per_s": batch * 1e6 / median_us if valid else None,
    }
    if not valid:
        row["error"] = error or "non-finite timing"
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", type=str, default="cpu", help="cpu, cuda, or mps")
    parser.add_argument(
        "--engines",
        type=lambda value: parse_selected_names(value, kind="engine", choices=ENGINES),
        default=frozenset(ENGINES),
        help="comma-separated engines (default: all)",
    )
    dtype_names = ("float32", "float16", "bfloat16", "float64")
    parser.add_argument(
        "--dtypes",
        type=lambda value: parse_selected_names(value, kind="dtype", choices=dtype_names),
        default=frozenset({"float32", "float16"}),
        help="comma-separated dtypes (default: float32,float16)",
    )
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
    dtypes = [d for d in dtype_names if d in args.dtypes]
    args.dtype = ",".join(dtypes)  # start_run prints it
    meta = start_run(
        "morphology engines",
        args,
        device,
        units="img/s",
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
    header += f" {'fastest':>12}" + "".join(f" {SHORT[e] + '-unfold':>13}" for e in diffs)
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
                row: dict[str, Any] = {
                    "dtype": dtype_name,
                    "batch": b,
                    "kernel": k,
                }
                result_by_engine: dict[str, dict[str, Any]] = {}
                for engine in engines:
                    median, iqr, error = time_us_checked(
                        bench_fn(x, kernel, engine, args.backward),
                        args.min_run_time,
                        sync,
                        device,
                        f"eager {engine} timing (dtype={dtype_name}, batch={b}, kernel={k})",
                    )
                    row[f"{engine}_ms"] = median / 1e3
                    row[f"{engine}_iqr_ms"] = iqr / 1e3
                    row[f"{engine}_error"] = error
                    engine_result = result_row(
                        dtype=dtype_name,
                        batch=b,
                        kernel=k,
                        backend=f"kornia (engine={engine})",
                        median_us=median,
                        iqr_us=iqr,
                        error=error,
                    )
                    results.append(engine_result)
                    result_by_engine[engine] = engine_result
                successful = [e for e in engines if row[f"{e}_error"] is None and math.isfinite(row[f"{e}_ms"])]
                row["fastest"] = min(successful, key=lambda e: row[f"{e}_ms"]) if successful else "-"
                if args.compile:
                    for engine in engines:
                        try:
                            fn = bench_fn(x, kernel, engine, args.backward, compiled=True)
                            start = time.perf_counter()
                            fn()  # compile outside the timed region
                            if sync is not None:
                                sync()
                            elif device.type == "cuda":
                                torch.cuda.synchronize(device)
                            row[f"compile_{engine}_s"] = time.perf_counter() - start
                            median, iqr, error = time_us_checked(
                                fn,
                                args.min_run_time,
                                sync,
                                device,
                                f"compiled {engine} timing (dtype={dtype_name}, batch={b}, kernel={k})",
                            )
                        except Exception as exc:
                            abort_if_cuda_context_poisoned(
                                device,
                                context=f"compiled {engine} warmup (dtype={dtype_name}, batch={b}, kernel={k})",
                                error_text=str(exc),
                                cause=exc,
                            )
                            row[f"compile_{engine}_s"] = float("nan")
                            median, iqr, error = float("nan"), float("nan"), type(exc).__name__
                        row[f"compiled_{engine}_ms"] = median / 1e3
                        row[f"compiled_{engine}_iqr_ms"] = iqr / 1e3
                        row[f"compiled_{engine}_error"] = error
                        compiled_result = result_row(
                            dtype=dtype_name,
                            batch=b,
                            kernel=k,
                            backend=f"kornia (engine={engine}, compiled)",
                            median_us=median,
                            iqr_us=iqr,
                            error=error,
                        )
                        compiled_result["compile_s"] = (
                            row[f"compile_{engine}_s"] if math.isfinite(row[f"compile_{engine}_s"]) else None
                        )
                        results.append(compiled_result)
                for engine in diffs:
                    try:
                        difference = max_abs_diff(x, kernel, engine, None)
                    except Exception as exc:
                        abort_if_cuda_context_poisoned(
                            device,
                            context=f"{engine}-unfold comparison (dtype={dtype_name}, batch={b}, kernel={k})",
                            error_text=str(exc),
                            cause=exc,
                        )
                        difference = float("nan")
                        result_by_engine[engine]["comparison_error"] = type(exc).__name__
                    row[f"max_abs_diff_{engine}"] = difference
                    result_by_engine[engine]["max_abs_diff_from_unfold"] = (
                        difference if math.isfinite(difference) else None
                    )
                if tf32_col:
                    try:
                        tf32_difference = max_abs_diff(x, kernel, "convolution", True)
                    except Exception as exc:
                        abort_if_cuda_context_poisoned(
                            device,
                            context=f"convolution TF32 comparison (dtype={dtype_name}, batch={b}, kernel={k})",
                            error_text=str(exc),
                            cause=exc,
                        )
                        tf32_difference = float("nan")
                        result_by_engine["convolution"]["tf32_comparison_error"] = type(exc).__name__
                    row["max_abs_diff_convolution_tf32"] = tf32_difference
                    result_by_engine["convolution"]["max_abs_diff_from_unfold_tf32"] = (
                        tf32_difference if math.isfinite(tf32_difference) else None
                    )
                line = f"{dtype_name:9} {b:>3} {k:>3}" + "".join(
                    f" {display_timing(row[e + '_ms']):>11}" for e in engines
                )
                line += f" {row['fastest']:>12}"
                line += "".join(f" {display_difference(row['max_abs_diff_' + e]):>13}" for e in diffs)
                if tf32_col:
                    line += f" {display_difference(row['max_abs_diff_convolution_tf32']):>13}"
                if args.compile:
                    line += "".join(
                        f" {display_timing(row['compiled_' + e + '_ms']):>18} "
                        f"{display_timing(row['compile_' + e + '_s']):>9}"
                        for e in engines
                    )
                print(line, flush=True)
    finish_run(args, "morphology-engines", meta, results)


if __name__ == "__main__":
    main()
