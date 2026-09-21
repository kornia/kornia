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

"""Shared methodology utilities for kornia benchmarks.

Non-negotiable methodology (W3 of the agent-era plan): warmup, device sync inside timed
regions, multiple repeats with median + spread, pinned seeds, and recorded hardware/software
metadata. ``torch.utils.benchmark.Timer.blocked_autorange`` supplies warmup, repeats and CUDA
sync; this module adds spread reporting, MPS sync support, metadata capture, and
machine-readable JSON export so every run is comparable and citable.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import platform
import random
import re
import subprocess
import sys
import time
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Optional

import torch
import torch.utils.benchmark as bench


def warm_up_cpu(seconds: float = 3.0) -> None:
    """Keep PyTorch's intra-op thread pool under sustained load before any timing.

    Hybrid CPUs schedule lightly loaded threads on efficiency cores and move them to performance
    cores only after sustained load; under WSL2 the guest cannot pin them. On an i7-14700K a 5x5
    oneDNN convolution measured 0.56 ms before and 0.22 ms after this warm-up, while a lighter
    implementation of the same filter was barely affected, so an unwarmed run can reverse A/B
    conclusions. ``blocked_autorange``'s own short warmup does not reach that state.
    """
    a = torch.rand(1024, 1024)
    end = time.perf_counter() + seconds
    while time.perf_counter() < end:
        a @ a


def time_us(
    fn: Callable[[], object],
    min_run_time: float = 1.0,
    sync: Optional[Callable[[], None]] = None,
    num_threads: Optional[int] = None,
) -> tuple[float, float]:
    """Median and interquartile-range wall clock of ``fn`` in microseconds.

    ``blocked_autorange`` warms up, runs many repeats, and synchronizes CUDA. Devices it does
    not sync (MPS) pass their sync as ``sync`` so it lands inside the timed region. Returns
    ``(nan, nan)`` if ``fn`` raises, so callers can render a skip cell instead of dying.
    ``num_threads`` controls the timed calls and defaults to the current ``torch.get_num_threads()``;
    Timer would otherwise override the caller's thread count with one.
    """
    stmt = "fn(); sync()" if sync is not None else "fn()"
    try:
        threads = torch.get_num_threads() if num_threads is None else num_threads
        m = bench.Timer(stmt=stmt, globals={"fn": fn, "sync": sync}, num_threads=threads).blocked_autorange(
            min_run_time=min_run_time
        )
        return m.median * 1e6, m.iqr * 1e6
    except Exception:
        return float("nan"), float("nan")


def git_commit() -> str:
    """Short hash of HEAD, ``-dirty`` when tracked files differ from it, or 'unknown' outside git.

    A run measured from a modified checkout is not reproducible from the hash alone. The LAF-ops
    result files were first committed with the hash of a commit that predated the harness edits
    they were measured with, so the recorded commit named a tree whose LAF generator differed
    from the one that actually ran. Untracked files are ignored: ``--contribute`` writes an
    untracked result file into the checkout, so counting those would mark every contributed run
    dirty and train readers to ignore the marker.
    """
    try:
        head = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()  # noqa: S607
    except Exception:
        return "unknown"
    try:
        modified = subprocess.check_output(["git", "status", "--porcelain", "-uno"], text=True).strip()  # noqa: S607
    except Exception:
        return head
    return f"{head}-dirty" if modified else head


def _optional_version(module: str) -> Optional[str]:
    try:
        return __import__(module).__version__
    except Exception:
        return None


def run_metadata(device: torch.device) -> dict[str, Any]:
    """Hardware/software metadata embedded in every result file (W3: date, hardware, versions)."""
    import kornia

    meta: dict[str, Any] = {
        "timestamp_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "kornia": kornia.__version__,
        "device": str(device),
        "torch_num_threads": torch.get_num_threads(),
        "opencv": _optional_version("cv2"),
        "torchvision": _optional_version("torchvision"),
        "numpy": _optional_version("numpy"),
        "albumentations": _optional_version("albumentations"),
        "kornia_rs": _optional_version("kornia_rs"),
        "pillow": _optional_version("PIL"),
        "skimage": _optional_version("skimage"),
    }
    if device.type == "cuda":
        meta["cuda_device"] = torch.cuda.get_device_name(device)
        meta["cuda_version"] = torch.version.cuda
    return meta


def _sanitize(obj: Any) -> Any:
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    return obj


def save_json(path: str | Path, metadata: dict[str, Any], results: list[dict[str, Any]]) -> Path:
    """Write one run as strict-valid JSON ``{"metadata": ..., "results": [...]}`` (non-finite → null).

    Keys are sorted so committed result files satisfy the ``pretty-format-json`` pre-commit hook.
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = _sanitize({"metadata": metadata, "results": results})
    out.write_text(json.dumps(payload, indent=2, allow_nan=False, sort_keys=True) + "\n")
    return out


def versions_line(meta: dict[str, Any]) -> str:
    """One-line software-stack summary for printed table headers (the JSON carries the same data)."""
    keys = ("torch", "kornia", "python", "opencv", "torchvision", "albumentations", "pillow", "kornia_rs", "skimage")
    return "# " + ", ".join(f"{k} {meta.get(k) or '-'}" for k in keys)


def collect_load_metrics() -> dict[str, Any]:
    """Aggregate system-load snapshot for run metadata.

    Privacy-preserving by design: numbers only (load averages, memory totals, CPU count) —
    never process or application names.
    """
    metrics: dict[str, Any] = {
        "load_avg_1m": None,
        "load_avg_5m": None,
        "load_avg_15m": None,
        "cpu_count": os.cpu_count(),
        "mem_total_bytes": None,
        "mem_available_bytes": None,
    }
    try:
        one, five, fifteen = os.getloadavg()
        metrics.update(load_avg_1m=one, load_avg_5m=five, load_avg_15m=fifteen)
    except (OSError, AttributeError):
        pass
    try:
        import psutil  # optional; aggregate numbers only

        vm = psutil.virtual_memory()
        metrics.update(mem_total_bytes=int(vm.total), mem_available_bytes=int(vm.available))
    except Exception:  # noqa: S110
        pass
    return metrics


def machine_slug(meta: dict[str, Any], override: Optional[str] = None) -> str:
    """Stable, human-readable machine identifier for result filenames."""
    if override:
        return re.sub(r"-+", "-", re.sub(r"[^a-z0-9]+", "-", override.lower())).strip("-")
    name = meta.get("cuda_device")
    if not name:
        if sys.platform == "darwin":
            try:
                name = subprocess.check_output(
                    ["sysctl", "-n", "machdep.cpu.brand_string"],  # noqa: S607
                    text=True,
                ).strip()
            except Exception:
                name = None
        elif sys.platform.startswith("linux"):
            try:
                for line in Path("/proc/cpuinfo").read_text().splitlines():
                    if line.lower().startswith("model name"):
                        name = line.split(":", 1)[1].strip()
                        break
            except Exception:
                name = None
    if not name:
        name = str(meta.get("machine", "unknown"))
    return re.sub(r"-+", "-", re.sub(r"[^a-z0-9]+", "-", name.lower())).strip("-")


def canonical_result_name(meta: dict[str, Any], suite: str, slug_override: Optional[str] = None) -> str:
    """Filename for a contributed run: <suite>--<machine-slug>--<device-type>.json."""
    device_type = str(meta["device"]).split(":")[0]
    return f"{suite}--{machine_slug(meta, slug_override)}--{device_type}.json"


def add_contribute_args(parser: argparse.ArgumentParser) -> None:
    """CLI options shared by every flagship suite for contributing canonical result files."""
    parser.add_argument(
        "--contribute",
        type=str,
        default=None,
        help="write this run to DIR/<kornia-version>/<suite>--<machine>--<device>.json for committing",
    )
    parser.add_argument(
        "--machine-slug", type=str, default=None, help="override the auto-detected machine name in the filename"
    )


def print_preflight(metrics: dict[str, Any]) -> None:
    """Measurement-hygiene notice. Advisory only; records nothing beyond aggregate numbers."""
    print("# preflight: close other applications, use mains power, let the machine cool before contributing.")
    load1, ncpu = metrics.get("load_avg_1m"), metrics.get("cpu_count")
    if load1 is not None and ncpu and load1 > ncpu:
        print(f"# preflight WARNING: load average {load1:.1f} exceeds {ncpu} CPUs - numbers will be noisy.")
    total, avail = metrics.get("mem_total_bytes"), metrics.get("mem_available_bytes")
    if total and avail is not None and avail < 0.1 * total:
        print("# preflight WARNING: less than 10% memory available - numbers will be noisy.")


def contribute_result(
    results_dir: str | Path,
    suite: str,
    metadata: dict[str, Any],
    results: list[dict[str, Any]],
    slug_override: Optional[str] = None,
) -> Path:
    """Write one run under the canonical results layout and print the git line to commit it."""
    version = str(metadata.get("kornia", "unknown"))
    out = Path(results_dir) / version / canonical_result_name(metadata, suite, slug_override)
    save_json(out, metadata, results)
    if str(metadata.get("git_commit", "")).endswith("-dirty"):
        print("# WARNING: measured from a modified checkout - the recorded commit does not describe this tree.")
    print(f"# contributed: {out}")
    print(f"# commit it with: git add {out}")
    return out


def run_batch_sweep(
    batches: Sequence[Any],
    build_ops: Callable[[Any], tuple[dict[str, dict[str, Optional[Callable[[], object]]]], dict[str, str]]],
    backends: list[str],
    row_fields: Callable[[Any], dict[str, Any]],
    sync: Optional[Callable[[], None]] = None,
    torch_backends: tuple[str, ...] = ("kornia (", "torchvision"),
    label_fn: Optional[Callable[[Any], str]] = None,
    items_fn: Optional[Callable[[Any], float]] = None,
    units: str = "",
    label_width: int = 26,
    col_width: int = 17,
    min_run_time: float = 1.0,
) -> list[dict[str, Any]]:
    """Sweep configurations, print one throughput table per config, and return JSON-ready rows.

    ``build_ops(config)`` returns ``({op: {backend: zero-arg callable | None}}, {op: exc_name})``;
    the second dict names ops whose ``torch.compile`` warmup failed, reported as a NOTE and as a
    result row whose timings are ``null`` and whose ``error`` field names the exception, so a
    failure is visible in the exported JSON rather than only on the console. ``sync`` lands
    inside the timed region only for backends whose name starts with one of ``torch_backends``
    — uint8 CPU-loop baselines are timed without it (the default prefix is ``"kornia ("`` so
    the CPU-only ``"kornia-rs"`` backend never matches).

    A config is a batch size by default: it labels the row ``batch=<b>``, is written to the row's
    ``"batch"`` field, and is the per-call item count the throughput divides by. A backend that
    raises while timed prints ``✗`` and gets a row with ``null`` timings and its exception name in
    ``error``. A suite whose
    config is not a single batch size (a ``(batch, n)`` pair, say) passes ``label_fn`` for the
    printed label, ``items_fn`` for the item count, and a ``"batch"`` key in ``row_fields`` — the
    committed-result schema requires that field to stay an ``int``. ``units`` names what the
    throughput counts in the printed header; the JSON never abbreviates it away.
    """
    label_of = label_fn if label_fn is not None else (lambda b: f"batch={b}")
    items_of = items_fn if items_fn is not None else (lambda b: b)
    results: list[dict[str, Any]] = []
    failed: list[str] = []
    header = ""
    for b in batches:
        ops, compile_failures = build_ops(b)
        if compile_failures:
            exc_names = sorted(set(compile_failures.values()))
            print(f"# NOTE: torch.compile warmup failed ({', '.join(exc_names)}) for: {', '.join(compile_failures)}")
        width = max([label_width, len(label_of(b)) + 1] + [len(op) + 1 for op in ops])
        header = f"{label_of(b):<{width}}" + "".join(f"{n[:col_width]:>{col_width + 1}}" for n in backends)
        print("-" * len(header))
        print(header + (f"   ({units})" if units else ""))
        print("-" * len(header))
        items = items_of(b)
        for op_name, row in ops.items():
            cells = []
            for backend in backends:
                fn = row.get(backend)
                if fn is None:
                    # Record the gap instead of dropping the row: a reader of the JSON otherwise
                    # cannot tell "this backend was never requested" from "it failed", and the
                    # exception type would survive only in the console NOTE above.
                    reason = compile_failures.get(op_name) if "compiled" in backend else None
                    results.append(
                        {
                            "op": op_name,
                            "backend": backend,
                            "batch": b,
                            **row_fields(b),
                            "median_us": None,
                            "iqr_us": None,
                            "throughput_per_s": None,
                            "error": reason or "unavailable",
                        }
                    )
                    cells.append(f"{'-':>{col_width + 1}}")
                    continue
                backend_sync = sync if backend.startswith(torch_backends) else None
                median, iqr = time_us(fn, min_run_time=min_run_time, sync=backend_sync)
                row_out: dict[str, Any] = {"op": op_name, "backend": backend, "batch": b, **row_fields(b)}
                if math.isnan(median):
                    # time_us swallows the exception; one untimed call recovers its name so the JSON
                    # says why the cell is empty, like a compile failure does.
                    results.append(
                        {
                            **row_out,
                            "median_us": None,
                            "iqr_us": None,
                            "throughput_per_s": None,
                            "error": _failure_name(fn),
                        }
                    )
                    failed.append(f"{op_name}/{backend}")
                    cells.append(f"{'✗':>{col_width + 1}}")
                    continue
                thr = items / (median * 1e-6)
                results.append({**row_out, "median_us": median, "iqr_us": iqr, "throughput_per_s": thr})
                cells.append(f"{thr:>{col_width + 1}.0f}")
            print(f"{op_name:<{width}}" + "".join(cells))
    if header:
        print("-" * len(header))
    if failed:
        names = ", ".join(dict.fromkeys(failed))
        print(f"# NOTE: '✗' = the call raised (exception name in the JSON 'error' field): {names}")
    return results


def _failure_name(fn: Callable[[], object]) -> str:
    try:
        fn()
    except Exception as exc:
        return type(exc).__name__
    return "timing failed"


Backend = Optional[Callable[[], object]]

#: The checkout this file lives in. Suites put it at ``sys.path[0]`` before importing kornia so
#: the measured kornia is this tree, not a wheel or another editable checkout.
REPO_ROOT = Path(__file__).resolve().parents[1]


def optional_import(name: str) -> tuple[Optional[ModuleType], Optional[str]]:
    """Import an optional baseline library, returning ``(module, None)`` or ``(None, reason)``."""
    try:
        return importlib.import_module(name), None
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


def parse_names(value: str) -> frozenset[str]:
    """Comma-separated names (``--ops``, ``--skip-compile-ops``) as a set; empty items are dropped."""
    return frozenset(s.strip() for s in value.split(",") if s.strip())


def add_flagship_args(
    parser: argparse.ArgumentParser, *, ops: Sequence[str] = (), batches: str = "1,8,32", size: int = 256
) -> None:
    """The command-line options every suite shares, so every suite is driven the same way.

    ``ops`` lists the suite's row names; ``--ops`` rejects any other name instead of silently
    running nothing. Suites whose configuration is not a batch sweep (``feature/laf_ops.py``) pass
    ``batches=""`` and add their own sweep option; ``--batches`` is then not registered.
    """

    def parse_ops(value: str) -> Optional[frozenset[str]]:
        selected = parse_names(value)
        unknown = selected.difference(ops)
        if ops and unknown:
            raise argparse.ArgumentTypeError(
                f"unknown operation(s): {', '.join(sorted(unknown))}. Available: {', '.join(ops)}"
            )
        return selected or None

    if batches:
        parser.add_argument("--batches", type=str, default=batches, help="comma-separated batch sizes to sweep")
    parser.add_argument("--size", type=int, default=size, help="square image side")
    parser.add_argument("--device", type=str, default="cpu", help="cpu, cuda, or mps")
    parser.add_argument("--dtype", type=str, default="float32", choices=["float32", "float16", "bfloat16", "float64"])
    parser.add_argument("--threads", type=int, default=4, help="torch intra-op threads")
    parser.add_argument("--compile", action="store_true", help="also time torch.compile'd kornia")
    parser.add_argument("--ops", type=parse_ops, default=None, help="comma-separated op rows to run (default: all)")
    parser.add_argument(
        "--skip-compile-ops",
        type=parse_names,
        default=frozenset(),
        help="comma-separated op names to keep eager-only (workaround for faulting compiled kernels)",
    )
    parser.add_argument("--min-run-time", type=float, default=1.0, help="seconds of repeats per measurement")
    parser.add_argument("--json", type=str, default=None, help="write machine-readable results to this path")
    add_contribute_args(parser)


def setup_run(args: argparse.Namespace) -> tuple[torch.device, torch.dtype, Optional[Callable[[], None]]]:
    """Thread count, CPU warm-up and pinned seeds; returns ``(device, dtype, sync)``.

    ``sync`` is the MPS synchronize for ``time_us`` (``blocked_autorange`` already syncs CUDA). The
    CPU warm-up runs on accelerator runs too, because they still time CPU-only baselines.
    """
    torch.set_num_threads(args.threads)
    warm_up_cpu()
    torch.manual_seed(0)
    random.seed(0)
    try:
        import numpy as np

        np.random.seed(0)  # noqa: NPY002 - albumentations samples from the legacy global RNG
    except ImportError:
        pass
    device = torch.device(args.device)
    return device, getattr(torch, args.dtype), (torch.mps.synchronize if device.type == "mps" else None)


def kornia_provenance(module_file: Optional[str]) -> tuple[str, str]:
    """``(console path, exported field)`` for the imported kornia package.

    The console gets the resolved absolute path, which answers "which tree did I measure?". The
    exported ``kornia_module`` field is checkout-relative (or ``"outside-checkout"``): a contributed
    file is public, and the privacy rule keeps home directories out of it.
    """
    if module_file is None:
        return "unknown", "outside-checkout"
    resolved = Path(module_file).resolve()
    if REPO_ROOT in resolved.parents:
        return str(resolved), resolved.relative_to(REPO_ROOT).as_posix()
    return str(resolved), "outside-checkout"


def start_run(
    title: str,
    args: argparse.Namespace,
    device: torch.device,
    *,
    units: str,
    regimes: Sequence[str] = (),
    missing: Sequence[tuple[str, Optional[str]]] = (),
) -> dict[str, Any]:
    """Print the shared run header and return the metadata to export.

    Every suite prints the same header in the same order: title, commit and platform; the
    software stack; the kornia source; the CUDA device; the run configuration with the throughput
    unit; the regime of each backend; and one ``NOTE`` per unavailable library or eager-only op.
    ``missing`` is ``(library, reason)`` for each baseline that could not be imported.
    """
    import kornia

    meta = run_metadata(device)
    meta["load"] = collect_load_metrics()
    console_path, module_field = kornia_provenance(kornia.__file__)
    meta["kornia_module"] = module_field
    # The docs page and the llms digest label the throughput column from this key.
    meta["units"] = units
    if args.contribute:
        print_preflight(meta["load"])
    print(f"# {title} benchmark — commit {meta['git_commit']} — {platform.platform()}")
    print(versions_line(meta))
    print(f"# kornia source: {console_path}")
    if module_field == "outside-checkout":
        # git_commit() reports this checkout's HEAD, so a kornia from anywhere else means the
        # numbers and the commit label describe different code.
        print(f"# WARNING: kornia resolved outside {REPO_ROOT} - these numbers are NOT this checkout's.")
    if device.type == "cuda":
        print(f"# CUDA device: {meta['cuda_device']} (CUDA {meta['cuda_version']})")
    size = f", size={args.size}" if getattr(args, "size", None) else ""
    print(f"# device={device}, dtype={args.dtype}, threads={torch.get_num_threads()}{size} — throughput {units}")
    for line in regimes:
        print(f"# {line}")
    print("# '-' = skipped: backend unavailable, no counterpart, or compile failure (see the JSON 'error' field)")
    for name, reason in missing:
        print(f"# NOTE: {name} not available ({reason}) — its column is skipped")
    if getattr(args, "skip_compile_ops", None):
        print(f"# NOTE: --skip-compile-ops keeps eager-only: {', '.join(sorted(args.skip_compile_ops))}")
    if getattr(args, "ops", None):
        print(f"# selected rows: {', '.join(sorted(args.ops))}")
    return meta


def finish_run(args: argparse.Namespace, suite: str, meta: dict[str, Any], results: list[dict[str, Any]]) -> None:
    """Write ``--json`` and ``--contribute`` outputs with the same messages for every suite."""
    if args.json:
        out = save_json(args.json, meta, results)
        print(f"# results written to {out}")
    if args.contribute:
        contribute_result(args.contribute, suite, meta, results, slug_override=args.machine_slug)


class KorniaRows:
    """Build the ``kornia (eager)`` / ``kornia (compiled)`` cells of one config's op rows.

    ``rows(label, target, *args)`` returns ``{"kornia (eager)": ..., "kornia (compiled)": ...}``
    where eager calls ``target(*args)`` and compiled calls ``torch.compile(target)(*args)`` after
    a warmup outside the timed region. A failed warmup leaves the compiled cell ``None`` and
    records the exception name in ``compile_failures`` for ``run_batch_sweep`` to report.

    ``step`` wraps both cells, e.g. to add a backward pass; the warmup runs the wrapped call, so
    a backward graph also compiles outside the timed region.

    ``reset_per_op`` resets dynamo before each op. The image suites need it: kornia's
    augmentation classes share one ``forward`` code object, so a config with more ops than the
    recompile limit would otherwise fall back to eager silently. A suite of distinct functions can
    reset once per config instead (``feature/laf_ops.py``), which keeps every earlier op's
    compiled graph valid until it is timed.
    """

    def __init__(
        self,
        device: torch.device,
        do_compile: bool,
        skip_compile: frozenset[str] = frozenset(),
        reset_per_op: bool = True,
        step: Optional[Callable[[Callable[[], object]], Callable[[], object]]] = None,
    ) -> None:
        self.device = device
        self.step = step if step is not None else (lambda call: call)
        self.do_compile = do_compile
        self.skip_compile = skip_compile
        self.reset_per_op = reset_per_op
        self.compile_failures: dict[str, str] = {}
        if do_compile and not reset_per_op:
            torch._dynamo.reset()

    def __call__(self, label: str, target: Callable[..., object], *args: Any) -> dict[str, Backend]:
        row: dict[str, Backend] = {"kornia (eager)": self.step(lambda: target(*args))}
        if not self.do_compile or label in self.skip_compile:
            return row
        if self.reset_per_op:
            torch._dynamo.reset()
        compiled = torch.compile(target)
        compiled_call = self.step(lambda: compiled(*args))
        try:
            compiled_call()  # warmup: compile + autotune before the timed region
            if self.device.type == "cuda":
                torch.cuda.synchronize()  # surface async kernel faults HERE, not at the next op
            row["kornia (compiled)"] = compiled_call
        except Exception as e:
            errors = str(e)
            if self.device.type == "cuda":
                try:
                    torch.cuda.synchronize()  # a FAILED warmup may still have launched kernels
                except Exception as sync_err:
                    errors += " | " + str(sync_err)
            if "illegal memory access" in errors:
                raise SystemExit(
                    f"FATAL: CUDA context poisoned during torch.compile warmup of '{label}' "
                    "(illegal memory access); no later measurement would be trustworthy. "
                    f"Rerun with --skip-compile-ops {label} to keep it eager-only, or without "
                    "--compile; CUDA_LAUNCH_BLOCKING=1 localizes the kernel."
                ) from e
            row["kornia (compiled)"] = None
            self.compile_failures[label] = type(e).__name__
        return row


def image_batch(b: int, h: int, w: int, device: torch.device, dtype: torch.dtype) -> tuple[list[Any], torch.Tensor]:
    """The shared seeded input: ``b`` uint8 HWC RGB images and the same batch as float BCHW in [0, 1]."""
    import numpy as np

    rng = np.random.default_rng(0)
    imgs_u8 = [(rng.random((h, w, 3)) * 255).astype(np.uint8) for _ in range(b)]
    batch_f = (
        torch.stack([torch.from_numpy(im).permute(2, 0, 1) for im in imgs_u8]).to(device=device, dtype=dtype).div(255)
    )
    return imgs_u8, batch_f


def image_row_fields(args: argparse.Namespace) -> Callable[[Any], dict[str, Any]]:
    """Per-row config fields of an image suite: the square size and dtype."""
    return lambda b: {"height": args.size, "width": args.size, "dtype": args.dtype}


def batch_list(args: argparse.Namespace) -> list[int]:
    return [int(x) for x in args.batches.split(",") if x.strip()]
