"""Per-transform CUDA Graph capture harness.

Attempts torch.cuda.graph capture on each kornia transform at a fixed input
shape. Reports per-transform (capture_status, eager_ms, replay_ms, speedup,
launch_overhead_pct) into a JSONL log.

Capture failures are the strongest test that a transform has no host syncs.
"""
from __future__ import annotations
import dataclasses
import json
import sys
from pathlib import Path
import torch

@dataclasses.dataclass
class GraphBenchResult:
    transform: str
    capture_status: str            # "OK" | "FAILED" | "SKIPPED"
    capture_error: str | None
    eager_ms: float | None
    replay_ms: float | None
    speedup: float | None
    launch_overhead_pct: float | None
    capture_ms: float | None
    n_replays: int


def _bench_one(
    transform_factory,                # callable returning a fresh transform
    name: str,
    input_shape: tuple[int, ...] = (8, 3, 512, 512),
    device: str = "cuda",
    warmup: int = 25,
    n_replays: int = 100,
) -> GraphBenchResult:
    if not torch.cuda.is_available():
        return GraphBenchResult(name, "SKIPPED", "no CUDA", None, None, None, None, None, 0)
    try:
        transform = transform_factory().to(device)
        transform.train(False)
    except Exception as e:
        return GraphBenchResult(name, "FAILED", f"construct: {e}", None, None, None, None, None, 0)
    static_x = torch.randn(*input_shape, device=device)

    # Warmup
    try:
        for _ in range(warmup):
            _out = transform(static_x)
        torch.cuda.synchronize()
    except Exception as e:
        return GraphBenchResult(name, "FAILED", f"warmup: {e}", None, None, None, None, None, 0)

    # Eager timing via cuda events
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(n_replays)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(n_replays)]
    for i in range(n_replays):
        starts[i].record()
        _out = transform(static_x)
        ends[i].record()
    torch.cuda.synchronize()
    eager_times = sorted(s.elapsed_time(e) for s, e in zip(starts, ends))
    eager_ms = eager_times[len(eager_times) // 2]

    # Capture
    g = torch.cuda.CUDAGraph()
    cap_start = torch.cuda.Event(enable_timing=True)
    cap_end = torch.cuda.Event(enable_timing=True)
    try:
        cap_start.record()
        with torch.cuda.graph(g):
            _out = transform(static_x)
        cap_end.record()
        torch.cuda.synchronize()
        capture_ms = cap_start.elapsed_time(cap_end)
    except Exception as e:
        return GraphBenchResult(name, "FAILED", f"capture: {e}", eager_ms, None, None, None, None, 0)

    # Replay timing
    rstarts = [torch.cuda.Event(enable_timing=True) for _ in range(n_replays)]
    rends = [torch.cuda.Event(enable_timing=True) for _ in range(n_replays)]
    for i in range(n_replays):
        rstarts[i].record()
        g.replay()
        rends[i].record()
    torch.cuda.synchronize()
    replay_times = sorted(s.elapsed_time(e) for s, e in zip(rstarts, rends))
    replay_ms = replay_times[len(replay_times) // 2]

    speedup = eager_ms / replay_ms if replay_ms > 0 else None
    overhead = 100.0 * (1 - replay_ms / eager_ms) if eager_ms > 0 else None
    return GraphBenchResult(name, "OK", None, eager_ms, replay_ms, speedup, overhead, capture_ms, n_replays)


def discover_transforms() -> list[tuple[str, callable]]:
    """Return list of (name, factory) pairs for every public concrete kornia transform."""
    import inspect
    import kornia.augmentation as K
    out: list[tuple[str, callable]] = []
    for name in sorted(dir(K)):
        if name.startswith("_"):
            continue
        obj = getattr(K, name)
        if not inspect.isclass(obj):
            continue
        if not issubclass(obj, torch.nn.Module):
            continue
        # Filter to concrete transforms — skip abstract bases and containers
        if name.endswith("Sequential") or name.endswith("Base") or name.endswith("Base2D") or name.endswith("Base3D"):
            continue
        if "Generator" in name:
            continue
        # Try to construct with no args
        try:
            sig = inspect.signature(obj.__init__)
            required = [
                p for p in sig.parameters.values()
                if p.name != "self"
                and p.default is inspect.Parameter.empty
                and p.kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
            ]
            if required:
                # Skip transforms that need required args we can't auto-fill
                continue
        except (ValueError, TypeError):
            continue
        out.append((name, obj))
    return out


def run_all(output_jsonl: Path, input_shape: tuple[int, ...] = (8, 3, 512, 512)) -> list[GraphBenchResult]:
    transforms = discover_transforms()
    results: list[GraphBenchResult] = []
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with output_jsonl.open("w") as fh:
        for name, factory in transforms:
            r = _bench_one(factory, name, input_shape=input_shape)
            results.append(r)
            fh.write(json.dumps(dataclasses.asdict(r)) + "\n")
            fh.flush()
            print(f"{r.capture_status:<8} {name:<40} eager={r.eager_ms} replay={r.replay_ms}")
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="benchmarks/cuda_graph/results.jsonl")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--resolution", type=int, default=512)
    args = parser.parse_args()
    results = run_all(Path(args.output), (args.batch, 3, args.resolution, args.resolution))
    n_failed = sum(1 for r in results if r.capture_status == "FAILED")
    n_ok = sum(1 for r in results if r.capture_status == "OK")
    n_skipped = sum(1 for r in results if r.capture_status == "SKIPPED")
    print(f"\n{n_ok} OK, {n_failed} FAILED, {n_skipped} SKIPPED out of {len(results)}")
    # Exit 1 if any FAILED; exit 0 if all SKIPPED (no CUDA) or all OK
    sys.exit(1 if n_failed > 0 else 0)
