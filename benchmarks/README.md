# Kornia benchmarks

Reproducible speed/quality benchmarks for the public kornia API, with honest cross-library
baselines. Goal: current, citable numbers with disclosed methodology — where kornia wins
**and** where it loses — replacing stale performance anecdotes.

## Directory map

| Directory | Contents |
| --- | --- |
| [`augmentation/`](augmentation/) | Cross-library augmentation throughput; see its [README](augmentation/README.md). |
| [`geometry/`](geometry/) | [`flagship.py`](geometry/flagship.py): core geometry ops vs OpenCV/torchvision v2. |
| [`color/`](color/) | pytest-benchmark microbenchmarks for color conversions. |
| [`feature/`](feature/) | Local-feature detector benchmarks incl. quality (matching) metrics. |
| [`common.py`](common.py) | Shared methodology utilities — use these in every new benchmark. |

## Methodology contract

Every benchmark here must follow the same rules (utilities in [`common.py`](common.py)):

- **Warmup + repeats:** time with `common.time_us(fn)` — it wraps
  `torch.utils.benchmark.Timer.blocked_autorange`, which warms up, runs many repeats, and
  reports **median** wall clock; `time_us` additionally returns the **IQR** as the spread.
  Never time a single call.
- **Device sync inside the timed region:** `blocked_autorange` syncs CUDA; for MPS pass
  `sync=torch.mps.synchronize` to `time_us`. A hand-rolled `time.time()` around a GPU call
  measures launch latency, not work.
- **Pinned seeds:** seed every RNG (`torch.manual_seed`, `np.random.default_rng(0)`) so runs
  are reproducible bit-for-bit on the same software stack.
- **Recorded metadata:** embed `common.run_metadata(device)` in every result file — date, git
  commit, platform, Python/torch/kornia versions, device (CUDA name + version when
  applicable), thread count, and baseline-library versions.
- **Machine-readable export:** support `--json PATH` and write via `common.save_json` —
  strict-valid JSON (`NaN` → `null`), shape `{"metadata": {...}, "results": [...]}`.
- **Equal footing + honest regimes:** identical transform parameters and interpolation across
  backends; state each backend's regime (batched float tensor vs per-image uint8 loop) instead
  of pretending the columns are apples-to-apples. Publish losses alongside wins.
- **Public API only:** benchmark `kornia.*` as users call it — no private helpers, no
  reimplementations inside the script.

## JSON schema

One file per run:

```json
{
  "metadata": {
    "timestamp_utc": "2026-08-07T17:58:12+00:00",
    "git_commit": "407b6dce",
    "platform": "macOS-26.5.1-arm64-arm-64bit",
    "machine": "arm64",
    "python": "3.11.14",
    "torch": "2.9.1",
    "kornia": "0.9.0rc1",
    "device": "cpu",
    "torch_num_threads": 4,
    "opencv": "4.11.0",
    "torchvision": null,
    "numpy": "2.4.0"
  },
  "results": [
    {
      "op": "warp_perspective",
      "backend": "kornia (eager)",
      "batch": 8,
      "height": 256,
      "width": 256,
      "dtype": "float32",
      "median_us": 1983.4,
      "iqr_us": 12.1,
      "throughput_per_s": 4033.5
    }
  ]
}
```

`throughput_per_s` counts items per second — images for image ops, point-set solves for
`get_perspective_transform`. `null` means the measurement failed (backend raised).

## Adding a new benchmark

1. Import the utilities (`benchmarks/` is not a package — scripts under a subdirectory add the
   parent to `sys.path`, see the top of `geometry/flagship.py`).
2. Time every backend with `time_us`, print a table with the git commit, platform, and device
   name in the header, and support `--json` via `save_json(path, run_metadata(device), results)`.
3. Baselines run **correctly and on equal footing** (same parameters, same interpolation, their
   native data regime) — a misconfigured baseline is worse than no baseline.
4. Missing optional libraries must degrade to a skip note, never a crash.
5. Document the regimes in the module docstring; keep the honest framing.

## Sample results — geometry flagship ops

Directional numbers only — reproduce on your own hardware for anything you cite. Measured
2026-08-07, commit `5eaa7a10`, Apple Silicon (macOS 26.5, arm64), Python 3.11, torch 2.9.1,
kornia 0.9.0rc1, OpenCV 4.11.0, float32, 256×256, 4 threads. Throughput in items/s (higher is
better); kornia runs a batched float BCHW tensor, OpenCV a per-image uint8 loop on CPU.

`--device cpu --compile`:

| batch=32 | kornia (eager) | kornia (compiled) | opencv |
| --- | --: | --: | --: |
| warp_perspective | 785 | 1079 | 2680 |
| warp_affine | 841 | 1389 | 3073 |
| rotate | 909 | 1400 | 3195 |
| resize | 3044 | 21834 | 32443 |
| get_perspective_transform | 242759 | **808428** | 753415 |

`--device mps --compile`:

| batch=32 | kornia (eager) | kornia (compiled) | opencv (CPU) |
| --- | --: | --: | --: |
| warp_perspective | 2072 | **4503** | 2861 |
| warp_affine | 2433 | **6662** | 3385 |
| rotate | 2019 | **4418** | 3443 |
| resize | 25576 | 26561 | 34531 |
| get_perspective_transform | 16336 | 69578 | 784576 |

The honest reading: OpenCV's uint8 single-image loop wins the CPU regime outright; compiled
kornia on an accelerator overtakes it on the warps once batched (kornia's regime), and the
batched `get_perspective_transform` solver beats OpenCV's per-pair solver even on CPU at
batch 32. `torch.compile` is worth 1.3–7× across these ops.
