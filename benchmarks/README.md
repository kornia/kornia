# Kornia benchmarks

Reproducible speed/quality benchmarks for the public kornia API, with honest cross-library
baselines. Goal: current, citable numbers with disclosed methodology — where kornia wins
**and** where it loses — replacing stale performance anecdotes.

## Directory map

| Directory | Contents |
| --- | --- |
| [`augmentation/`](augmentation/) | Cross-library augmentation benchmarks — [`flagship.py`](augmentation/flagship.py) (class-API, parameter sampling included, vs torchvision v2/albumentations/OpenCV/PIL) plus pipeline/per-op scripts; see its [README](augmentation/README.md). |
| [`geometry/`](geometry/) | [`flagship.py`](geometry/flagship.py): core geometry ops vs OpenCV/torchvision v2. |
| [`filters/`](filters/) | [`flagship.py`](filters/flagship.py): core filters vs OpenCV/albumentations/torchvision v2/kornia-rs/PIL. |
| [`color/`](color/) | pytest-benchmark microbenchmarks for color conversions. |
| [`feature/`](feature/) | Local-feature detector benchmarks incl. quality (matching) metrics; [`laf_ops.py`](feature/laf_ops.py) microbenchmarks the shared LAF operations and [`ellipse_to_laf.py`](feature/ellipse_to_laf.py) drills into one of them (both base-revision A/B — no cross-library baseline exists). |
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

## Contributing results (any machine)

1. Check out the release tag you are measuring.
2. Quiet the machine: close other applications, mains power, let it cool. Only aggregate load
   numbers (load average, memory) are recorded in the file - never process or app names.
3. Run each suite with `--contribute`:

       python benchmarks/augmentation/flagship.py --device cuda --contribute benchmarks/results

   The run lands at `benchmarks/results/<kornia-version>/<suite>--<machine>--<device>.json`
   (override the machine name with `--machine-slug`).
4. Commit the file and open a PR. CI validates the schema (`benchmarks/results_schema.py`);
   the docs page and the llms digest regenerate from it automatically at the next docs build
   (`python docs/generate_benchmarks.py --refresh-llms` refreshes the committed digest).

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

### CUDA (batch=32, fp32, 256×256, throughput items/s)

Measured 2026-08-07 on commits `b317c16d`/`f4cb83eb`, torch 2.x, full tables in
[PR #3906](https://github.com/kornia/kornia/pull/3906). OpenCV column is the same box's CPU
uint8 per-image loop.

NVIDIA L4 (Intel Cascade Lake host):

| op | kornia (eager) | kornia (compiled) | torchvision v2 | opencv (CPU) |
| --- | --: | --: | --: | --: |
| warp_perspective | 24897 | **56552** | - | 1246 |
| warp_affine | 27747 | **53575** | - | 1946 |
| rotate | 14230 | 44547 | **74258** | 2002 |
| resize | **1021334** | 322329 | 838383 | 25742 |
| get_perspective_transform | 26723 | **139442** | - | 388340 |

NVIDIA RTX PRO 6000 Blackwell (AMD Turin host):

| op | kornia (eager) | kornia (compiled) | torchvision v2 | opencv (CPU) |
| --- | --: | --: | --: | --: |
| warp_perspective | 96022 | **232170** | - | 3223 |
| warp_affine | 120089 | **217083** | - | 5747 |
| rotate | 58357 | 159196 | **298016** | 5742 |
| resize | 625142 | **1204817** | 625100 | 60394 |
| get_perspective_transform | 78071 | **431034** | - | 272523 |

### The honest reading (across Apple Silicon, L4, RTX PRO 6000, RTX 4090/WSL2)

- **Batched GPU is kornia's regime and the margin is large:** compiled `warp_perspective` at
  batch 32 beats OpenCV's per-image CPU loop by ~45× (L4) to ~72× (RTX PRO 6000); eager alone
  is ~20–30×.
- **`rotate` is a found weak spot:** torchvision v2 beats kornia on every GPU tested
  (~1.7–2×, up to ~4× vs eager where compile was unavailable). First data-driven optimization
  target for the Stage-3 iteration.
- **`resize`:** kornia eager matches torchvision almost exactly (same underlying kernel);
  `torch.compile` is a large win at big batches on newer GPUs (3.6M img/s at batch 128 on
  Blackwell) but *regressed* resize on L4 at batch ≤ 32 — compile is not a free win, measure
  per shape.
- **Batched `get_perspective_transform` beats OpenCV's per-pair solver even on CPU** once
  batched: crossover by batch ≈ 32, up to ~13× at batch 128 on an AMD Turin CPU
  (3.5M solves/s compiled).
- **CPU per-image warps remain OpenCV's win everywhere**, as expected and published.
- **WSL2 + RTX 4090: inductor failed for all ops** (`InductorError`, reported by the harness's
  compile-failure NOTE rather than silently skipped); eager still led OpenCV by ~23× on
  batch-128 warp_perspective.

## Sample results — feature LAF ops

Directional numbers only — reproduce on your own hardware for anything you cite. Measured
2026-08-30, commit `2009933e`, Apple Silicon (macOS 26.5, arm64), Python 3.11, torch 2.9.1,
kornia 0.9.0rc1, float32, image 256×256, patch size 32, 4 threads. Throughput in LAFs/s
(higher is better); no cross-library column exists — no other library exposes LAFs.

`--device cpu --compile`, B=1 N=20000:

| op | kornia (eager) | kornia (compiled) |
| --- | --: | --: |
| laf_from_center_scale_ori | 23969858 | **83857443** |
| make_upright | 24353058 | **94321158** |
| ellipse_to_laf | 7977868 | 8732195 |
| laf_to_boundary_points | 809834 | 1022052 |
| laf_is_inside_image | 4726938 | **9708540** |
| extract_patches_simple | 41525 | **320274** |
| extract_patches_from_pyramid | 10050 | **75731** |

`--device mps --compile`, B=1 N=20000 (`-` = torch.compile warmup failed, reported as a NOTE):

| op | kornia (eager) | kornia (compiled) |
| --- | --: | --: |
| laf_from_center_scale_ori | 10680032 | **21311556** |
| make_upright | 11381861 | **33145729** |
| ellipse_to_laf | 10907 | - |
| laf_to_boundary_points | 1302105 | 1683478 |
| laf_is_inside_image | 1966278 | - |
| extract_patches_simple | 96756 | **1328058** |
| extract_patches_from_pyramid | 23636 | 299586 |

The honest reading:

- **Patch extraction dominates everything.** `extract_patches_from_pyramid` runs at ~10k LAFs/s
  eager on CPU — a 20k-keypoint `SIFTDescriptor` pass pays ~2 s in patch sampling before any
  descriptor math. It runs a full `grid_sample` of all N patches at every pyramid level
  (deliberate, for compile-friendliness); `torch.compile` recovers ~7.5× for both extractors
  on CPU and ~13× for the simple one on MPS at B=1 — but the win collapses with batch: at
  B=8 the compiled pyramid extractor is no faster than eager on either device.
- **Found weak spot: `ellipse_to_laf` on MPS is ~700× slower than CPU** (10.7k vs 7.7M LAFs/s)
  — the batched 2×2 `torch.inverse` hits a pathological MPS linalg path and additionally emits
  a deprecated-resize `UserWarning` per call from `laf.py`. A closed-form 2×2 inverse removes
  both; on CPU it is also the difference between 7.7M LAFs/s and an arithmetic-bound kernel.
- **`laf_to_boundary_points` is ~30× slower than the similar-sized `make_upright`** on CPU and
  the only op whose IQR is large (~40% of the median): it rebuilds its `linspace` basis on CPU
  in float32 every call and pays a host→device transfer.
- **Compile coverage on MPS is holey:** `ellipse_to_laf` (`AssertionError`) and
  `laf_is_inside_image` (`InductorError`) fail to compile on this stack in every config.

## Sample results — augmentation flagship (class API)

Directional numbers only — reproduce on your own hardware for anything you cite. Measured
2026-08-08, commit `c97b0f9a`, Apple Silicon (macOS 26.5, arm64), Python 3.11, torch 2.9.1,
torchvision 0.24.1, albumentations 2.0.8, OpenCV 4.11.0, Pillow 12.3, float32, 256×256,
4 threads, batch 32, throughput img/s. Timed region = parameter sampling + application through
each library's random-transform class API; kornia/torchvision run a batched float tensor,
albumentations/OpenCV/PIL a per-image uint8 CPU loop. CUDA tables follow the PR-thread protocol
used for the geometry suite.

`--device cpu --compile`:

| op | kornia (eager) | kornia (compiled) | torchvision v2 | albumentations | opencv | PIL |
| --- | --: | --: | --: | --: | --: | --: |
| RandomHorizontalFlip | 10488 | 16754 | 10481 | 33421 | **37019** | 9813 |
| RandomAffine | 899 | 1939 | 1373 | **5162** | - | - |
| RandomPerspective | 843 | 1396 | 1086 | **5890** | - | - |
| RandomResizedCrop | 3653 | ✗ | 3975 | **20241** | - | - |
| ColorJiggle | 105 | ✗ | 269 | **2060** | - | - |
| RandomGaussianBlur | 1103 | 75 | 993 | **5008** | - | - |
| RandomBrightness | 6457 | **15590** | 9156 | 13846 | - | - |
| RandomGrayscale | 5615 | 12592 | 22666 | 23614 | **49286** | 14373 |

`--device mps --compile` (uint8 loop backends are CPU, repeated for reference):

| op | kornia (eager) | kornia (compiled) | torchvision v2 | albumentations | opencv | PIL |
| --- | --: | --: | --: | --: | --: | --: |
| RandomHorizontalFlip | 16651 | 21301 | 20428 | 62662 | **77361** | 13492 |
| RandomAffine | 1605 | 2701 | 2600 | **6508** | - | - |
| RandomPerspective | 1611 | ✗ | 2071 | **7481** | - | - |
| RandomResizedCrop | 3680 | ✗ | 17471 | **26307** | - | - |
| ColorJiggle | 48 | ✗ | 538 | **2548** | - | - |
| RandomGaussianBlur | 2968 | 4315 | 3616 | **6026** | - | - |
| RandomBrightness | 4018 | 13510 | 7692 | **24982** | - | - |
| RandomGrayscale | 5306 | 17504 | 10569 | 39590 | **92889** | 18483 |

The honest reading (this box only — an integrated GPU is not the datacenter regime):

- **albumentations owns the CPU single-image race here**, winning almost every row at batch ≤ 32
  — published as-is; kornia's regime is large-batch discrete-GPU + differentiable.
- **Found weak spot: `ColorJiggle`** — ~20× behind albumentations and ~2.5× behind torchvision's
  `ColorJitter` on CPU, worse on MPS (48 img/s), and `torch.compile` fails on it
  (`InductorError` on this stack). On MPS its path hits a `torch._assert_async` CPU fallback in
  `kornia/enhance/adjust.py` (MPS does not support the op), which forces device sync per call.
- **Found weak spot: compile coverage** — `RandomResizedCrop` fails to compile
  (`GuardOnDataDependentSymNode` from data-dependent crop parameters) on both devices;
  `RandomPerspective` additionally fails on MPS — and on an NVIDIA L4 its *compiled* warmup goes
  further and triggers a **CUDA illegal memory access** (inductor emits an out-of-bounds indexing
  kernel for the data-dependent parameter graph; the harness names the op and exits, and
  `--skip-compile-ops RandomPerspective` keeps the rest of the compiled column measurable).
  Direct input for the S5 compile-cleanliness work.
- **`RandomGaussianBlur` compiled regresses ~15× on CPU** (conv-bound; compile overhead exceeds
  the kernel) — consistent with the historical all-libraries finding; don't compile blindly.
- Where compile works on pointwise ops it delivers: `RandomBrightness` 6.5k → 15.6k (CPU),
  4k → 13.5k (MPS); `RandomGrayscale` 2.2× (CPU) / 3.3× (MPS).

## Sample results — filters flagship

Same machine, stack, and caveats as above; batch 32, 256×256, float32, throughput img/s.
kornia-rs 0.1.10 (this wheel) ships no filter functions — its column is skipped and reported.
PIL matches exactly on box (`BoxBlur(2)`) and median (`MedianFilter(5)`); its Gaussian is a
box-approximation with radius = sigma (matched in spirit).

`--device cpu --compile`:

| op | kornia (eager) | kornia (compiled) | torchvision v2 | albumentations | opencv | PIL |
| --- | --: | --: | --: | --: | --: | --: |
| gaussian_blur2d | 423 | 391 | 672 | 5345 | **9253** | 844 |
| sobel | 826 | 1183 | - | - | **2907** | - |
| laplacian | 674 | 618 | - | - | **2593** | - |
| median_blur | 13 | 13 | - | 7552 | **7990** | 21 |
| box_blur | 676 | 625 | - | 15116 | **16646** | 1940 |
| canny | 37 | 74 | - | - | **2568** | - |

`--device mps --compile` (uint8 loop backends are CPU, repeated for reference):

| op | kornia (eager) | kornia (compiled) | torchvision v2 | albumentations | opencv | PIL |
| --- | --: | --: | --: | --: | --: | --: |
| gaussian_blur2d | 2941 | 2592 | 3860 | 5988 | **12327** | 869 |
| sobel | 2688 | **6673** | - | - | 3440 | - |
| laplacian | **5891** | 4411 | - | - | 2905 | - |
| median_blur | 1 | 1 | - | 6191 | **6473** | 20 |
| box_blur | 7233 | 4493 | - | **14575** | 14226 | 1860 |
| canny | 187 | 376 | - | - | **2406** | - |

The honest reading:

- **Found weak spot (worst in the whole harness so far): `median_blur`** — 13 img/s CPU and
  **1 img/s MPS** vs ~8k for OpenCV/albumentations and 21 for PIL. The unfold-based kernel is
  ~600× off the native implementations; top Stage-3 candidate alongside `rotate`.
- **Found weak spot: `canny`** — ~35–70× behind OpenCV on CPU (37–74 vs 2568 img/s).
- Even on an integrated GPU, batched MPS starts winning the derivative filters: `sobel`
  (compiled) and `laplacian` beat the OpenCV loop, and `box_blur` closes to within 2× of the
  uint8 backends; the remaining blurs still lose to SIMD.
- PIL is the slowest float-correct reference on most ops, as expected — but still ~1.6–3× ahead
  of kornia's CPU blurs at this scale, and ~60× ahead on `median_blur`.
