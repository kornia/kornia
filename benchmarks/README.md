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

For Oxford affine graf speed and reprojection error, run
`python -m benchmarks.feature.local_features --seq /data/graf --device cuda --json graf.json`
from the checkout root (`--device mps` is supported; it evaluates RANSAC on CPU). It compares SIFT, SIFT-AffNet-HardNet, and KeyNet-HardNet with fixed
pipeline settings; use `--device cpu --timing-pairs 2` to time the representative 1–2 pair
while still evaluating quality on all five pairs. The module docstring defines the exact
timed region and corner-error metric.
See the [graf CPU/CUDA comparison](feature/graf_benchmark.md) for measured results and raw JSON.
The [historical SIFT runtime chart](feature/sift_runtime.md) compares 0.8.2, 0.8.3,
current eager/compiled, and OpenCV across CPU and CUDA batch sizes 1, 4, and 8.

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
3. Run each suite with `--contribute` (most suites are a `flagship.py`; the directory map above
   names the script for the ones that are not, such as `feature/laf_ops.py`):

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

The full run is committed as
[`feature-laf-ops--i7-14700k-rtx-4090--cpu.json`](results/0.9.0rc1/feature-laf-ops--i7-14700k-rtx-4090--cpu.json)
and
[`feature-laf-ops--i7-14700k-rtx-4090--cuda.json`](results/0.9.0rc1/feature-laf-ops--i7-14700k-rtx-4090--cuda.json),
so the docs performance page renders it; the tables below are the B=1 N=20000 slice of those
files. Linux/WSL2 (kernel 6.18, x86_64), Intel i7-14700K + NVIDIA RTX 4090, Python 3.13,
torch 2.14.0+cu130, CUDA 13.0, kornia 0.9.0rc1, float32, image 256×256, patch size 32,
14 threads; LAF scales are stratified across all four pyramid levels that a 256×256 image at
PS=32 provides. Throughput in LAFs/s (higher is better); no cross-library column exists — no
other library exposes LAFs. Every op compiled on both devices on this stack, so there is no `-`
cell. Compare columns within one table, never numbers across machines.

`--device cpu --compile`, B=1 N=20000:

| op | kornia (eager) | kornia (compiled) |
| --- | --: | --: |
| laf_from_center_scale_ori | 39273029 | **99964263** |
| make_upright | 33959226 | **122176202** |
| ellipse_to_laf | 110509852 | **149762253** |
| laf_to_boundary_points | 3613236 | **9903714** |
| laf_is_inside_image | 6325084 | **11126178** |
| extract_patches_simple | **53309** | 27357 |
| extract_patches_from_pyramid | **37758** | 36983 |

`--device cuda --compile`, B=1 N=20000:

| op | kornia (eager) | kornia (compiled) |
| --- | --: | --: |
| laf_from_center_scale_ori | 147968107 | **254559803** |
| make_upright | 74905012 | **378282785** |
| ellipse_to_laf | 119929997 | **447807983** |
| laf_to_boundary_points | 14107628 | **17835708** |
| laf_is_inside_image | 28078872 | **56037949** |
| extract_patches_simple | 3499935 | **31880283** |
| extract_patches_from_pyramid | 2979848 | **25969422** |

Apple M1 (MacBook Air, 8 cores, 4 torch threads), macOS 26.5, Python 3.11, torch 2.9.1,
kornia 0.9.0rc1, same float32 / 256×256 / PS=32 / stratified-scale workload. Committed as
[`feature-laf-ops--apple-m1--cpu.json`](results/0.9.0rc1/feature-laf-ops--apple-m1--cpu.json)
and
[`feature-laf-ops--apple-m1--mps.json`](results/0.9.0rc1/feature-laf-ops--apple-m1--mps.json).
Every op compiled on both devices here too. This is a different CPU architecture, thread count
and torch version from the box above, so it answers "does the finding hold?", not "which box is
faster?".

`--device cpu --compile`, B=1 N=20000:

| op | kornia (eager) | kornia (compiled) |
| --- | --: | --: |
| laf_from_center_scale_ori | 25551758 | **84299262** |
| make_upright | 25171744 | **108229794** |
| ellipse_to_laf | 75817885 | **177777778** |
| laf_to_boundary_points | 838245 | **1078421** |
| laf_is_inside_image | 4880514 | **9227220** |
| extract_patches_simple | 40806 | **294408** |
| extract_patches_from_pyramid | 30084 | **200881** |

`--device mps --compile`, B=1 N=20000:

| op | kornia (eager) | kornia (compiled) |
| --- | --: | --: |
| laf_from_center_scale_ori | 13318018 | **22368750** |
| make_upright | 12793187 | **35554228** |
| ellipse_to_laf | 14233594 | **39990002** |
| laf_to_boundary_points | 1282786 | **1874608** |
| laf_is_inside_image | 2194321 | **4115015** |
| extract_patches_simple | 88430 | **476392** |
| extract_patches_from_pyramid | 73561 | **689227** |

Run-to-run spread on that laptop, from two independent full runs of each device: CPU cells move
by a median of 1.0% (worst 19%, on the cheapest op at the smallest config), MPS cells by a median
of 7.4% (worst 54%). Treat an MPS difference under ~2× on this class of machine as noise.

The honest reading:

- **Patch extraction still dominates, but only on CPU.** `extract_patches_from_pyramid` runs at
  ~38k LAFs/s eager on CPU — ~900× below `make_upright` on the same box, so a 20k-keypoint
  descriptor pass pays ~0.53 s in patch sampling before any descriptor math. On CUDA the same op
  is ~3.0M LAFs/s eager and ~26M compiled (0.77 ms for those 20k LAFs), only ~25× below the cheap
  ops. Batched GPU extraction is the regime to be in.
- **`torch.compile` loses on `extract_patches_simple` on *that* CPU, not on CPUs.** It is a
  consistent 1.9–2.1× loss in all three configs on the i7-14700K (torch 2.14, 14 threads) and a
  ~9.1× win for the same op on CUDA — but on the Apple M1 (torch 2.9.1, 4 threads) it is a
  **7.2× win** (40806 → 294408 LAFs/s), and 6.7× for the pyramid variant. So the regression is a
  property of that CPU inductor path, not of the op or of compiling for CPU, and a fix should
  start by bisecting stack against machine rather than reading the kernel. It is not the
  N-chunking added in #4128: at B=1 N=2000 the sampling grid is ~16 MiB against a 64 MiB budget,
  so that config takes the single-chunk fast path and still loses.
- **`laf_to_boundary_points` was the weakest compiled op on CUDA** — 17.8M LAFs/s, below both
  patch extractors, ~25× below `ellipse_to_laf`, and the least moved by `torch.compile` there
  (1.26×, against up to 9.1× elsewhere). On CPU it was ~9× slower than the similar-sized
  `make_upright`. #4217 removed the cause; see the last bullet for what it actually was. The
  tables above predate that change, as do their `laf_is_inside_image` rows, which share it.
- **Small N on CUDA is launch-latency bound, not work bound:** at B=1 N=2000 the three cheapest
  ops all land within 8.2–14.2M LAFs/s regardless of what they compute, ~8.5–11× below their own
  B=1 N=20000 figures. Read the N=20000 rows for kernel cost and the N=2000 rows for per-call
  overhead.
- **Pyramid depth is currently free.** Stratifying the LAF scales across all four levels, instead
  of leaving every LAF on level 0, moved `extract_patches_from_pyramid` by ~3% — the packed-atlas
  implementation from #4128 pays for the whole atlas whatever the LAFs select. That is a property
  of today's code, not of the op: a change that skips unused levels would look free on an
  all-level-0 workload, which is why the generator stratifies.
- **`ellipse_to_laf` is now the fastest LAF op measured**, at 110.5M LAFs/s eager on CPU. This
  benchmark originally recorded it as a hot spot (a batched 2×2 `torch.inverse`, pathological on
  MPS); the closed-form inverse in #4122 fixed it, and the dedicated A/B lives in
  [`ellipse_to_laf.py`](feature/ellipse_to_laf.py).
- **On Apple silicon, MPS is the wrong device for the cheap LAF ops.** Every bookkeeping op is
  faster in M1 *CPU* eager than in MPS eager — `ellipse_to_laf` by 5.3× (75.8M vs 14.2M LAFs/s),
  `make_upright` and `laf_from_center_scale_ori` by ~2× — and the gap widens under
  `torch.compile` (4.4× and 3.8×). MPS only wins where there is real work per LAF: patch
  extraction, by 2.2–2.4× eager and up to 3.4× compiled. A pipeline that moves LAFs to the GPU
  for the frame math alone pays for the transfer twice.
- **`laf_to_boundary_points` was the worst op on every machine measured, and the obvious
  diagnosis was the wrong one.** It built its 50-point basis with no `device=`, `.expand()`ed it
  to `(B*N, n_pts, 3)` and only then called `.to(device)`, so every call materialized — and on a
  GPU transferred — a tensor scaling with the LAF count: ~12 MiB per call at N=20000 for a basis
  with 50 distinct rows. That is real, and it is a memory problem, not the speed problem: fixing
  only it measures **1.00× on CPU**. The cost was one gemm-shape cliff. The op appended a constant
  `[0, 0, 1]` row to every LAF so that it could divide the result by a homogeneous coordinate that
  is always exactly 1, and torch's CPU batched gemm falls off a fast path at that third row — at
  B=1 N=20000, `bmm` on a `(B, 3, 3)` operand takes 14.9 ms where `(B, 2, 3)` takes 0.65 ms, 23×
  for 1.5× the arithmetic, identically at 1, 4 and 8 threads (M=1 0.33 ms, M=2 0.65, M=3 14.87,
  M=4 16.26). #4217 multiplies by the `(2, 3)` LAF directly. Quiet back-to-back A/B on the M1
  (torch 2.14.0, 4 threads, B=1 N=20000): `laf_to_boundary_points` 1.07M → 29.6M LAFs/s eager on
  CPU and 1.77M → 4.27M on MPS; `laf_is_inside_image`, which calls it with `n_pts=12` on every
  detector forward, 9.75M → 29.0M on CPU and 2.70M → 8.53M on MPS. **The lesson is the method:**
  the transfer was visible by reading the source and the cliff was only visible by timing the
  components, so the readable diagnosis got written up first and would have shipped a 1.00× fix.

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
