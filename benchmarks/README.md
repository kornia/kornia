# Kornia benchmarks

Reproducible speed/quality benchmarks for the public kornia API, with honest cross-library
baselines. Goal: current, citable numbers with disclosed methodology — where kornia wins
**and** where it loses — replacing stale performance anecdotes.

## Directory map

Every kornia module with image or tensor ops a user would time has one `flagship.py`: a small,
representative set of that module's ops — not every export — against the libraries users would
otherwise reach for. All flagships share one command line, one console layout and one JSON format
(see [Shared output format](#shared-output-format)).

| Suite | Flagship ops | Baselines |
| --- | --- | --- |
| [`augmentation`](augmentation/flagship.py) | flip, affine, perspective, resized crop, color jiggle, blur, brightness, grayscale (class API, sampling included) | torchvision v2, albumentations, OpenCV, PIL |
| [`color`](color/flagship.py) | grayscale, HSV, Lab, YCbCr, Bayer demosaic | torchvision v2, scikit-image, OpenCV (per image and stacked), PIL |
| [`contrib`](contrib/flagship.py) | exact connected components, distance transform | scikit-image, SciPy, OpenCV |
| [`enhance`](enhance/flagship.py) | normalize, gamma, hue, saturation, equalize, CLAHE | torchvision v2, albumentations, scikit-image, OpenCV, PIL |
| [`feature`](feature/flagship.py) | Harris and GFTT responses, SIFT detect+describe, SNN matching | scikit-image, OpenCV |
| [`filters`](filters/flagship.py) | Gaussian, Sobel, Laplacian, median, box, Canny, unsharp, bilateral, guided, motion, Otsu | torchvision v2, albumentations, scikit-image, OpenCV, kornia-rs, PIL |
| [`geometry`](geometry/flagship.py) | warp perspective/affine, rotate, resize, perspective-transform solve | torchvision v2, OpenCV |
| [`io`](io/flagship.py) | JPEG and PNG decode to an RGB uint8 tensor | torchvision, OpenCV, PIL |
| [`losses`](losses/flagship.py) | binary focal, focal, Dice, SSIM, total variation (forward + backward) | torchvision |
| [`metrics`](metrics/flagship.py) | PSNR, SSIM, mean IoU | scikit-image, OpenCV |
| [`morphology`](morphology/flagship.py) | dilation, erosion, opening, gradient | torchmorph (CUDA only), albumentations, scikit-image, OpenCV |

Modules without a flagship: `models`, `tracking` and the model wrappers in `contrib` need
downloaded weights (learned local features are covered by `feature/local_features.py`); `nerf`,
`sensors`, `x`, `onnx`, `transpiler`, `grad_estimator`, `image`, `core`, `utils` and `testing`
are training loops, containers, exporters or plumbing whose cost is the ops above.

Deeper, single-topic scripts:

| Directory | Contents |
| --- | --- |
| [`augmentation/`](augmentation/) | Cross-library augmentation benchmarks — [`flagship.py`](augmentation/flagship.py) (class-API, parameter sampling included, vs torchvision v2/albumentations/OpenCV/PIL) plus pipeline/per-op scripts; see its [README](augmentation/README.md). |
| [`geometry/`](geometry/) | [`flagship.py`](geometry/flagship.py): core geometry ops vs OpenCV/torchvision v2. [`ransac.py`](geometry/ransac.py): batched RANSAC runtime and IMC pose accuracy on cached correspondences; [audit and results](geometry/ransac.md). |
| [`morphology/`](morphology/) | [`flagship.py`](morphology/flagship.py): representative morphology ops vs torchmorph, albumentations, scikit-image and OpenCV. [`engines.py`](morphology/engines.py): dilation engine comparison across explicit public engines. |
| [`filters/`](filters/) | [`flagship.py`](filters/flagship.py): core filters vs OpenCV/albumentations/torchvision v2/kornia-rs/PIL/scikit-image. [`gaussian_cpu.py`](filters/gaussian_cpu.py): Gaussian blur and scale-pyramid base/branch timing and numerical comparisons; [report](filters/gaussian_cpu.md). |
| [`color/`](color/) | pytest-benchmark microbenchmarks for color conversions (`*_test.py`). |
| [`contrib/`](contrib/) | [`connected_components.py`](contrib/connected_components.py): union-find vs pooling labeling, validated against SciPy; [report](contrib/connected_components.md). |
| [`feature/`](feature/) | Local-feature detector benchmarks incl. quality (matching) metrics; [`laf_ops.py`](feature/laf_ops.py) microbenchmarks the shared LAF operations and [`ellipse_to_laf.py`](feature/ellipse_to_laf.py) drills into one of them (both base-revision A/B — no cross-library baseline exists). [`local_features.py`](feature/local_features.py) measures Oxford graf speed and homography corner error for SIFT, SIFT-AffNet-HardNet and KeyNet-HardNet on CPU, CUDA or MPS (`--device cpu --timing-pairs 2` times the representative 1–2 pair and still scores all five); results in [`graf_benchmark.md`](feature/graf_benchmark.md). [`sift_runtime.py`](feature/sift_runtime.py) and [`plot_sift_runtime.py`](feature/plot_sift_runtime.py) chart scale-space SIFT runtime across releases and batch sizes; results in [`sift_runtime.md`](feature/sift_runtime.md). |
| [`common.py`](common.py) | Shared methodology utilities — use these in every new benchmark. |

[`feature/sift_scale_space.py`](feature/sift_scale_space.py) compares complete SIFT
extraction, matching and homography quality on CPU, CUDA or MPS;
[device results and usage](feature/sift_summary.md).

## Methodology contract

Every benchmark here must follow the same rules (utilities in [`common.py`](common.py)):

- **Warmup + repeats:** time with `common.time_us(fn)` — it wraps
  `torch.utils.benchmark.Timer.blocked_autorange`, which warms up, runs many repeats, and
  reports **median** wall clock; `time_us` additionally returns the **IQR** as the spread.
  Never time a single call.
- **Thread consistency:** `time_us` uses the current `torch.get_num_threads()` for timing,
  matching warmup and metadata. Older results collected before this fix timed PyTorch at
  `Timer`'s default of one thread even when metadata named a larger thread count; do not
  interpret those historical files as measurements at the advertised count.
  When comparing against a revision with the old timer, use one thread in both runs
  or apply the timer correction to the baseline as well.
- **Sustained CPU warm-up:** call `common.warm_up_cpu()` once after setting the thread count.
  Hybrid CPUs (performance + efficiency cores) keep lightly loaded threads on efficiency cores
  until they have carried sustained load, and WSL2 cannot pin them. On an i7-14700K this moved a
  5x5 oneDNN convolution from 0.56 ms to 0.22 ms while a slice-based filter barely changed, so
  an unwarmed A/B can pick the wrong implementation. Every flagship does this through
  `common.setup_run`, including accelerator runs because they also time CPU-only library baselines.
- **Checkout provenance:** every flagship imports Kornia from its own checkout, prints the source
  path, warns when kornia resolved elsewhere, and exports the checkout-relative path as
  `kornia_module`. Direct script execution must not silently benchmark an installed wheel or
  another editable checkout while recording the current tree's commit.
- **Device sync inside the timed region:** `blocked_autorange` syncs CUDA; for MPS pass
  `sync=torch.mps.synchronize` to `time_us`. A hand-rolled `time.time()` around a GPU call
  measures launch latency, not work.
- **Pinned seeds:** seed every RNG (`torch.manual_seed`, `np.random.default_rng(0)`) so runs
  are reproducible bit-for-bit on the same software stack.
- **Recorded metadata:** embed `common.run_metadata(device)` in every result file — date, git
  commit, platform, Python/torch/kornia versions, device (CUDA name + version when
  applicable), thread count, and baseline-library versions.
- **Version + commit identify a run, not its date:** a `<kornia-version>` directory spans many
  commits, so a snapshot can carry the current version and a recent timestamp and still measure an
  implementation that no longer exists. Quote `kornia` and `git_commit` together whenever a number
  is cited; the performance page and the llms digest both print the commit for this reason.
- **Supersede stale snapshots:** when a merged change alters the speed of ops a committed snapshot
  measures, re-measure that machine. When the hardware is not available, move the run to
  `benchmarks/results/superseded/<version>/` and add a row to that directory's README naming the
  change that superseded it. Leaving it published turns a kornia change into an apparent hardware
  difference, because the page invites column-by-column reading within one table.
- **Machine-readable export:** support `--json PATH` and write via `common.save_json` —
  strict-valid JSON (`NaN` → `null`), shape `{"metadata": {...}, "results": [...]}`.
- **Equal footing + honest regimes:** identical transform parameters and interpolation across
  backends; state each backend's regime (batched float tensor vs per-image uint8 loop) instead
  of pretending the columns are apples-to-apples. Publish losses alongside wins.
- **Every baseline at its best:** a baseline cell times the library's idiomatic fastest call,
  not a convenient one. Build its inputs outside the timed call, as kornia's tensor is: PIL gets
  ready-made `Image` objects, a mask gets the dtype the library's fast path takes (`bool` for
  scikit-image's `label`). Use the library's own idioms (`cv2.split`/`cv2.merge`, not strided
  slices plus `np.stack`), and fill a column wherever the library has the op (`cv2.PSNR`,
  albumentations' transforms). When the call does different work, such as albumentations' `CLAHE`
  equalizing only L in Lab, say so in the regime text. Each of these cost a published ratio 1.3x
  to 3.3x in #4723's first review.
- **One thread count for every library:** `setup_run` pins OpenCV to `--threads` as well as torch
  and records `opencv_num_threads`; the header prints both. OpenCV builds whose parallel backend
  ignores `setNumThreads` (GCD in the macOS wheels) keep every core, and the header says so in a
  `NOTE`. Suites with no OpenCV-backed column (`losses`, `feature/laf_ops.py`) leave OpenCV out.
  The `0.9.0rc1` result files predate the pin and have no `opencv_num_threads`: on the
  i7-14700K Linux runs, the OpenCV and albumentations cells used every core while torch used 4.
  The Apple runs are unaffected, because GCD ignores the pin.
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
    "opencv_num_threads": 4,
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
`get_perspective_transform`. `metadata.units` names the item for the whole file (`img/s`,
`items/s`, `LAFs/s`); the docs page labels its tables from it. `metadata.kornia_module` is the
checkout-relative path kornia was imported from (`outside-checkout` otherwise), and
`metadata.load` holds the aggregate load snapshot. A row whose timings are `null` carries an
`error`: the exception name when the backend or its `torch.compile` warmup raised, or
`unavailable` when the library is missing or has no counterpart for that op.

## Shared output format

All flagships build on the same helpers in [`common.py`](common.py), so their output lines up:

- **Command line** (`add_flagship_args`): `--batches`, `--size`, `--device`, `--dtype`,
  `--threads`, `--compile`, `--ops`, `--skip-compile-ops`, `--min-run-time`, `--json`,
  `--contribute`, `--machine-slug`. `--ops` rejects names the suite does not have.
- **Header** (`start_run`), in this order: `# <suite> benchmark — commit … — platform`, the
  software stack, `# kornia source: …`, the CUDA device when there is one,
  `# device=…, dtype=…, threads=… (opencv …), size=… — throughput <units>`, one line per backend regime,
  the meaning of `-`, then one `# NOTE:` per unavailable library or eager-only op.
- **Tables** (`run_batch_sweep`): one per batch size, op names left, one right-aligned
  throughput column per backend, units at the end of the header row. `-` is a skipped cell,
  `✗` a call that raised; both are explained in the JSON `error` field.
- **Footer** (`finish_run`): `# results written to <path>` for `--json`, and the canonical
  path plus the `git add` line for `--contribute`.

## Adding a new benchmark

1. Start from the smallest flagship (`morphology/flagship.py`): it puts the checkout root and
   `benchmarks/` on `sys.path` (`benchmarks/` is not a package), then uses `add_flagship_args`,
   `setup_run`, `start_run`, `KorniaRows`, `run_batch_sweep` and `finish_run`. Do not hand-roll
   the header, the compile warmup or the JSON writing; that is how suites drifted apart.
2. Pick a representative handful of the module's ops, not every export, and list them in the
   module docstring with each baseline's counterpart.
3. Baselines run **correctly and on equal footing** (same parameters, same interpolation, their
   native data regime) — a misconfigured baseline is worse than no baseline.
4. Missing optional libraries must degrade to a skip note, never a crash.
5. Document the regimes in the module docstring; keep the honest framing.

The filters flagship includes optional scikit-image baselines. To exercise every
current baseline, install the benchmark-only dependencies with
`uv pip install --upgrade --prerelease allow scikit-image kornia-rs`; this currently
selects scikit-image 0.26 and kornia-rs 0.1.15rc5. Its module docstring lists
differences in padding, kernel support, normalization, and clipping; empty cells
indicate an unavailable dependency or a missing native counterpart.
The kornia-rs adapters detect APIs individually: stable 0.1.14 supplies Gaussian
and box blur; 0.1.15rc5 also supplies median, Sobel, and grayscale bilateral.
The latter has a separate row because its Python API only accepts grayscale.
See [median parallelism notes](filters/median_parallelism.md) for the CPU/CUDA/MPS
implementation audit and related PyTorch issues and pull requests.

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

### Comparison artefacts

A base-versus-branch A/B or a cross-release comparison is not a release snapshot: it records a
revision that is by construction not the current tree, so it does not fit
`benchmarks/results/<kornia-version>/` and does not feed the docs performance page. Keep such
raw JSON next to the report that cites it, as `benchmarks/<area>/<name>_results/*.json`
(for example [`feature/graf_results/`](feature/graf_results/) and
[`feature/sift_runtime_results/`](feature/sift_runtime_results/)), named by the compared
revision or series. CI validates them with `results_schema.validate_artefact`: the same
envelope, metadata, privacy and row-type rules as a release snapshot, without the filename and
version-directory rules; `load` is optional there because a base revision's harness may predate
it. The report states what was measured, on which commits, with which command, in the style of
the sample-results sections below.

PR #4638 keeps its historical measurements in an [immutable archive](https://github.com/kornia/kornia/tree/efb04dbf9c85e4cf71625cc2467bd5243b0c803c/benchmarks),
with figures embedded in the PR description. Local reruns should write JSON outside the checkout.

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
14 threads (torch's default there; the script now defaults to `--threads 4` like every flagship, so pass
`--threads 14` to reproduce); LAF scales are stratified across all four pyramid levels that a 256×256 image at
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
used for the geometry suite. These runs predate prebuilt PIL inputs: the PIL cells include the
`Image.fromarray` conversion, so PIL is faster than shown.

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
