# Oxford graf local-feature benchmark

Measured 2026-09-05. Base `601b5a4a` versus optimized implementation `e8e4ec0f`.
Original 640×800 images, 4,096 requested features, grayscale float32. The first two
sections use eager evaluation without autocast or compilation. Pyramid levels, scales, iteration budgets, descriptors,
weights, matching threshold, and RANSAC settings are identical between revisions.

## GPU speed and quality

RTX 4090, PyTorch 2.14.0+cu130, Python 3.11.14, Kornia 0.9.0rc1, WSL2 Linux.
Times include extraction of **both** images and SNN matching; they exclude image I/O
and RANSAC. Each table entry is the mean of the five pair-specific median times.
The JSON files retain each median and IQR. Quality is mean L1 corner reprojection
error in pixels against the supplied homography, using the existing benchmark metric.

| Pipeline | Base ms/pair | Optimized ms/pair | Speedup | Mean corner error, base → optimized |
| --- | ---: | ---: | ---: | ---: |
| SIFT | 215.9 | 114.5 | 1.89× | 223.640 → 223.640 |
| SIFT–AffNet–HardNet | 269.4 | 174.1 | 1.55× | 2.183 → 2.183 |
| KeyNet–HardNet | 131.5 | 126.9 | 1.04× | 5.638 → 5.638 |

KeyNet's roughly 4% GPU gain is smaller than the observed timing spread; it is not
strong evidence of an end-to-end GPU speedup. SIFT and SIFT–AffNet–HardNet show much
larger improvements. Match counts, RANSAC inlier counts, and corner errors are identical
between the two GPU runs for every pair and every pipeline.

| Pair | SIFT error px | SIFT–AffNet–HardNet error px | KeyNet–HardNet error px |
| --- | ---: | ---: | ---: |
| 1–2 | 1.381 | 1.325 | 2.029 |
| 1–3 | 1.506 | 1.216 | 1.428 |
| 1–4 | 2.777 | 0.860 | 2.531 |
| 1–5 | 507.132 | 2.624 | 3.700 |
| 1–6 | 605.405 | 4.891 | 18.500 |

Plain SIFT already fails badly on the two largest viewpoint changes, pairs 1–5 and
1–6. The optimization preserves those failures; the large mean error is not hidden
by averaging only successful pairs.

| Pipeline | Base peak extra CUDA MiB | Optimized peak extra CUDA MiB |
| --- | ---: | ---: |
| SIFT | 576.0 | 665.8 |
| SIFT–AffNet–HardNet | 1058.3 | 1058.6 |
| KeyNet–HardNet | 1057.7 | 1057.7 |

Memory is the maximum extra allocated tensor memory during an extraction-and-match
call, above already resident images/models/outputs; it excludes RANSAC and allocator
reserved memory. The SIFT speed improvement trades approximately 90 MiB of extra
peak memory for fewer refinement launches.

## CPU speed and quality

Intel Core i7-14700K, one CPU thread. The representative 1–2 pair is timed repeatedly;
all five pairs are still evaluated for quality. Other pairs have `null` timing fields
in the JSON, meaning deliberately untimed. Each cell below is median ± IQR, in seconds
for extracting and matching both images.

| Pipeline | Base s/pair | Optimized s/pair | Speedup |
| --- | ---: | ---: | ---: |
| SIFT | 3.861 ± 0.022 | 2.856 ± 0.051 | 1.35× |
| SIFT–AffNet–HardNet | 16.343 ± 0.145 | 12.714 ± 0.080 | 1.29× |
| KeyNet–HardNet | 16.368 ± 0.031 | 12.887 ± 0.113 | 1.27× |

CPU match counts, inlier counts, and reprojection errors are also identical between
revisions on every pair. CPU and GPU results differ because their adaptive refinement
backends and numerical kernels differ; compare each device against its own baseline.

| Pair | SIFT error px | SIFT–AffNet–HardNet error px | KeyNet–HardNet error px |
| --- | ---: | ---: | ---: |
| 1–2 | 1.247 | 1.343 | 1.263 |
| 1–3 | 1.435 | 1.227 | 1.679 |
| 1–4 | 2.611 | 1.548 | 1.833 |
| 1–5 | 507.037 | 1.792 | 3.658 |
| 1–6 | 605.427 | 11.658 | 8.828 |

## Selective compilation: CUDA

Base `601b5a4a` versus `fb66de1d` (the same optimized library implementation, with a
compiled benchmark mode). The existing factory compiles the scale-space pyramid,
response, and subpixel modules, or KeyNet response/NMS, using `dynamic=True` and
default Inductor settings. Descriptors, orientation, affine adaptation, matching,
and RANSAC stay eager: this is **not whole-pipeline compilation**. The workload,
hardware, feature budgets, and quality metric are the same as the eager pair benchmark.

| Pipeline | Base compiled ms/pair | Optimized compiled ms/pair | Speedup | Mean corner error, base → optimized |
| --- | ---: | ---: | ---: | ---: |
| SIFT | 94.64 | 71.23 | 1.33× | 223.654 → 223.654 |
| SIFT–AffNet–HardNet | 158.27 | 132.87 | 1.19× | 2.381 → 2.381 |
| KeyNet–HardNet | 106.01 | 105.08 | 1.01× | 5.638 → 5.638 |

Entries again average the five pair-specific medians. KeyNet's difference is timing
noise. Match counts, inliers, feature counts, and corner errors are exactly identical
between compiled revisions for all pairs. Compilation itself changes some scale-space
results relative to eager, so eager and compiled quality must be compared separately.
For example, SIFT–AffNet–HardNet's mean error is 2.381 px compiled versus 2.183 px eager.

Each revision started in a separate process with a fresh `TORCHINDUCTOR_CACHE_DIR`.
Initial calls are excluded from warmup and steady-state timing, and recorded in JSON.
SIFT's first extraction-and-match call took 43.42 s on the base and 43.38 s optimized.
KeyNet's first call took 15.25 s and 16.84 s. SIFT–AffNet–HardNet ran after SIFT and
reused its already compiled detector graphs: its 0.233/0.209 s initial calls are
**not independent cold-compilation measurements**. Initial latency includes useful
execution and initialization as well as compilation, not compiler time alone.

Maximum extra CUDA allocation: SIFT 566.9 → 690.1 MiB; SIFT–AffNet–HardNet
1058.6 → 1058.6 MiB; KeyNet–HardNet 1057.7 → 1057.7 MiB. All five per-pair
medians/IQRs and initial-call latencies are retained in the raw files.

```bash
.venv/bin/python -m benchmarks.feature.local_features --seq /data/graf --device cuda --compile --json graf-compiled.json
```

The additional CPU pair-benchmark sweep was interrupted to prioritize the requested
historical batching chart. It is not presented as a completed CPU compiled comparison.

## Implementation

- Accumulate each gradient pixel into its two orientation bins with `scatter_add_`,
  replacing 36 complete patch scans. Half-precision histograms accumulate in float32.
- Batch positive and negative response refinement for the built-in subpixel modules.
  Keep NMS neighborhoods separate, and retain separate calls for custom modules and
  refiners with a candidate cap. The number of refinement iterations is unchanged.
- Gather only selected coordinates before merging positive/negative results, avoiding
  a dense three-coordinate merge over the whole octave.
- Use channels-last float32 activations inside KeyNet's narrow convolution block on
  CPU/CUDA, and inside HardNet on CPU. Parameter layouts and checkpoint keys stay intact.

## Reproduction

Data: [Oxford affine graf archive](https://www.robots.ox.ac.uk/~vgg/research/affine/det_eval_files/graf.tar.gz),
from the [Oxford affine evaluation dataset](https://www.robots.ox.ac.uk/~vgg/research/affine/).
Archive SHA-256: `999871b945ee968a00a0d5f9af957d1382fb9dae1511cdee9553366817b53b5b`.
Every input image and homography hash is also recorded in the JSON metadata.

```bash
.venv/bin/python -m benchmarks.feature.local_features --seq /data/graf --device cuda --json graf-cuda.json
.venv/bin/python -m benchmarks.feature.local_features --seq /data/graf --device cpu --timing-pairs 2 --json graf-cpu.json
```

For the base comparison, run the same harness using `runpy.run_path` from the base
worktree root with the primary checkout's explicit interpreter. Confirm that the
printed `kornia.__file__` points into that base worktree, not the editable primary
checkout. Final benchmark processes ran sequentially, without concurrent tests.

The scale-space pipelines use `ScalePyramid(3, 1.6, 32, double_image=True)` with its
three extra levels, DoG minima/maxima, `AdaptiveQuadInterp3d` defaults, and 32-pixel
gradient orientation patches. SIFT uses `SIFTDescriptor(32, rootsift=True)`;
SIFT–AffNet–HardNet adds pretrained AffNet before orientation and pretrained HardNet
afterwards. KeyNet–HardNet uses the public preset with pretrained OriNet.
SNN ratio is 0.85. Homography RANSAC uses threshold 2.0, max_iter 10, batch_size 8196,
confidence 0.9999, and seed 3407. Model construction uses PyTorch seed 0.

The harness uses `benchmarks/common.py` timing and JSON helpers. It expands the timing
budget to at least five warmup-call durations, so a slow CPU call does not become a
single measured sample. CPU thread count is one, matching the shared Timer. Matmul TF32 is disabled;
cuDNN retains its default TF32 setting, recorded in the metadata.

## Verification

- Full `pixi run pre-commit-all` passed; `pixi run typecheck` passed with the existing
  `lightglue.py` AMP deprecation warning.
- Focused CPU/CUDA float32/float64 tests: 391 passed, 35 skipped, two cases of the
  existing half-versus-float32 test deselected. A subsequent final detector run passed
  all 292 selected tests, with 12 skips.
- CPU float16 and bfloat16 known-failure-profile runs: 137 passed and 11 skipped each.
- CPU/CUDA compilation tests for KeyNet and gradient orientation: four passed.
  Compiled joint min/max refinement also matched eager output in a direct check.
- Independent review checked numerical behavior, extension points, checkpoint
  compatibility, memory tradeoffs, and MPS portability of the added tests.

The broader initial test run found the existing
`TestOrientationHalfPrecisionIsFinite::test_half_precision_matches_float32[cuda]`
failure. An explicit base-worktree reproduction produced the same float32 and
float16 outputs as the optimized tree (one orientation differs by about 2.66 radians
between precisions). No tolerance or known-failure manifest was changed.

## Raw results

- [Base CUDA](graf_results/base-cuda.json)
- [Optimized CUDA](graf_results/optimized-cuda.json)
- [Base CPU](graf_results/base-cpu.json)
- [Optimized CPU](graf_results/optimized-cpu.json)
- [Base selectively compiled CUDA](graf_results/base-compiled-cuda.json)
- [Optimized selectively compiled CUDA](graf_results/optimized-compiled-cuda.json)

These are local benchmark results rather than release-wide performance guarantees.
