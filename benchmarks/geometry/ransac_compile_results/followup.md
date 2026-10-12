# PR #5117 follow-up: correctness and CPU speed

Compared the original PR head `eedc9996b` with the revised implementation `82443d5af`.
The revisions include the same benchmark harness; its hash is recorded in every result.

On Apple M1 the large explicit homography batch is **2.19× faster** with confidence stopping and
unchanged true-inlier recall. Equal-budget default cases are within 4% of the previous throughput;
these small differences are not evidence of a broad speedup. On M1 a fully exhausted large
batch is about 5% slower after splitting; on an Intel i7-14700K it is **2× faster**, so the split
also raises scoring throughput there (see [Intel CPU and CUDA](#intel-cpu-and-cuda)). Default F7
is about 6% slower over the extended 16-seed set (6–11% on the i7). The private random stream
changes which samples lead to confidence stopping, so default timings include sampling/stopping
variance; equal-budget timings help separate that effect from computation throughput. This is a
targeted improvement, not a universal speedup. On CUDA the change is neutral except for
homographies and essential matrices, about 6% and 3% slower when both revisions run in one process.

## Changes measured

- Use a private generator in a compiled sampler primitive, avoiding global RNG save/restore
  and races between seeded calls and unrelated random draws. No execution lock serializes calls.
- Split compiled CPU sampling batches whose candidate-by-correspondence matrix would exceed
  `2**22` entries. For H with N=5000 and requested batch 8192, the maximum possible matrix
  drops from 40,960,000 to 4,190,000 entries per batch. This is an allocation bound, not an RSS
  measurement. Extra boundaries permit earlier confidence stopping; confidence=1 retains
  the entire sample budget and exposes the throughput tradeoff.
- Protect excluded rows before singular projective/Sampson divisions and mask the resulting
  residuals/Jacobians. Remove an unnecessary homography target concatenation.
- Distinguish thread count/default dtype/default device in artifact keys, disable autocast
  consistently, fall back to tracing on artifact guard rejection, and publish artifacts atomically.

## Method

Apple M1, macOS 26.5.1 arm64, PyTorch 2.14.0, Python 3.11.14, CPU float32, four intra-op threads.
Both revisions used the same interpreter and preloaded synthetic scenes, 50% true inliers,
0.5-pixel noise, and a 1.5-pixel threshold (camera-normalized for E). N=2000 uses one fixed
planar/two-view scene per model; N=5000 uses a separate fixed planar scene.

Timed public `RANSAC.forward`, including staging, scoring, local optimization and final
refinement. Constructor and first-call compilation are excluded. The shared benchmark
helpers warm the CPU and use `torch.utils.benchmark.Timer.blocked_autorange`; raw rows contain
per-seed medians and IQRs. Runs were serial, with test/compilation workers stopped during
timing. Artifact mode was disabled for both revisions (`KORNIA_RANSAC_AOT=0`) to keep the
comparison independent of the original cache compatibility bug. First calls warm compilation
before any timing. Source location was verified and source digests recorded.

The table reports the **median of the per-seed latency medians**, not a pooled-call median.
Default, fixed-budget and large-batch cases use seeds 0,1,2 and a one-second minimum timing
window per cell. The supplemental default-F7 case uses seeds 0–15 and 0.5 seconds per cell;
it was added to investigate the initial three-seed F7 regression, and those original results
are retained. Default confidence is 0.999 and budget 20480. Fixed-budget cases use confidence=1
and the stated sample budget. Default batches use `auto`; large cases request batch 8192.

| Case | Model | Original compiled ms | Revised compiled ms | Original/revised |
| --- | --- | ---: | ---: | ---: |
| Default, N=2000 | H | 7.91 | 7.54 | 1.05× |
| Default, N=2000 | F7 | 17.01 | 19.11 | 0.89× |
| Default, N=2000 | F8 | 14.92 | 15.09 | 0.99× |
| Default, N=2000 | E | 18.13 | 17.60 | 1.03× |
| 2048 samples, N=2000 | H | 11.50 | 11.08 | 1.04× |
| 2048 samples, N=2000 | F7 | 24.95 | 24.76 | 1.01× |
| 2048 samples, N=2000 | F8 | 15.90 | 15.41 | 1.03× |
| 2048 samples, N=2000 | E | 96.89 | 95.14 | 1.02× |
| Batch 8192, N=5000 | H | 37.73 | 17.26 | 2.19× |
| 8192 samples/batch, N=5000 | H | 39.27 | 41.25 | 0.95× |
| Default F7, 16 seeds | F7 | 18.54 | 19.61 | 0.95× |

Some large-batch and supplemental F7 timing rows have wide IQRs; consult the raw data before
interpreting small changes. These results cover one CPU and controlled scenes, not real-world
pose accuracy, other platforms, CUDA throughput, cold compilation, or concurrent throughput.

## Intel CPU and CUDA

Intel Core i7-14700K (WSL2) and NVIDIA RTX 4090, PyTorch 2.14.0+cu130, Python 3.11.14, float32,
four intra-op threads. Same harness, cases and seeds as above, with `KORNIA_RANSAC_AOT=0`, timing
only the compiled backend. Each case ran twice per revision in opposite orders (original then
revised, then revised then original); both rounds are shown as `round 1 / round 2`.

| Case | Model | Original compiled ms | Revised compiled ms | Original/revised |
| --- | --- | ---: | ---: | ---: |
| Default, N=2000 | H | 4.11 / 4.15 | 4.18 / 4.11 | 0.98× / 1.01× |
| Default, N=2000 | F7 | 6.51 / 6.75 | 8.40 / 8.31 | 0.77× / 0.81× |
| Default, N=2000 | F8 | 5.89 / 5.95 | 6.44 / 6.28 | 0.91× / 0.95× |
| Default, N=2000 | E | 8.24 / 7.86 | 8.58 / 8.53 | 0.96× / 0.92× |
| 2048 samples, N=2000 | H | 5.00 / 5.17 | 4.99 / 5.17 | 1.00× / 1.00× |
| 2048 samples, N=2000 | F7 | 9.38 / 9.64 | 9.73 / 10.03 | 0.96× / 0.96× |
| 2048 samples, N=2000 | F8 | 6.20 / 6.72 | 6.24 / 6.32 | 0.99× / 1.06× |
| 2048 samples, N=2000 | E | 44.24 / 45.58 | 44.23 / 45.67 | 1.00× / 1.00× |
| Batch 8192, N=5000 | H | 29.80 / 28.94 | 8.97 / 8.90 | 3.32× / 3.25× |
| 8192 samples/batch, N=5000 | H | 29.78 / 29.61 | 14.78 / 15.42 | 2.02× / 1.92× |
| Default F7, 16 seeds | F7 | 6.79 / 7.48 | 7.66 / 7.97 | 0.89× / 0.94× |

The default F7 seeds range from 5.9 to 10.5 ms depending on the batch at which confidence stops,
against a within-seed IQR of about 0.2 ms; the equal-budget F7 cost is 4%.

CUDA, N=2000 (the CPU batch split does not apply on CUDA):

| Case | Model | Inliers | Original compiled ms | Revised compiled ms | Original/revised |
| --- | --- | ---: | ---: | ---: | ---: |
| Default | H | 50% | 8.72 / 8.35 | 10.03 / 9.74 | 0.87× / 0.86× |
| Default | H | 20% | 9.69 / 8.96 | 11.45 / 11.19 | 0.85× / 0.80× |
| Default | F7 | 50% | 7.19 / 7.22 | 7.05 / 7.08 | 1.02× / 1.02× |
| Default | F7 | 20% | 13.32 / 12.46 | 12.54 / 12.75 | 1.06× / 0.98× |
| Default | F8 | 50% | 5.74 / 5.61 | 5.74 / 5.57 | 1.00× / 1.01× |
| Default | F8 | 20% | 8.79 / 8.74 | 8.30 / 8.56 | 1.06× / 1.02× |
| Default | E | 50% | 7.49 / 7.38 | 7.87 / 7.62 | 0.95× / 0.97× |
| Default | E | 20% | 239.27 / 235.56 | 239.87 / 244.96 | 1.00× / 0.96× |
| 2048 samples | H | 50% | 7.99 / 8.31 | 9.50 / 9.37 | 0.84× / 0.89× |
| 2048 samples | F8 | 50% | 5.82 / 5.48 | 5.51 / 5.31 | 1.06× / 1.03× |
| 2048 samples | E | 50% | 50.39 / 48.76 | 51.25 / 48.25 | 0.98× / 1.01× |

Equal-budget F7 on CUDA varied up to 2× for the same seed between rounds and is omitted. Every
revised homography seed at 50% inliers is slower in both rounds (9.7–10.1 ms against 8.3–8.8 ms)
with the same 992 inliers.

Separate processes exaggerate these CUDA differences on this host, whose host-side work moves
between performance and efficiency cores. Loading both revisions into one process and alternating
them over six rounds (default settings, N=2000, 50% inliers, all four programs compiled), the
revised implementation took 0.94× (H), 0.96× (F7), 1.03× (F8) and 0.97× (E) of the original's
throughput. The homography difference comes from masking the full residual and Jacobian of
excluded rows in the refinement: a revision with the original `homography.py` was 4–13% faster.

On CUDA with 20% inliers, where 7- and 8-point sampling rarely draws an all-inlier sample within the
budget, 64 seeds gave a mean inlier recall of 0.843 ± 0.021 (original) and 0.841 ± 0.020 (revised,
with the SplitMix64 batch seeds that followed) for F7, against 0.850 ± 0.022 for eager; F8 gave
0.557 ± 0.030, 0.534 ± 0.027 and 0.509 ± 0.027. Three-seed differences at this ratio are sampling luck.

## Recovery

Every recorded call returned a model. True-inlier recall was unchanged across revisions:
H=0.992 at N=2000, H=0.990 at N=5000, and F7/F8/E=1.000. H had zero false positives. In the
extended F7 set, false positives ranged from 6–7 originally and 6–8 after the change, out of
1000 synthetic outliers. The per-seed support and false-positive counts are included in the
raw results; equal recall does not mean identical models or masks. These few fixed scenes
are a regression diagnostic and do not establish general geometric accuracy.

## Reproduce

Run from each checkout root with the same interpreter, using an identical copy of
`benchmarks/geometry/ransac_compile_synthetic.py` in the baseline checkout:

```bash
KORNIA_RANSAC_AOT=0 python benchmarks/geometry/ransac_compile_synthetic.py \
  --compile --sizes 2000 --ratios 0.5 --seeds 0,1,2 --threads 4 --min-run-time 1 \
  --json /tmp/default.json
```

For the fixed-budget case add `--confidence 1 --max-samples 2048`. For the large-batch case
replace `--sizes 2000` with `--ops H --sizes 5000 --sample-batch 8192`. For the large fixed-budget
case also add `--confidence 1 --max-samples 8192`. For the supplemental F7 case use
`--ops F7 --seeds 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 --min-run-time 0.5` with N=2000.

| Raw case | Original PR head | Revised implementation |
| --- | --- | --- |
| default | [base](https://github.com/kornia/kornia/blob/733729055be12f2281625be82f0b4df350e2fc50/benchmarks/geometry/ransac_compile_results/eedc9996b-default.json) | [revised](https://github.com/kornia/kornia/blob/733729055be12f2281625be82f0b4df350e2fc50/benchmarks/geometry/ransac_compile_results/82443d5af-default.json) |
| fixed-budget | [base](https://github.com/kornia/kornia/blob/733729055be12f2281625be82f0b4df350e2fc50/benchmarks/geometry/ransac_compile_results/eedc9996b-fixed-budget.json) | [revised](https://github.com/kornia/kornia/blob/733729055be12f2281625be82f0b4df350e2fc50/benchmarks/geometry/ransac_compile_results/82443d5af-fixed-budget.json) |
| large-batch | [base](https://github.com/kornia/kornia/blob/733729055be12f2281625be82f0b4df350e2fc50/benchmarks/geometry/ransac_compile_results/eedc9996b-large-batch.json) | [revised](https://github.com/kornia/kornia/blob/733729055be12f2281625be82f0b4df350e2fc50/benchmarks/geometry/ransac_compile_results/82443d5af-large-batch.json) |
| large-fixed-budget | [base](https://github.com/kornia/kornia/blob/733729055be12f2281625be82f0b4df350e2fc50/benchmarks/geometry/ransac_compile_results/eedc9996b-large-fixed-budget.json) | [revised](https://github.com/kornia/kornia/blob/733729055be12f2281625be82f0b4df350e2fc50/benchmarks/geometry/ransac_compile_results/82443d5af-large-fixed-budget.json) |
| f7-seeds | [base](https://github.com/kornia/kornia/blob/733729055be12f2281625be82f0b4df350e2fc50/benchmarks/geometry/ransac_compile_results/eedc9996b-f7-seeds.json) | [revised](https://github.com/kornia/kornia/blob/733729055be12f2281625be82f0b4df350e2fc50/benchmarks/geometry/ransac_compile_results/82443d5af-f7-seeds.json) |

## Validation

- CPU float32/float64 geometry suite: 1217 passed, 32 skipped, 10 non-strict XPASS.
- Compiled RANSAC CPU suite: 38 passed, including concurrent RNG isolation, full seed domain,
  ambient guards, inference/autocast, large batches and fresh-process artifact reuse.
- Focused CPU half precision: float16 19 passed/17 skipped; bfloat16 20 passed/16 skipped.
- PyTorch 2.5.1 refinement and essential backward regression checks: 93 passed.
- Full pre-commit checks and type checking passed. Benchmark artifacts validated against
  `benchmarks.results_schema.validate_artefact`.
- Independent review found no remaining actionable findings in the changed areas.
- CUDA compilation/runtime and GPU performance remain unverified locally.
