# Shared quadratic interpolation: CPU/CUDA A/B

For the consolidated device comparison and current figures, see the [SIFT benchmark summary](../feature/sift_summary.md).
Raw measurements are linked to commit `efb04dbf`; they are intentionally absent from the PR file diff.

The change groups finite differences and four Cramer determinants into shared tensor operations on CUDA float32/float64. CPU, MPS and reduced-precision inputs keep the scalar paths. It also reaches the default patch-descriptor `SIFTFeatureScaleSpace` pipeline. Public `SIFTDescriptor` pooling is unchanged; this report does not measure the multiresolution `SIFTFeature` pipeline.

Measured 2026-09-17 on an NVIDIA GeForce RTX 4090 and Intel Core i7-14700K (WSL2), torch 2.14.0+cu130, Python 3.11.14, eager float32 inference. Base is **`40bcf122`**; the optimized tree is that revision plus the shared-interpolation patch. Both record `-dirty` because the same benchmark harness changes were installed in both trees. Full source/input SHA256 hashes are in the raw metadata.
`spatial_soft_argmax.py` SHA256: base `1f4d5a487c25a774b9d3baadd05daae3f4670f92b527ad6d07450e9be54266f5`; optimized `ac9162bd9df935d93388e3ea7178b420b6e8f64aa6ceb23c37b6391b3faac577`.
The library source received formatting-only edits after measurement; its Python AST was verified identical. The raw hashes identify the measured formatting.

Two complete runs reverse revision order: r1 base→optimized, r2 optimized→base. Jobs ran serially. Seed 0 pins inputs; SIFT RANSAC uses seed 3407. `common.time_us` uses warmed, synchronized `blocked_autorange` medians/IQR, at least 1 second per case. `Timer.num_threads` explicitly honors 1/14 CPU threads (CUDA host: 1); metadata records the actual setting and system load. TF32 and compilation are disabled.

Run from each measured checkout root with its selected interpreter (the harness prints and checks the imported checkout):
```bash
python -m benchmarks.geometry.quad_interp --device cuda --threads 1 --json quad.json
python -m benchmarks.feature.sift_scale_space --seq /path/to/graf --expected-checkout "$PWD" --methods patch --device cuda --threads 1 --json sift.json
```
Repeat with `--device cpu --threads 1` and `--device cpu --threads 14`, then reverse revision order. Quad includes NMS and output allocation (`n_iters=5`, no strict-maxima bonus); volumes have B=C=1, D=5, H=W as named, with 128² sparse/flat cases. SIFT times the entire forward on each of six 800×640 graf images, with 4096 features and RootSIFT; image I/O, matching and RANSAC are excluded from timing.

| Raw artifacts | Run 1 | Run 2 |
|---|---|---|
| quad, CUDA / 1 thread | [base](https://github.com/kornia/kornia/blob/efb04dbf9c85e4cf71625cc2467bd5243b0c803c/benchmarks/geometry/quad_interp_results/quad-base-cuda-t1-r1.json) / [opt](https://github.com/kornia/kornia/blob/efb04dbf9c85e4cf71625cc2467bd5243b0c803c/benchmarks/geometry/quad_interp_results/quad-opt-cuda-t1-r1.json) | [base](https://github.com/kornia/kornia/blob/efb04dbf9c85e4cf71625cc2467bd5243b0c803c/benchmarks/geometry/quad_interp_results/quad-base-cuda-t1-r2.json) / [opt](https://github.com/kornia/kornia/blob/efb04dbf9c85e4cf71625cc2467bd5243b0c803c/benchmarks/geometry/quad_interp_results/quad-opt-cuda-t1-r2.json) |
| quad, CPU / 1 thread | [base](https://github.com/kornia/kornia/blob/efb04dbf9c85e4cf71625cc2467bd5243b0c803c/benchmarks/geometry/quad_interp_results/quad-base-cpu-t1-r1.json) / [opt](https://github.com/kornia/kornia/blob/efb04dbf9c85e4cf71625cc2467bd5243b0c803c/benchmarks/geometry/quad_interp_results/quad-opt-cpu-t1-r1.json) | [base](https://github.com/kornia/kornia/blob/efb04dbf9c85e4cf71625cc2467bd5243b0c803c/benchmarks/geometry/quad_interp_results/quad-base-cpu-t1-r2.json) / [opt](https://github.com/kornia/kornia/blob/efb04dbf9c85e4cf71625cc2467bd5243b0c803c/benchmarks/geometry/quad_interp_results/quad-opt-cpu-t1-r2.json) |
| quad, CPU / 14 threads | [base](https://github.com/kornia/kornia/blob/efb04dbf9c85e4cf71625cc2467bd5243b0c803c/benchmarks/geometry/quad_interp_results/quad-base-cpu-t14-r1.json) / [opt](https://github.com/kornia/kornia/blob/efb04dbf9c85e4cf71625cc2467bd5243b0c803c/benchmarks/geometry/quad_interp_results/quad-opt-cpu-t14-r1.json) | [base](https://github.com/kornia/kornia/blob/efb04dbf9c85e4cf71625cc2467bd5243b0c803c/benchmarks/geometry/quad_interp_results/quad-base-cpu-t14-r2.json) / [opt](https://github.com/kornia/kornia/blob/efb04dbf9c85e4cf71625cc2467bd5243b0c803c/benchmarks/geometry/quad_interp_results/quad-opt-cpu-t14-r2.json) |
| sift, CUDA / 1 thread | [base](https://github.com/kornia/kornia/blob/efb04dbf9c85e4cf71625cc2467bd5243b0c803c/benchmarks/geometry/quad_interp_results/sift-base-cuda-t1-r1.json) / [opt](https://github.com/kornia/kornia/blob/efb04dbf9c85e4cf71625cc2467bd5243b0c803c/benchmarks/geometry/quad_interp_results/sift-opt-cuda-t1-r1.json) | [base](https://github.com/kornia/kornia/blob/efb04dbf9c85e4cf71625cc2467bd5243b0c803c/benchmarks/geometry/quad_interp_results/sift-base-cuda-t1-r2.json) / [opt](https://github.com/kornia/kornia/blob/efb04dbf9c85e4cf71625cc2467bd5243b0c803c/benchmarks/geometry/quad_interp_results/sift-opt-cuda-t1-r2.json) |
| sift, CPU / 1 thread | [base](https://github.com/kornia/kornia/blob/efb04dbf9c85e4cf71625cc2467bd5243b0c803c/benchmarks/geometry/quad_interp_results/sift-base-cpu-t1-r1.json) / [opt](https://github.com/kornia/kornia/blob/efb04dbf9c85e4cf71625cc2467bd5243b0c803c/benchmarks/geometry/quad_interp_results/sift-opt-cpu-t1-r1.json) | [base](https://github.com/kornia/kornia/blob/efb04dbf9c85e4cf71625cc2467bd5243b0c803c/benchmarks/geometry/quad_interp_results/sift-base-cpu-t1-r2.json) / [opt](https://github.com/kornia/kornia/blob/efb04dbf9c85e4cf71625cc2467bd5243b0c803c/benchmarks/geometry/quad_interp_results/sift-opt-cpu-t1-r2.json) |
| sift, CPU / 14 threads | [base](https://github.com/kornia/kornia/blob/efb04dbf9c85e4cf71625cc2467bd5243b0c803c/benchmarks/geometry/quad_interp_results/sift-base-cpu-t14-r1.json) / [opt](https://github.com/kornia/kornia/blob/efb04dbf9c85e4cf71625cc2467bd5243b0c803c/benchmarks/geometry/quad_interp_results/sift-opt-cpu-t14-r1.json) | [base](https://github.com/kornia/kornia/blob/efb04dbf9c85e4cf71625cc2467bd5243b0c803c/benchmarks/geometry/quad_interp_results/sift-base-cpu-t14-r2.json) / [opt](https://github.com/kornia/kornia/blob/efb04dbf9c85e4cf71625cc2467bd5243b0c803c/benchmarks/geometry/quad_interp_results/sift-opt-cpu-t14-r2.json) |

## Public interpolation latency

Each cell is **base median (IQR) → optimized median (IQR), milliseconds**. Read both repetitions; these are per-case measurements, not a universal speedup.

| Device / threads | Case / function | Run 1, ms | Run 2, ms |
|---|---|---|---|
| CUDA / 1 thread | dense-32 / conv | 5.693 (1.559) → 5.281 (0.248) | 6.352 (1.838) → 5.356 (0.405) |
| CUDA / 1 thread | dense-32 / iterative | 8.171 (0.348) → 7.310 (2.406) | 8.493 (1.585) → 6.193 (0.172) |
| CUDA / 1 thread | dense-128 / conv | 5.572 (0.227) → 5.289 (1.107) | 5.721 (0.483) → 5.146 (0.295) |
| CUDA / 1 thread | dense-128 / iterative | 7.890 (0.380) → 5.936 (0.181) | 8.136 (0.265) → 6.271 (0.570) |
| CUDA / 1 thread | dense-320 / conv | 5.582 (0.738) → 5.647 (2.350) | 5.868 (0.178) → 5.303 (0.399) |
| CUDA / 1 thread | dense-320 / iterative | 7.986 (0.161) → 5.958 (1.393) | 8.158 (0.327) → 6.242 (1.441) |
| CUDA / 1 thread | sparse / conv | 5.719 (1.270) → 5.048 (0.309) | 5.748 (0.219) → 5.301 (0.395) |
| CUDA / 1 thread | sparse / iterative | 7.970 (4.111) → 6.533 (2.210) | 7.521 (0.653) → 5.897 (0.435) |
| CUDA / 1 thread | flat / conv | 2.052 (0.326) → 1.836 (0.431) | 1.780 (0.175) → 2.034 (0.045) |
| CUDA / 1 thread | flat / iterative | 1.771 (0.431) → 2.013 (0.478) | 1.779 (0.131) → 1.927 (0.136) |
| CPU / 1 thread | dense-32 / conv | 0.901 (0.026) → 0.900 (0.009) | 0.917 (0.031) → 0.909 (0.017) |
| CPU / 1 thread | dense-32 / iterative | 0.888 (0.011) → 0.885 (0.015) | 0.908 (0.022) → 0.908 (0.020) |
| CPU / 1 thread | dense-128 / conv | 7.176 (0.442) → 7.405 (0.395) | 7.654 (0.661) → 7.379 (0.319) |
| CPU / 1 thread | dense-128 / iterative | 2.490 (0.103) → 2.475 (0.104) | 2.478 (0.068) → 2.473 (0.117) |
| CPU / 1 thread | dense-320 / conv | 64.952 (2.143) → 67.754 (2.806) | 68.139 (2.755) → 67.670 (2.526) |
| CPU / 1 thread | dense-320 / iterative | 11.246 (0.250) → 11.183 (0.478) | 11.421 (0.295) → 11.224 (0.225) |
| CPU / 1 thread | sparse / conv | 1.365 (0.021) → 1.380 (0.027) | 1.368 (0.025) → 1.377 (0.004) |
| CPU / 1 thread | sparse / iterative | 1.532 (0.044) → 1.523 (0.051) | 1.550 (0.016) → 1.541 (0.020) |
| CPU / 1 thread | flat / conv | 1.272 (0.012) → 1.244 (0.023) | 1.243 (0.008) → 1.248 (0.008) |
| CPU / 1 thread | flat / iterative | 1.423 (0.033) → 1.390 (0.026) | 1.384 (0.015) → 1.388 (0.032) |
| CPU / 14 threads | dense-32 / conv | 1.134 (0.248) → 0.936 (0.073) | 1.079 (0.042) → 0.985 (0.097) |
| CPU / 14 threads | dense-32 / iterative | 0.888 (0.012) → 0.887 (0.027) | 0.901 (0.010) → 0.900 (0.037) |
| CPU / 14 threads | dense-128 / conv | 4.329 (0.895) → 4.379 (0.659) | 4.664 (0.581) → 4.614 (0.106) |
| CPU / 14 threads | dense-128 / iterative | 2.520 (0.038) → 2.073 (0.447) | 2.283 (0.405) → 2.305 (0.568) |
| CPU / 14 threads | dense-320 / conv | 19.533 (1.561) → 19.155 (0.356) | 20.318 (0.873) → 19.976 (2.212) |
| CPU / 14 threads | dense-320 / iterative | 4.933 (0.339) → 5.375 (0.656) | 5.431 (0.691) → 5.288 (0.687) |
| CPU / 14 threads | sparse / conv | 1.344 (0.398) → 1.318 (0.075) | 1.450 (0.243) → 1.364 (0.122) |
| CPU / 14 threads | sparse / iterative | 1.688 (0.046) → 1.679 (0.071) | 1.615 (0.162) → 1.624 (0.438) |
| CPU / 14 threads | flat / conv | 1.305 (0.085) → 1.274 (0.040) | 1.334 (0.128) → 1.305 (0.162) |
| CPU / 14 threads | flat / iterative | 1.534 (0.083) → 1.515 (0.167) | 1.584 (0.102) → 1.665 (0.168) |

Raw CPU losses are retained: CPU / 1 thread: 7/20 optimized medians were slower; largest observed increase 4.3% (dense-320/conv_quad_interp3d/r1); CPU / 14 threads: 5/20 optimized medians were slower; largest observed increase 9.0% (dense-320/iterative_quad_interp3d/r1). Counts report measured median increases, not significance tests; compare the IQR and reversed-order repeat. CPU retains its scalar computation path, so these changes do not establish a CPU optimization benefit.

## Default SIFTFeatureScaleSpace, patch descriptors

Summary is the arithmetic mean of **six per-image medians × two runs** (12 medians per revision), not a pooled timing median. Speedup is base mean / optimized mean.

| Device / threads | Base mean, ms | Optimized mean, ms | Speedup |
|---|---:|---:|---:|
| CUDA / 1 thread | 59.500 | 57.032 | 1.043× |
| CPU / 1 thread | 1450.400 | 1629.855 | 0.890× |
| CPU / 14 threads | 466.520 | 478.942 | 0.974× |

| Device / threads | Graf image | Run 1, ms (IQR) | Run 2, ms (IQR) |
|---|---:|---|---|
| CUDA / 1 thread | 1 | 60.057 (13.839) → 55.783 (5.924) | 58.549 (4.103) → 55.940 (4.404) |
| CUDA / 1 thread | 2 | 60.906 (16.986) → 57.339 (8.779) | 58.186 (8.977) → 62.591 (13.427) |
| CUDA / 1 thread | 3 | 58.523 (12.626) → 55.882 (3.479) | 58.113 (13.709) → 58.386 (9.106) |
| CUDA / 1 thread | 4 | 58.361 (14.489) → 55.547 (10.016) | 59.571 (12.226) → 57.485 (16.468) |
| CUDA / 1 thread | 5 | 62.260 (16.516) → 58.196 (16.026) | 62.561 (13.073) → 55.999 (2.656) |
| CUDA / 1 thread | 6 | 55.994 (5.609) → 55.050 (3.332) | 60.922 (10.410) → 56.192 (6.245) |
| CPU / 1 thread | 1 | 1421.950 (27.113) → 1856.632 (20.938) | 1470.116 (25.555) → 1451.557 (31.982) |
| CPU / 1 thread | 2 | 1412.690 (12.233) → 1831.102 (51.453) | 1449.576 (6.607) → 1483.529 (44.930) |
| CPU / 1 thread | 3 | 1439.593 (27.196) → 1628.489 (44.179) | 1468.014 (45.427) → 1506.405 (33.625) |
| CPU / 1 thread | 4 | 1421.485 (12.725) → 1685.338 (22.606) | 1468.768 (15.755) → 1509.936 (13.146) |
| CPU / 1 thread | 5 | 1408.734 (28.285) → 1850.510 (22.688) | 1472.939 (33.607) → 1493.339 (19.509) |
| CPU / 1 thread | 6 | 1505.456 (12.908) → 1732.946 (59.292) | 1465.476 (30.712) → 1528.479 (43.009) |
| CPU / 14 threads | 1 | 427.346 (8.887) → 449.114 (20.831) | 480.894 (12.135) → 528.047 (11.979) |
| CPU / 14 threads | 2 | 437.735 (6.700) → 444.527 (7.355) | 534.999 (6.465) → 501.087 (10.052) |
| CPU / 14 threads | 3 | 426.260 (7.960) → 474.695 (16.956) | 508.416 (6.592) → 530.677 (6.080) |
| CPU / 14 threads | 4 | 449.608 (5.737) → 470.997 (7.698) | 516.772 (5.274) → 474.328 (11.839) |
| CPU / 14 threads | 5 | 443.164 (20.976) → 455.227 (8.593) | 481.507 (21.836) → 482.617 (16.502) |
| CPU / 14 threads | 6 | 435.555 (8.963) → 444.113 (8.738) | 455.979 (17.938) → 491.872 (5.041) |

## CUDA allocation

Separate forward, incremental peak allocated tensor memory above resident inputs, including returned outputs; MiB (2²⁰ bytes). This is neither reserved memory nor end-to-end SIFT peak memory.

| Case / function | Run 1 base → optimized, MiB | Run 2 base → optimized, MiB |
|---|---:|---:|
| dense-32 / conv | 0.823 → 0.961 | 0.823 → 0.961 |
| dense-32 / iterative | 0.149 → 0.149 | 0.149 → 0.149 |
| dense-128 / conv | 15.274 → 17.871 | 15.274 → 17.871 |
| dense-128 / iterative | 2.355 → 2.354 | 2.355 → 2.354 |
| dense-320 / conv | 99.110 → 116.021 | 99.110 → 116.021 |
| dense-320 / iterative | 14.910 → 14.952 | 14.910 → 14.952 |
| sparse / conv | 1.686 → 1.684 | 1.686 → 1.684 |
| sparse / iterative | 1.466 → 1.466 | 1.466 → 1.466 |
| flat / conv | 1.643 → 1.643 | 1.643 → 1.643 |
| flat / iterative | 1.466 → 1.466 | 1.466 → 1.466 |

## Same-process SIFT control

This follow-up holds the current default `SIFTFeatureScaleSpace` object and graf image 1 fixed, swaps only the actual base/optimized `conv_quad_interp3d` functions, and shuffles their order in each of three rounds (order seeds 42–44). Each timing uses at least 3 seconds. It isolates process/startup drift from the shared refinement change; it does not replace the separate-process six-image results or erase their reported losses. Full CPU output tuples are checked exactly equal. CUDA full-pipeline outputs are checked with atol=4e-5 and rtol=1e-5: orientation/descriptor accumulation varies slightly even when the same revision is repeated, so bitwise equality is not required for the full SIFT pipeline.

| Device / threads | Round | Base ms (IQR) → optimized ms (IQR) | CUDA peak base → optimized, MiB |
|---|---:|---|---:|
| [CUDA / 1 thread](https://github.com/kornia/kornia/blob/efb04dbf9c85e4cf71625cc2467bd5243b0c803c/benchmarks/geometry/quad_interp_results/control-cuda-t1.json) | 1 | 60.210 (13.177) → 54.922 (3.786) | 650.135 → 650.542 |
| [CUDA / 1 thread](https://github.com/kornia/kornia/blob/efb04dbf9c85e4cf71625cc2467bd5243b0c803c/benchmarks/geometry/quad_interp_results/control-cuda-t1.json) | 2 | 57.359 (8.891) → 55.243 (10.069) | 650.135 → 650.542 |
| [CUDA / 1 thread](https://github.com/kornia/kornia/blob/efb04dbf9c85e4cf71625cc2467bd5243b0c803c/benchmarks/geometry/quad_interp_results/control-cuda-t1.json) | 3 | 57.246 (11.736) → 57.200 (10.096) | 650.135 → 650.542 |
| [CPU / 1 thread](https://github.com/kornia/kornia/blob/efb04dbf9c85e4cf71625cc2467bd5243b0c803c/benchmarks/geometry/quad_interp_results/control-cpu-t1.json) | 1 | 1846.969 (11.796) → 1824.844 (16.541) | — |
| [CPU / 1 thread](https://github.com/kornia/kornia/blob/efb04dbf9c85e4cf71625cc2467bd5243b0c803c/benchmarks/geometry/quad_interp_results/control-cpu-t1.json) | 2 | 1871.556 (16.323) → 1808.286 (17.962) | — |
| [CPU / 1 thread](https://github.com/kornia/kornia/blob/efb04dbf9c85e4cf71625cc2467bd5243b0c803c/benchmarks/geometry/quad_interp_results/control-cpu-t1.json) | 3 | 1889.926 (27.459) → 1848.887 (18.384) | — |
| [CPU / 14 threads](https://github.com/kornia/kornia/blob/efb04dbf9c85e4cf71625cc2467bd5243b0c803c/benchmarks/geometry/quad_interp_results/control-cpu-t14.json) | 1 | 475.144 (21.725) → 443.949 (13.555) | — |
| [CPU / 14 threads](https://github.com/kornia/kornia/blob/efb04dbf9c85e4cf71625cc2467bd5243b0c803c/benchmarks/geometry/quad_interp_results/control-cpu-t14.json) | 2 | 440.840 (13.239) → 511.707 (24.111) | — |
| [CPU / 14 threads](https://github.com/kornia/kornia/blob/efb04dbf9c85e4cf71625cc2467bd5243b0c803c/benchmarks/geometry/quad_interp_results/control-cpu-t14.json) | 3 | 491.179 (10.728) → 537.949 (15.543) | — |

Control allocation includes the entire SIFT forward and returned outputs, measured separately above resident model/input tensors. The interpolation-only allocation table above has a narrower scope.

Maximum absolute base/optimized output differences, in return order (LAFs/responses/descriptors): CUDA / 1 thread: 2.86e-05/0/1.58e-05; CPU / 1 thread: 0/0/0; CPU / 14 threads: 0/0/0. A separate CUDA repeatability probe found LAF/descriptor maxima 2.86e-5/1.41e-5 across revisions, compared with 2.67e-5/1.58e-5 for two base calls and 1.91e-5/1.11e-5 for two optimized calls; responses were exact in all three comparisons. This qualifies full-pipeline output parity separately from exact interpolation-forward arithmetic.

## Correctness and scope

Saved public interpolation tensors also match bitwise for all 60 float32 case/device/thread/repetition comparisons.

All 30 paired quality rows (five image pairs × three device/thread settings × two runs) are exactly equal between revisions, including matches, correct matches, precision, transfer error, RANSAC inliers/status and corner error. Equality is checked within each device/run; it is not a claim that CPU and CUDA give identical results.

The refactor preserves forward arithmetic. Backward agreement is covered on conditioned inputs; near-singular cases can differ through gradient accumulation order, so this is not a claim of identical gradients. No training, float64, MPS, compilation, or multiresolution SIFT performance claim is made.

An exploratory descriptor-pooling alternative was rejected: fixed separable factors would ignore custom/trainable 2D pooling weights. General angle batching preserved that contract but gave mixed timings and raised CUDA memory; at 4096 patches the exploratory 32/41-pixel cases regressed about 18–19% with nearly double the allocation peak. Those short exploratory results are not part of the controlled interpolation matrix above. Production descriptor pooling remains unchanged.

Native float16/bfloat16 CPU and CUDA forward timings and correctness checks are in [the half-precision follow-up](quad_interp_half.md).

Validation: 727 CPU/CUDA float32/float64 tests passed (47 skipped); four CPU/CUDA Inductor checks passed. See the half report for the retained baseline failures. Full pre-commit and type checking passed; type checking retains the existing LightGlue deprecation warning.
