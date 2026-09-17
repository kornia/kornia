# Native half-precision SIFT and interpolation

For the consolidated device comparison and current figures, see the [SIFT benchmark summary](../feature/sift_summary.md).
Raw measurements are linked to commit `efb04dbf`; they are intentionally absent from the PR file diff.

Follow-up to [the float32 shared-interpolation benchmark](quad_interp.md), on the same Intel i7-14700K / RTX 4090, WSL2, PyTorch 2.14.0+cu130, Python 3.11.14. Base `40bcf122` versus the shared-interpolation change. One process per device/dtype/revision; one timing run per configuration, serialized without other tests or benchmarks. This is a compatibility and timing probe, not evidence of a new half-precision speedup.

The shared packed path dispatches only for CUDA float32/float64. Native float16/bfloat16 quadratic interpolation keeps the original arithmetic, including float16 Cramer promotion to float32. The specialized pyramid SIFT backend already promotes internal work for half inputs; its packed refinement was added before this comparison.

Run from each compared checkout root, with the same harness and selected interpreter:

```bash
python -m benchmarks.feature.sift_half --seq /path/to/graf --device cuda --dtype float16 --threads 1 --json half.json
```

Repeat with `bfloat16`, and with CPU at one and 14 threads. Native half input, no autocast, eager `inference_mode`, TF32 disabled, seed 0. Public interpolation uses a seeded 1×1×5×128×128 response volume, five iterations, and no maxima bonus. Both public `SIFTFeatureScaleSpace` descriptor backends process full Graf image 1 (640×800), requesting 4096 features. Timed forwards include detection/orientation/description; image I/O and output checks are excluded. Matching quality in half precision is not measured. Timings use warmed, synchronized `common.time_us`, at least one second, with the requested Timer thread count.

All 48 forwards returned finite outputs in their requested half dtype; all 24 SIFT forwards returned 4096 filled features. The 12 paired interpolation results and all six paired default patch SIFT results were bitwise equal between revisions. Pyramid SIFT outputs matched exactly on CPU; on CUDA LAFs and responses matched exactly, with maximum descriptor differences 0.0001220703125 (float16) and 0.00048828125 (bfloat16). The pyramid implementation is unchanged in this comparison.

Each cell is base → optimized **median (IQR), milliseconds**. Raw losses are retained; these single-pass numbers fluctuate despite unchanged native-half code paths. CPU/one-thread patch SIFT exceeds the one-second timing budget, so its zero IQR comes from a single timed observation and does not establish timing stability.

| Device / threads | Dtype | Public operation | Base ms (IQR) | Optimized ms (IQR) |
|---|---|---|---:|---:|
| cuda, 1 | float16 | conv_quad_interp3d | 6.777 (1.441) | 5.703 (0.325) |
| cuda, 1 | float16 | iterative_quad_interp3d | 9.327 (2.147) | 8.364 (0.461) |
| cuda, 1 | float16 | sift-patch | 55.888 (15.377) | 63.704 (11.770) |
| cuda, 1 | float16 | sift-pyramid | 72.848 (14.881) | 69.975 (8.777) |
| Raw JSON | float16 | cuda, 1 | [base](https://github.com/kornia/kornia/blob/efb04dbf9c85e4cf71625cc2467bd5243b0c803c/benchmarks/geometry/quad_interp_results/half-base-cuda-t1-float16.json) | [optimized](https://github.com/kornia/kornia/blob/efb04dbf9c85e4cf71625cc2467bd5243b0c803c/benchmarks/geometry/quad_interp_results/half-opt-cuda-t1-float16.json) |
| cpu, 1 | float16 | conv_quad_interp3d | 6.834 (0.227) | 6.757 (0.292) |
| cpu, 1 | float16 | iterative_quad_interp3d | 2.951 (0.094) | 2.924 (0.096) |
| cpu, 1 | float16 | sift-patch | 1979.873 (0.000) | 1988.228 (0.000) |
| cpu, 1 | float16 | sift-pyramid | 488.144 (3.761) | 486.000 (6.001) |
| Raw JSON | float16 | cpu, 1 | [base](https://github.com/kornia/kornia/blob/efb04dbf9c85e4cf71625cc2467bd5243b0c803c/benchmarks/geometry/quad_interp_results/half-base-cpu-t1-float16.json) | [optimized](https://github.com/kornia/kornia/blob/efb04dbf9c85e4cf71625cc2467bd5243b0c803c/benchmarks/geometry/quad_interp_results/half-opt-cpu-t1-float16.json) |
| cpu, 14 | float16 | conv_quad_interp3d | 4.209 (0.431) | 4.378 (0.788) |
| cpu, 14 | float16 | iterative_quad_interp3d | 2.933 (0.191) | 2.647 (0.064) |
| cpu, 14 | float16 | sift-patch | 779.356 (5.829) | 775.083 (4.900) |
| cpu, 14 | float16 | sift-pyramid | 228.507 (15.242) | 226.791 (16.863) |
| Raw JSON | float16 | cpu, 14 | [base](https://github.com/kornia/kornia/blob/efb04dbf9c85e4cf71625cc2467bd5243b0c803c/benchmarks/geometry/quad_interp_results/half-base-cpu-t14-float16.json) | [optimized](https://github.com/kornia/kornia/blob/efb04dbf9c85e4cf71625cc2467bd5243b0c803c/benchmarks/geometry/quad_interp_results/half-opt-cpu-t14-float16.json) |
| cuda, 1 | bfloat16 | conv_quad_interp3d | 5.521 (0.534) | 5.489 (0.990) |
| cuda, 1 | bfloat16 | iterative_quad_interp3d | 8.589 (2.034) | 7.771 (1.140) |
| cuda, 1 | bfloat16 | sift-patch | 53.804 (6.359) | 53.268 (9.098) |
| cuda, 1 | bfloat16 | sift-pyramid | 80.134 (9.731) | 69.991 (12.542) |
| Raw JSON | bfloat16 | cuda, 1 | [base](https://github.com/kornia/kornia/blob/efb04dbf9c85e4cf71625cc2467bd5243b0c803c/benchmarks/geometry/quad_interp_results/half-base-cuda-t1-bfloat16.json) | [optimized](https://github.com/kornia/kornia/blob/efb04dbf9c85e4cf71625cc2467bd5243b0c803c/benchmarks/geometry/quad_interp_results/half-opt-cuda-t1-bfloat16.json) |
| cpu, 1 | bfloat16 | conv_quad_interp3d | 6.251 (0.291) | 6.334 (0.482) |
| cpu, 1 | bfloat16 | iterative_quad_interp3d | 2.869 (0.116) | 2.854 (0.110) |
| cpu, 1 | bfloat16 | sift-patch | 2051.894 (0.000) | 2045.639 (0.000) |
| cpu, 1 | bfloat16 | sift-pyramid | 510.902 (1.673) | 495.720 (3.703) |
| Raw JSON | bfloat16 | cpu, 1 | [base](https://github.com/kornia/kornia/blob/efb04dbf9c85e4cf71625cc2467bd5243b0c803c/benchmarks/geometry/quad_interp_results/half-base-cpu-t1-bfloat16.json) | [optimized](https://github.com/kornia/kornia/blob/efb04dbf9c85e4cf71625cc2467bd5243b0c803c/benchmarks/geometry/quad_interp_results/half-opt-cpu-t1-bfloat16.json) |
| cpu, 14 | bfloat16 | conv_quad_interp3d | 3.921 (0.524) | 4.145 (0.803) |
| cpu, 14 | bfloat16 | iterative_quad_interp3d | 2.656 (0.352) | 2.633 (0.412) |
| cpu, 14 | bfloat16 | sift-patch | 719.003 (4.805) | 696.186 (0.514) |
| cpu, 14 | bfloat16 | sift-pyramid | 235.592 (3.436) | 241.252 (18.290) |
| Raw JSON | bfloat16 | cpu, 14 | [base](https://github.com/kornia/kornia/blob/efb04dbf9c85e4cf71625cc2467bd5243b0c803c/benchmarks/geometry/quad_interp_results/half-base-cpu-t14-bfloat16.json) | [optimized](https://github.com/kornia/kornia/blob/efb04dbf9c85e4cf71625cc2467bd5243b0c803c/benchmarks/geometry/quad_interp_results/half-opt-cpu-t14-bfloat16.json) |

CUDA peak allocations (including returned outputs, above resident input/model allocations) are in the raw rows. Default patch SIFT uses about 383 MiB for float16 and 420 MiB for bfloat16; pyramid SIFT uses about 238 MiB for both.

CPU focused tests with the repository known-failure profiles give 52 passed / 14 expected failures for float16 and 56 passed / 10 expected failures for bfloat16, identical on the base and changed revisions (new float32/float64-only tests skip for half). Known failures include CPU 3D pooling support and the old quadratic diagonal-reference tolerance; their manifests are unchanged.

The isolated CUDA half run passed 30 tests (24 skipped) and retains two failures in `TestConvQuadInterp3dModule.test_diag` (float16 and bfloat16); both reproduce on exact base `40bcf122`. They are existing hardcoded-reference tolerance failures, not newly passing coverage.
