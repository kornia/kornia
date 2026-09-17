# SIFT on Intel CPU and CUDA

For the consolidated device comparison and current figures, see the [SIFT benchmark summary](sift_summary.md).
Raw measurements are linked to commit `efb04dbf`; they are intentionally absent from the PR file diff.

PR #4638 measured on Intel Core i7-14700K + NVIDIA RTX 4090, WSL2, Python 3.11.14, PyTorch 2.14.0+cu130, CUDA runtime 13.0, oneDNN enabled. Measurements: 2026-09-17.

## Do the earlier optimizations transfer?

Yes for the specialized pipeline. The table compares exact library revisions `3172b5f1` (reviewed implementation), `bc790974` (first optimization) and `7547be2c` (memory follow-up). These are three stages **inside the PR**, not the existing default patch backend. Each value averages six per-image medians over two full runs with reversed revision order, in milliseconds.

| Device | Reviewed | First pass | Memory follow-up | Overall speedup |
| --- | ---: | ---: | ---: | ---: |
| cpu, 1 CPU thread(s) | 1236.77 | 665.43 | 510.95 | 2.42× |
| cpu, 14 CPU thread(s) | 424.80 | 279.32 | 241.44 | 1.76× |
| cuda, 1 CPU thread(s) | 120.77 | 103.52 | 102.49 | 1.18× |

The latest memory follow-up improved Intel further, but its additional CUDA gain was within timing variation. The default patch backend measured 1508.99 ms (CPU one thread), 454.63 ms (CPU 14 threads), and 62.08 ms (CUDA). Thus the pre-follow-up pyramid backend was faster on Intel but 1.65× slower than the default on CUDA. Both return 4096 features; their algorithms and matching quality differ.

## CUDA-specific follow-up

The new CUDA float32/float64 refiner packs finite differences and four Cramer determinants into shared tensor operations, maintaining the original arithmetic order. It runs five iterations without per-iteration host scalar checks, masking dead samples before arithmetic. CPU, MPS, and direct reduced-precision refinement retain their prior path (public half inputs are already promoted before refinement). CUDA also reuses separable descriptor histogram pooling.

Feature budgets, strict extrema, masks, integration support, and 1024-feature accelerator descriptor chunks are unchanged. Larger chunks were rejected: a 4096-feature chunk raised peak extra CUDA tensor allocation on Graf image 1 from about 238 MiB to 420 MiB for only a modest additional runtime win. The retained packed/separable path measured about 238 MiB as well. This is PyTorch peak allocated tensor memory in a separate forward, excluding resident input/model/output references, not total device memory or reserved allocator capacity.

Fresh A/B against exact library head `7547be2c`, two full runs in opposite revision order:

| Device | Previous head ms | CUDA follow-up ms | Speedup |
| --- | ---: | ---: | ---: |
| cpu, 1 CPU thread(s) | 499.73 | 495.76 | 1.01× |
| cpu, 14 CPU thread(s) | 234.76 | 240.87 | 0.97× |
| cuda, 1 CPU thread(s) | 105.50 | 78.99 | 1.34× |

CPU retains its original numerical implementation; the 14-thread separate-process comparison was about 3% slower and should not be read as a CPU improvement. The default remains the patch backend. This follow-up makes no end-to-end compiled performance claim.


Same-process control (Graf image 1, three shuffled rounds, median of round medians):

| Device | Previous ms | Follow-up ms | Speedup |
| --- | ---: | ---: | ---: |
| cpu, 1 CPU thread(s) | 487.74 | 489.33 | 1.00× |
| cpu, 14 CPU thread(s) | 239.00 | 236.46 | 1.01× |
| cuda, 1 CPU thread(s) | 102.57 | 71.75 | 1.43× |

Additional public-forward probes (one run per revision, CPU one thread; full-size batch two requests 4096 features per image, small inputs are 192×240 with 512 features per image):

| Device / case | Previous ms | Follow-up ms | Speedup |
| --- | ---: | ---: | ---: |
| cpu, small | 58.69 | 56.26 | 1.04× |
| cpu, small-batch2 | 103.16 | 114.53 | 0.90× |
| cpu, checkerboard | 154.74 | 157.39 | 0.98× |
| cpu, graf-batch2 | 1012.12 | 1029.60 | 0.98× |
| cuda, small | 65.08 | 51.33 | 1.27× |
| cuda, small-batch2 | 79.74 | 68.63 | 1.16× |
| cuda, checkerboard | 10.08 | 10.22 | 0.99× |
| cuda, graf-batch2 | 140.19 | 97.41 | 1.44× |

Checkerboards return zero detections on both revisions; all other probes return their requested budgets. The one-off CPU small-batch probe is slower and is retained here; CPU arithmetic remains unchanged, and these additional probes were not repeated. CUDA full-size batch two benefits by about 1.44×. A separate profiler forward counts 8533 → 5644 kernel launches; profiler timings themselves are not used as benchmark timings.

## Gaussian distinction

The generic Gaussian/ScalePyramid results in [PR #4641](https://github.com/kornia/kornia/pull/4641#issuecomment-5718439595) do not determine the result for this specialized grayscale SIFT pyramid, which doubles Graf to 1280×1600 and builds six cached-kernel levels per octave. In a same-process Graf image 1 control, replacing only the `7547be2c` specialized pyramid with the actual older convolution pyramid slowed the full pipeline from 511.85 to 756.63 ms at one thread and from 244.53 to 339.67 ms at 14 threads. Those are medians of three shuffled round medians; other SIFT components are identical. This supports the specialized workload, not a generic Gaussian dispatch change.

## Matching quality

SNN ratio 0.8; correct matches have forward GT Euclidean error ≤3 px. RANSAC homography uses seed 3407, 2 px inlier threshold, 10 batches of 8196 hypotheses, confidence 0.9999, and runs on the extraction device outside timing. Corner error is mean L1 GT discrepancy at four corners. Graf informed development and is not held out.

| CUDA pair | Patch correct / matches | Previous pyramid correct / matches | Follow-up correct / matches | Corner L1 px, patch / previous / follow-up |
| --- | ---: | ---: | ---: | ---: |
| 1-2 | 1467 / 1650 | 1654 / 1838 | 1654 / 1838 | 1.413 / 1.298 / 1.298 |
| 1-3 | 481 / 706 | 686 / 1038 | 686 / 1038 | 1.863 / 1.168 / 1.168 |
| 1-4 | 119 / 224 | 167 / 342 | 167 / 342 | 2.604 / 1.751 / 1.751 |
| 1-5 | 12 / 101 | 33 / 174 | 33 / 174 | 473.180 / 3.254 / 3.254 |
| 1-6 | 1 / 61 | 3 / 136 | 3 / 136 | 615.071 / 458.764 / 458.764 |

The new backend finds more correct Graf matches than the patch alternative, and recovers pair 1–5 where the patch homography fails. Pair 1–6 still fails. All CPU and CUDA rows, including failures and both runs, are retained in the raw results. CPU quality changed by at most one correct match during the first historical optimization; the later memory pass left it unchanged.

## Validation

- Focused CPU/CUDA float32/float64 feature tests: 362 passed, 53 skipped.
- Focused CPU float16/bfloat16: 135 passed, 13 skipped. CUDA public half paths, isolated subprocesses: 12 passed.
- Inductor component checks on CPU/CUDA float32: 4 passed, 2 skipped.
- Benchmark methodology helper tests: 23 passed, including explicit timed thread control/restoration.
- The final combined feature and benchmark-helper run passed 385 tests, with 53 skipped. Full `pixi run pre-commit-all`, `pixi run typecheck`, artifact schema validation and diff checks passed.
- Independent spec and correctness reviews found no actionable issue. Packed/refined outputs are pinned against the previous CPU reference, with gradients, random candidates, quadratic fits, singular systems and all-dead NaN samples.
- The existing CUDA descriptor gradcheck failed even at `3172b5f1` because its nondeterminism tolerance was zero. Repeated float64 backward differed by at most 4.16e-16 before optimization and 5.55e-16 at `7547be2c`; both passed with `nondet_tol=1e-12`. The test now permits that CUDA-only accumulation noise.
- MPS hardware, older PyTorch CUDA versions, held-out quality, and end-to-end compiled speed were not retested in this follow-up. CPU/MPS implementations remain unchanged.

## Method and reproduction

All timings use the full public `SIFTFeatureScaleSpace(4096, rootsift=True, descriptor_backend="pyramid").eval()` in inference mode, Oxford Graf images 1–6 (640×800 grayscale float32), batch one, including allocations and pyramid construction. Loading, transfers, matching, RANSAC and profiling are excluded. `benchmarks.common.time_us` provides warmed synchronized medians/IQRs with at least max(1 second, five initial-call durations). Seeds are fixed. Both cuDNN and matmul TF32 are disabled. CPU thread counts are explicit inside Timer; merely calling `torch.set_num_threads` would not override Timer’s default of one. Timing jobs, tests and profiling are sequential.

Run from each measured worktree root with the same explicit Pixi-selected interpreter. Copy the current `benchmarks/common.py` and `benchmarks/feature/sift_scale_space.py` into the base worktree first. This changes only the harness; library sources stay at exact `7547be2c`. Both result metadata may name `7547be2c-dirty`: the implementation SHA256 hashes distinguish the measured libraries. After hashes match the source files accompanying this report. The script prints and verifies `kornia.__file__` and records source/input hashes and aggregate load.

```bash
/path/to/python -m benchmarks.feature.sift_scale_space \
  --seq /path/to/graf --expected-checkout "$PWD" --methods pyramid \
  --device cuda --threads 1 --json /tmp/sift-cuda.json
# Repeat on CPU with --threads 1 and --threads 14; reverse revision order.
# Use --methods patch for the default-backend comparison.
```

Raw measurements: [`sift_device_results/`](https://github.com/kornia/kornia/tree/efb04dbf9c85e4cf71625cc2467bd5243b0c803c/benchmarks/feature/sift_device_results/). `before`, `middle`, and `head` name the three historical stages above; `base-final` and `opt` name the fresh A/B. `r1`/`r2` are separate full runs. Control files hold same-process component comparisons. The initial `extra` files cover resized Graf, batch two and checkerboard inputs before/after the historical optimizations. Per-image IQRs describe timing spread, not confidence intervals.
