# CPU median and Laplacian performance investigation

Measured 2026-09-17 on Apple M1, macOS, PyTorch 2.14.0, Python 3.11,
OpenCV 4.11.0. Base: `3382b96e0`; after: the optimization working-tree snapshot on that
base, with exact implementation and harness SHA256 values in each JSON.
The measurements predate moving these changes onto main; the base contains
PR #4646, which changes box blur only and does not affect the measured filters.
CPU float32, RGB 256×256, 5×5 kernel, batches 1 and 8. These measurements characterize this machine and software stack.

## Four-thread A/B (milliseconds per batch; lower is better)

Both compilation/warmup and timed execution use four PyTorch threads;
OpenCV also receives `setNumThreads(4)`. Compilation is excluded.

| Operation | Batch | Mode | Before ms | After ms | Speedup |
| --- | ---: | --- | ---: | ---: | ---: |
| median_blur | 1 | (eager) | 25.38 | 16.39 | 1.55× |
| median_blur | 1 | (compiled) | 25.32 | 11.67 | 2.17× |
| laplacian | 1 | (eager) | 2.20 | 0.78 | 2.81× |
| laplacian | 1 | (compiled) | 8.19 | 0.27 | 30.67× |
| median_blur | 8 | (eager) | 302.11 | 146.83 | 2.06× |
| median_blur | 8 | (compiled) | 360.31 | 84.93 | 4.24× |
| laplacian | 8 | (eager) | 15.60 | 8.82 | 1.77× |
| laplacian | 8 | (compiled) | 61.90 | 2.20 | 28.18× |

Matched float32 OpenCV after-run medians: median blur 1.14/10.27 ms (B=1/8),
Laplacian `filter2D` 0.27/3.36 ms. Median remains about 10× slower even
compiled. Laplacian compiled reaches similar or better throughput than this
matched OpenCV reference. Native uint8 OpenCV is also reported separately.
Do not substitute `cv2.Laplacian` for the matched reference: it computes a
sum of second Sobel derivatives, not Kornia's kernel.

The earlier one-thread diagnostic in `base-cpu.json` / `after-cpu.json`
measured 5×5 B=1 eager median 82.36 → 20.53 ms and eager Laplacian
2.05 → 0.56 ms; compiled Laplacian 25.49 → 0.42 ms. That run preceded the
final autocast guard (no autocast was active). Its source fingerprints record
the intermediate snapshot. The final four-thread run above includes all guards.

## What was wrong

- Median blur extracted every neighborhood with 9 or 25 one-hot convolutions,
  then invoked ATen's general median reduction. Profiling isolated the reduction
  as the main cost; switching to unfold, contiguous last-axis median, sort,
  top-k, or kthvalue did not provide the needed eager gain. OpenCV's 3×3/5×5
  implementation uses a native SIMD sorting network even for float32;
  larger uint8 kernels use other algorithms.
- The new CPU inference path uses a pruned Batcher odd-even merge selection
  network (24/113 compare-exchanges for 3×3/5×5). It avoids patch materialization
  and general selection. It is still many PyTorch operations, so it does not
  match OpenCV's single SIMD loop. Backward and forward-mode AD, CPU autocast,
  other kernel shapes/sizes and accelerators keep the old path. A local invalid
  mask preserves the original convolution's NaN/Inf propagation.
- Kornia's Laplacian kernel is all ones with center `1 - kernel_area`.
  Separable neighborhood sums minus `kernel_area * center` compute it without
  dense convolution. The fast path is limited to CPU float32/float64 on builds
  without oneDNN; half types, autocast and accelerated backends retain the old
  implementation. Floating-point summation order changes within tested tolerance.
- Inductor makes the original convolution particularly bad on this M1 build:
  generated code converts NCHW into NHWC, invokes external grouped convolution,
  then converts back. With oneDNN unavailable, it executes three
  `aten::_slow_conv2d_forward` calls. The profiler attributed about 89% of
  compiled CPU time to this fallback. The new arithmetic can be fused.
  This is a compiler/backend/layout problem, not evidence that CPUs in general
  are unsuitable for compilation. Compiling the new median is still slower
  than eager at one thread, but wins at four threads in these runs.
- The shared benchmark helper silently used `Timer`'s default one thread,
  despite warmup and metadata using the configured count. It now passes the
  current PyTorch thread count explicitly. Historical result files are left
  unchanged and must not be read as measurements at their advertised count.

## Existing work

GitHub issue and PR searches on 2026-09-17 found no open performance fix for
`median_blur` or filter `laplacian`. [Roadmap PR #3803](https://github.com/kornia/kornia/pull/3803)
mentions median as a native/Triton opportunity. [PR #4646](https://github.com/kornia/kornia/pull/4646)
optimizes box blur; [issue #3927](https://github.com/kornia/kornia/issues/3927) and
[PR #3971](https://github.com/kornia/kornia/pull/3971) concern Laplacian pyramid
padding, a separate operation.

## Reproduction and validation

Run the identical new `median_laplacian.py` and corrected `common.py` in both
checkouts. Use an explicit interpreter and invoke as a module from the checkout
root; the script prints and checks `kornia.__file__` to defeat editable-install
cross-contamination.

```bash
python -m benchmarks.filters.median_laplacian --json result.json --kernels 5 --compile --threads 4
```

Inputs are seeded. Timings use `common.time_us` / `blocked_autorange`, median
and IQR, 0.5-second minimum runs. Work is forward inference including allocations;
OpenCV runs a per-image loop, Kornia a batch. The matched OpenCV median includes
zero-padding and cropping; the matched Laplacian uses Kornia's normalized kernel.
Matched float outputs and eager/compiled results are checked before timing.
Raw JSON stores runtime metadata, load, fingerprints, medians and IQRs.

Validation: CPU float16/bfloat16/float32/float64 focused suites (359 passed,
1 existing skip), 12 Inductor checks, ONNX export/runtime coverage, strict
export, numerical references, reverse and forward AD with ties, autocast dtype,
empty batch/channel and non-contiguous cases. Full pre-commit and type checks
pass (type checking reports an unrelated LightGlue deprecation warning).
CUDA and x86/oneDNN performance are unmeasured; their Laplacian paths are unchanged.

References: [Batcher 1968](https://doi.org/10.1145/1468075.1468121),
[OpenCV median implementation](https://github.com/opencv/opencv/blob/4.x/modules/imgproc/src/median_blur.simd.hpp),
[OpenCV filtering definitions](https://docs.opencv.org/4.x/d4/d86/group__imgproc__filter.html).
