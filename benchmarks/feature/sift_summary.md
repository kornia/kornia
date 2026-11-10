# SIFT pipeline benchmark

[`sift_scale_space.py`](sift_scale_space.py) compares the public patch and pyramid
`SIFTFeatureScaleSpace` backends on an Oxford-format six-image sequence, including
matching and homography quality. It supports CPU, CUDA and MPS, explicit thread counts,
optional OpenCV SIFT, quality-only runs and separate CPU tensor-allocation profiling.
The default library backend remains `patch`.

## Usage

Run from each measured checkout root with the same Pixi-selected interpreter:

```bash
python -m benchmarks.feature.sift_scale_space \
  --seq /path/to/graf --expected-checkout "$PWD" \
  --device cpu --threads 1 --methods patch pyramid --json /tmp/sift-cpu.json
# Use --device cuda or --device mps for accelerators.
# Use --methods opencv --device cpu for the optional native uint8 OpenCV baseline.
# --quality-only skips timing; --profile-memory profiles CPU allocations separately.
```

For revision comparisons, run the same script and `benchmarks/common.py` in both
checkouts and reverse revision order. The script verifies the imported checkout and
records versions, source/input hashes, feature counts, timing medians/IQRs and quality.
CPU profiling measures PyTorch tensor allocations, not RSS or DRAM traffic.
Keep generated JSON outside the repository.

## Recorded device results

These are optimization stages **within PR #4638**, not comparisons with main or a release.
Each row has its own measured baseline. Ratios are before / after; below 1× means slower.
Do not multiply ratios from separate measurement sessions into a cumulative speedup.

| Backend / optimization stage | Hardware / execution | Before → after (ms/image) | Speedup | Library revisions |
| --- | --- | ---: | ---: | --- |
| Pyramid / initial pass | Apple M1 CPU, 1 thread | 1201.12 → 794.80 | 1.51× | `3172b5f1` → `bc790974` |
| Pyramid / initial pass | Apple M1 MPS | 622.61 → 587.22 | 1.06× | `3172b5f1` → `bc790974` |
| Pyramid / memory pass | Apple M1 CPU, 1 thread | 821.86 → 674.91 | 1.22× | `bc790974` → `7547be2c` |
| Pyramid / memory pass | Apple M1 MPS | 677.95 → 628.79 | 1.08× | `bc790974` → `7547be2c` |
| Pyramid / initial + memory passes | Intel i7-14700K CPU, 1 thread | 1236.77 → 510.95 | 2.42× | `3172b5f1` → `7547be2c` |
| Pyramid / initial + memory passes | Intel i7-14700K CPU, 14 threads | 424.80 → 241.44 | 1.76× | `3172b5f1` → `7547be2c` |
| Pyramid / initial + memory passes | NVIDIA RTX 4090 CUDA | 120.77 → 102.49 | 1.18× | `3172b5f1` → `7547be2c` |
| Pyramid / CUDA pass | Intel i7-14700K CPU, 1 thread | 499.73 → 495.76 | 1.01× | `7547be2c` → `40bcf122` |
| Pyramid / CUDA pass | Intel i7-14700K CPU, 14 threads | 234.76 → 240.87 | 0.97× | `7547be2c` → `40bcf122` |
| Pyramid / CUDA pass | NVIDIA RTX 4090 CUDA | 105.50 → 78.99 | 1.34× | `7547be2c` → `40bcf122` |
| Default patch / shared interpolation | Intel i7-14700K CPU, 1 thread | 1450.40 → 1629.86 | 0.890× | `40bcf122` → `fa3d732c` |
| Default patch / shared interpolation | Intel i7-14700K CPU, 14 threads | 466.52 → 478.94 | 0.974× | `40bcf122` → `fa3d732c` |
| Default patch / shared interpolation | NVIDIA RTX 4090 CUDA | 59.50 → 57.03 | 1.043× | `40bcf122` → `fa3d732c` |

Graf images 1–6, 640×800 grayscale, batch one, FP32 RootSIFT, 4,096 returned features/image,
eager full forward including pyramid construction and allocations. Timings exclude I/O,
transfers, matching, RANSAC and profiling. Values average six warmed per-image medians;
Intel/CUDA uses two reversed-order runs, Apple one run per revision. IQR describes spread,
not confidence. Apple: M1, macOS 26.5.1, PyTorch 2.14.0. Intel/NVIDIA: i7-14700K / RTX 4090,
WSL2, PyTorch 2.14.0+cu130, TF32 disabled, CUDA host one thread. Both use Python 3.11.14.

### Review follow-up: descriptor octave selection

`SIFTDescriptorFromPyramid` binned gradients and pooled descriptor histograms at
**every** octave below the highest one any LAF selected, even when no LAF sampled
them. Building the maps only for occupied octaves leaves the descriptors bitwise
unchanged on CPU and costs nothing when every octave is occupied. 640×800 image,
64 frames, one forward, `torch.utils.benchmark` medians, same process per revision,
`99e92b78` → review fixes:

| Frame octaves | i7-14700K, 1 thread | i7-14700K, 14 threads | RTX 4090 |
| --- | ---: | ---: | ---: |
| all octave 0 (nothing skipped) | 538.5 → 528.5 ms (1.02×) | 114.1 → 115.5 ms (0.99×) | 7.58 → 7.57 ms (1.00×) |
| all octave 1 | 658.3 → 118.6 ms (5.5×) | 146.6 → 36.8 ms (4.0×) | 12.05 → 6.29 ms (1.9×) |
| all octave 2 | 693.0 → 33.4 ms (20.7×) | 155.4 → 12.7 ms (12.2×) | 14.77 → 6.51 ms (2.3×) |
| all octave 3 | 688.5 → 14.9 ms (46×) | 157.8 → 6.5 ms (24×) | 18.58 → 6.83 ms (2.7×) |
| mixed scales (every octave occupied) | 690.7 → 693.9 ms (1.00×) | 166.9 → 160.4 ms (1.04×) | 22.47 → 22.79 ms (0.99×) |

The octave-2 case also drops its largest CPU allocation from 382 to 18 MiB and CUDA
peak from 314 to 29 MiB. A full DoG detector spreads its detections over every
octave, so `SIFTFeature(descriptor_backend="pyramid")` end to end is unchanged
(RTX 4090 34.3 → 34.4 ms; 14 threads 332 → 320 ms, both within spread). The win
applies to externally supplied or scale-filtered frames.

The same session re-ran the table above's harness across the review fixes. Graf
quality is identical for both backends, and no timing moved outside its spread:
pyramid 69.4 → 70.8 ms CUDA, 487 → 486 ms at one thread, 245.7 → 248.9 ms at 14
threads; patch 55.15 → 55.16 ms CUDA. Making the pyramid's Gaussian kernels
float64 references rather than float32 has no measurable cost: interleaved in one
process, `_SIFTScalePyramid` measures 2576 µs with float64 buffers and 2577 µs with
float32 buffers on CUDA, 132.9 ms and 134.3 ms at one CPU thread.

CUDA pyramid batch two additionally measured **140.19 → 97.41 ms/batch (1.44×)** in one run
per revision. Apple memory changes reduced peak live CPU tensor allocations by **20% on
Graf and 73% on checkerboard**. CUDA pyramid peak extra tensor allocation remained ~238 MiB.
The last shared-interpolation pass retains CPU arithmetic; its measured CPU losses remain
in the table and establish no CPU speedup.

## Quality and limits

Default patch SIFT remains faster on CUDA (~57 versus ~79 ms in separate latest sessions).
Pyramid finds more correct Graf matches and recovers pair 1–5; **both fail pair 1–6**.
Matching uses SNN ratio 0.8, a 3 px correct-match threshold, and homography RANSAC with
2 px threshold and seed 3407; corner error is mean L1 discrepancy against ground truth.
All quality rows are unchanged by the CUDA/shared-interpolation passes. The first CPU
pass changed correct-match counts by at most one, with both better and worse corner errors.

Graf informed development and is not held out. Native FP16/BF16 forwards were checked,
but no half-precision speedup is established. Larger batches and end-to-end compiled
performance are unverified. MPS was not rerun for the CUDA follow-ups. These are archived
measurements, not new measurements of the merged PR head.

Figures live in the [PR description](https://github.com/kornia/kornia/pull/4638).
Historical raw measurements, per-image IQRs, controls, detailed reports and one-off scripts
remain available in the [archive](https://github.com/kornia/kornia/tree/efb04dbf9c85e4cf71625cc2467bd5243b0c803c/benchmarks).
