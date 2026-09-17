# SIFT follow-up: fewer copies and bounded sparse buffers

This pass reviews PR #4638 at `bc7909746f96bf3e9acb61fd368dc95ce0bca37f`, **after** the optimizations in the [previous report](sift_scale_space.md). It measures the full public `SIFTFeatureScaleSpace(4096, descriptor_backend="pyramid")` call. The feature budget, six Gaussian levels, 19×19/41×41 integration support, one dominant orientation and top-K policy are preserved.

## Runtime

Apple M1, macOS 26.5.1, Python 3.11.14, PyTorch 2.14.0, float32 RootSIFT, batch one, one CPU thread. Oxford Graf images 1–6 are 640×800. All runs return 4,096 valid features per image. Runtime excludes loading, matching, RANSAC and profiling. Values average six warmed per-image medians; measurements use `benchmarks.common.time_us`, at least five warm-call durations, and synchronization inside MPS timing.

| Device | PR head ms/image | Follow-up ms/image | Speedup |
| --- | ---: | ---: | ---: |
| CPU | 821.86 | 674.91 | 1.22× |
| MPS | 677.95 | 628.79 | 1.08× |

| Graf image | CPU before | CPU after | MPS before | MPS after |
| --- | ---: | ---: | ---: | ---: |
| 1 | 801.64 ± 14.42 | 726.12 ± 18.83 | 684.41 ± 16.67 | 633.15 ± 3.51 |
| 2 | 798.18 ± 2.52 | 665.66 ± 5.16 | 683.20 ± 0.55 | 632.73 ± 9.63 |
| 3 | 797.21 ± 17.59 | 682.10 ± 32.91 | 666.83 ± 16.30 | 630.48 ± 13.13 |
| 4 | 834.64 ± 6.10 | 654.55 ± 6.68 | 666.49 ± 17.41 | 613.07 ± 12.28 |
| 5 | 830.63 ± 12.58 | 657.82 ± 16.75 | 683.57 ± 8.59 | 630.02 ± 13.96 |
| 6 | 868.86 ± 16.66 | 663.23 ± 15.30 | 683.19 ± 8.86 | 633.33 ± 12.71 |

Cells are median ± IQR in milliseconds; IQR describes timing spread, not a confidence interval. Other applications were left untouched; aggregate load is recorded in every raw result. No CUDA or compiled-speed claim is made.

## CPU tensor memory

A **separate**, untimed inference forward records PyTorch CPU allocator events. Peak live bytes sum allocations minus frees starting at forward entry; total allocated bytes sum positive allocation events, including repeated allocations. These metrics exclude resident inputs/model tensors, Python objects, RSS and native-library workspaces outside PyTorch's allocator. Total allocated bytes are an allocation-volume proxy, **not measured DRAM traffic**. MiB means 2²⁰ bytes.

| Workload | Peak MiB before / after | Peak reduction | Total allocated MiB before / after |
| --- | ---: | ---: | ---: |
| Graf 640×800, 4096 features | 198.60 / 158.68 | 20.1% | 2767.68 / 2437.62 |
| checkerboard-512 | 347.97 / 95.62 | 72.5% | 629.37 / 545.19 |
| graf-small | 32.09 / 25.77 | 19.7% | 355.97 / 315.70 |
| graf-small-batch2 | 48.56 / 35.91 | 26.1% | 650.60 / 578.53 |

The Graf memory row uses the maximum peak across six images and the mean allocated volume. Other rows use one forward each. Small Graf inputs are images 1 and 2 resized to 192×240, requesting and returning 512 features each. The 512×512 checkerboard requests 4096 and correctly returns zero before and after.

| Additional CPU probe | Before ms | After ms | Speedup |
| --- | ---: | ---: | ---: |
| checkerboard-512 | 214.13 ± 3.70 | 180.66 ± 4.06 | 1.19× |
| graf-small | 94.53 ± 3.76 | 75.87 ± 2.17 | 1.25× |
| graf-small-batch2 | 180.98 ± 3.67 | 142.96 ± 4.07 | 1.27× |

## What changed

- CPU inference builds each Gaussian octave in its final volume. The final separable filtering pass writes directly into that volume. Large levels use row tiles to improve cache reuse. Autograd/compiled construction retains the functional path; accelerators retain the faster stacked construction.
- Central differences write channel-major gradient atlases directly, avoiding the convolution work buffer and full gradient transpose copy. Training uses differentiable tensor operations.
- CPU inference reuses pixel-coordinate storage for normalized sampling grids. Componentwise gradient transforms avoid tiny strided matrix-multiplication copies per feature. MPS retains functional grid construction: strided in-place writes regressed its public runtime.
- Eight spatial comparisons reject diagonal ties before gathering. Strict-neighbour screening and refinement use bounded chunks (16,384 candidates on CPU, 65,536 on accelerators), preserving candidate order. Repeated-scale extrema therefore cannot cause an unbounded 27-value temporary.
- Eight-bin indices use bit masks. On MPS, separable spatial pooling replaces one dense 16-cell contraction. The CPU keeps the faster existing contraction.
- The generic `SIFTDescriptorFromPyramid` selects octaves before building them, skips all-invalid pyramids, and maps each selected LAF set with one clone.

Direct eight-vote scattering, sparse per-column histogram integration and CPU separable histogram pooling were rejected after slower measurements. No runtime dependency was added.

## Correctness

The reviewed head had four actionable findings, all fixed locally: checkerboard-triggered oversized sparse gathers; detached outputs for no-detection specialized calls; floating masks below float16's representable range remaining eligible after promotion; and detached generic descriptors for empty frame sets/batches. Tests cover those cases, dense candidate chunking, numerical/gradient equivalence of atlases and histograms, border behavior, Gaussian preallocation, and eager/compiled serialization. A final independent review found no remaining actionable issue.

**All recorded matching and RANSAC quality rows are identical before/after on both CPU and MPS.** The protocol is SNN ratio 0.8, forward ground-truth Euclidean error ≤3 px, and homography RANSAC seed 3407 with a 2 px threshold, 10 batches of 8196 hypotheses and confidence 0.9999. MPS RANSAC runs on CPU outside extraction timing. Graf informed development; it is not held out. Pair 1–6 still fails geometrically.

| Pair | CPU correct / matches | CPU corner L1 px | MPS correct / matches | MPS corner L1 px |
| --- | ---: | ---: | ---: | ---: |
| 1-2 | 1656 / 1840 | 2.035 | 1655 / 1839 | 1.265 |
| 1-3 | 687 / 1040 | 1.253 | 688 / 1040 | 2.096 |
| 1-4 | 166 / 341 | 1.824 | 167 / 342 | 1.729 |
| 1-5 | 32 / 173 | 5.637 | 33 / 174 | 3.457 |
| 1-6 | 3 / 137 | 444.140 | 3 / 136 | 396.751 |

Validation on the final implementation:

- CPU float32/float64 feature, integration, API-surface and benchmark checks: 454 passed, 24 skipped.
- CPU float16/bfloat16 focused checks: 121 passed, 5 skipped.
- MPS float32 focused checks: 56 passed, 11 skipped.
- Minimum-supported PyTorch 2.5.1 across all four CPU dtypes: 227 passed, 17 skipped. Native-padding skips apply to direct private helpers; public half inputs are promoted and covered.
- Inductor pyramid/refinement/fullgraph filter checks: 3 passed.
- Full pre-commit and type checking passed.

CUDA, large full-resolution batches, held-out quality, training-memory usage and end-to-end compiled performance remain unverified.

## Reproduction and provenance

The [raw result files](sift_memory_results/) retain versions, source/input hashes, medians, IQRs and every quality row. Before library sources are the exact reviewed head; the new benchmark harnesses are copied into that worktree. Both before and after metadata may therefore say `bc7909746-dirty`: library SHA256 values distinguish the revisions. After hashes match the final library files. CPU timing and allocation profiles are stored separately.

Run from each measured checkout root, using the same explicit interpreter. Both scripts print `sys.executable` and `kornia.__file__` and enforce the expected checkout. Copy the two harness files into the before checkout before invoking them there.

```bash
/path/to/python -m benchmarks.feature.sift_scale_space \
  --seq /path/to/graf --expected-checkout "$PWD" --methods pyramid \
  --device cpu --json /tmp/sift-cpu.json
# Repeat with --device mps. Profile CPU allocations outside timing:
/path/to/python -m benchmarks.feature.sift_scale_space \
  --seq /path/to/graf --expected-checkout "$PWD" --methods pyramid \
  --device cpu --profile-memory --quality-only --json /tmp/sift-memory.json
/path/to/python -m benchmarks.feature.sift_memory \
  --seq /path/to/graf --expected-checkout "$PWD" --json /tmp/sift-cases.json
```
