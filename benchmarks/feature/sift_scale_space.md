# Dedicated scale-space SIFT on Oxford graf

`SIFTFeatureScaleSpace(descriptor_backend="pyramid")` remains a sparse local
feature pipeline. It uses the same DoG detector, feature budget, masks, responses,
and frame centres as the default patch path. It retains the detector's actual
Gaussian pyramid and octave/layer provenance through feature selection. The
nearest refined Gaussian layer supplies gradients; the continuously refined scale
sets the integration support. No descriptor-owned pyramid is built.

Gradients are computed once per used Gaussian layer and sampled directly for
orientation and description. Used layers are batched into a gradient atlas per
octave, with sampling clamped within each layer to prevent boundary leakage.
Equivalent dense angular weights and matrix multiplication accumulate descriptor
votes, reducing GPU scatter overhead. The SIFT-specific implementation uses a 36-bin
orientation histogram, a 4×4×8 descriptor, Gaussian weighting, trilinear spatial
and angular voting, clipping, and RootSIFT normalization. Orientation support is
4.5 sigma with a 1.5-sigma Gaussian; descriptor cell width is 3 sigma with a
6-sigma Gaussian. These constants follow
[OpenCV SIFT](https://github.com/opencv/opencv/blob/4.x/modules/features2d/src/sift.simd.hpp).
Fixed 19×19/41×41 quadrature grids and bilinear gradient sampling differ from
OpenCV's integer-pixel integration. One dominant orientation is retained per
feature. The rotated spatial grid and pulled-back gradient directions both account
for orientation. Magnitude and angle are evaluated after gradient interpolation,
avoiding interpolation across the angular wraparound. This is distinct from the
[generic DenseSIFT-histogram backend](dense_sift.md).

## Protocol

Original Oxford graf 640×800 PPM images 1–6, Pillow grayscale float32, batch one,
4,096 requested detections, RootSIFT, no affine adaptation or compilation. CPU uses
one thread on Apple M1; MPS is explicitly synchronized. CUDA is unavailable.
The public feature module's entire forward pass is timed, including detection,
pyramid construction, orientation, and description. Image loading, matching, and
RANSAC are excluded. Each image has a warmed median and IQR from `time_us`, with
at least max(1 second, five warm-call durations) of repeated timing.

All five pairs use SNN ratio 0.8. A correct match has forward ground-truth transfer
Euclidean error at most 3 px. Homography RANSAC uses seed 3407, 2 px inlier threshold,
10 iterations, batch size 8196, confidence 0.9999, and default local refinement.
Corner error is mean L1 transfer error against the supplied homography over all
four image corners. MPS extraction/matching uses CPU RANSAC. Large errors are
retained: consensus count alone does not establish a correct homography.

The pre-PR baseline is `5be74dc9f`. Both baseline and branch use the same benchmark
harness, run as a module from the measured checkout with an explicit interpreter
and an asserted `kornia.__file__`. Raw JSON records versions, machine/load details,
input and source hashes, feature counts, per-image medians/IQRs, and all quality
results. The branch retains the default patch path for an additional same-revision
comparison. Results are algorithm alternatives, not numerically equivalent
implementations.

## Reproduction

```bash
.venv/bin/python -m benchmarks.feature.sift_scale_space \
  --seq /path/to/graf --expected-checkout "$PWD" --device cpu \
  --json /tmp/sift-scale-space-cpu.json
```

Use `--device mps` for MPS, `--methods patch` in the base checkout, and
`--quality-only` to skip repeated timing. Copy this harness and `dense_sift.py`
(the common RANSAC scoring helper) into the base checkout before running.

## Results

Mean of the six per-image medians, milliseconds per image (lower is better):

| Device | Base patch (`5be74dc9f`) | Branch patch | Specialized pyramid | Speedup |
| --- | ---: | ---: | ---: | ---: |
| CPU, one thread | 2155.72 | 2159.27 | 1553.24 | 1.39× vs base |
| MPS | not measured | 679.72 | 767.88 | 0.89× vs branch patch |

The specialized path improves CPU throughput. MPS remains slower than patch
extraction, despite batching gradients and histogram integration. The opt-in
backend is not a universal speedup. The base and branch patch paths have identical
CPU match and RANSAC records.

### CPU matching and homography recovery

| Pair | Patch correct / matches | Pyramid correct / matches | Patch precision | Pyramid precision | Patch / pyramid RANSAC inliers | Patch / pyramid corner L1 (px) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1-2 | 1467/1651 | 1706/1919 | 88.86% | 88.90% | 1340 / 1554 | 0.98 / 0.88 |
| 1-3 | 481/706 | 690/1005 | 68.13% | 68.66% | 420 / 617 | 1.58 / 1.20 |
| 1-4 | 118/223 | 177/365 | 52.91% | 48.49% | 92 / 152 | 1.87 / 2.13 |
| 1-5 | 12/101 | 21/148 | 11.88% | 14.19% | 18 / 22 | 472.69 / 482.11 |
| 1-6 | 1/61 | 3/124 | 1.64% | 2.42% | 12 / 12 | 617.19 / 4605.24 |

More correct matches do not guarantee higher precision or successful geometry.
The specialized path has lower precision on 1–4; both paths fail homography
recovery on the hardest viewpoint pairs 1–5 and 1–6. The large corner errors are
part of the result and are not removed from the comparison. Graf was used during
development, so these are not held-out quality results.

### MPS matching and homography recovery

| Pair | Patch correct / matches | Pyramid correct / matches | Patch precision | Pyramid precision | Patch / pyramid RANSAC inliers | Patch / pyramid corner L1 (px) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1-2 | 1467/1650 | 1706/1919 | 88.91% | 88.90% | 1355 / 1555 | 1.30 / 0.88 |
| 1-3 | 481/706 | 690/1005 | 68.13% | 68.66% | 420 / 617 | 1.58 / 1.20 |
| 1-4 | 119/224 | 178/366 | 53.12% | 48.63% | 94 / 149 | 3.44 / 3.15 |
| 1-5 | 12/101 | 22/149 | 11.88% | 14.77% | 18 / 21 | 472.69 / 472.96 |
| 1-6 | 1/61 | 3/124 | 1.64% | 2.42% | 12 / 12 | 617.19 / 4605.24 |

Raw results: [base-cpu.json](sift_scale_space_results/base-cpu.json), [shared-cpu.json](sift_scale_space_results/shared-cpu.json), [shared-mps.json](sift_scale_space_results/shared-mps.json).

Measured code revision: `8c55c89b6` (CPU) and `8c55c89b6-dirty` (MPS); source SHA-256 records distinguish any documentation-only working-tree changes. CUDA and older supported PyTorch versions were not tested.
