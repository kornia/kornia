# Specialized top-K scale-space SIFT on Oxford graf

`SIFTFeatureScaleSpace(descriptor_backend="pyramid")` is a sparse local feature
pipeline with a dedicated SIFT detector and descriptor. It ranks valid strict DoG
extrema by absolute refined response and retains the top K: **no contrast
threshold and no edge rejection**, including during candidate generation. Masks
and zero padding preserve the requested output budget. The default patch path
still uses the generic detector and is unchanged.

The implementation lives in `kornia/feature/sift/`: `scale_space.py` owns the
specialized Gaussian pyramid, detector, and descriptor together; `pyramid.py`
owns the reusable descriptor for arbitrary LAFs. Public feature presets remain
in `integrated.py`, and established patch descriptors remain in `siftdesc.py`.

The optimized path builds one six-level Gaussian pyramid with three intervals per
octave, cached separable kernels, precise image doubling, and integer octave
decimation. Strict spatial NMS on the three searchable layers precedes sparse
cross-scale comparisons and iterative 3-D quadratic refinement. This avoids
refining the full response volume. Converged integer layers and continuously
refined scales survive top-K selection and supply descriptor provenance and
support sizes respectively. No descriptor-owned pyramid is built.

Gradients are computed once per used Gaussian layer and shared by orientation and
description. Used layers form an atlas per octave, with sampling clamped inside
each layer. A 36-bin orientation histogram and 4×4×8 descriptor use Gaussian
weights, rotated spatial sampling, trilinear voting, clipping, and RootSIFT.
Orientation support is 4.5 sigma with a 1.5-sigma Gaussian; descriptor cell width
is 3 sigma with a 6-sigma Gaussian, following
[OpenCV SIFT](https://github.com/opencv/opencv/blob/4.x/modules/features2d/src/sift.simd.hpp).
Fixed 19×19/41×41 sampling grids differ from integer-pixel OpenCV integration.
One dominant orientation is retained per detection. Magnitude and angle are
computed after gradient interpolation to avoid angular wraparound artifacts.
This is separate from the [generic DenseSIFT-histogram backend](dense_sift.md).

## Protocol and reproduction

Oxford graf 640×800 PPM images 1–6, Pillow grayscale float32, batch one, 4,096
requested detections, RootSIFT, no affine adaptation or compilation. Apple M1,
macOS 26.5.1, Python 3.11.14, PyTorch 2.14.0. CPU uses one thread; MPS is explicitly
synchronized. The entire public feature forward is timed, including detection,
pyramid construction, orientation, and description. Loading, matching, and RANSAC
are excluded. Each image uses a warmed median and IQR from `time_us`, with at least
max(1 second, five warm-call durations) of repeated timing.

Every pair uses SNN ratio 0.8; correct matches have forward ground-truth transfer
Euclidean error at most 3 px. Homography RANSAC uses seed 3407, 2 px threshold,
10 iterations, batch size 8196, confidence 0.9999, and default local refinement.
Corner error is mean L1 transfer error against ground truth over four corners.
MPS extraction/matching uses CPU RANSAC. Consensus count alone does not establish
a correct homography.

The before/after comparison measures the previous shared-pyramid implementation
at `deba6b45d` and the specialized detector at `4a8c7472f`, using the same protocol
and public API. Each run asserts the imported checkout. Raw JSON contains source
and input hashes, versions, load metadata, per-image timings, feature counts, and
quality metrics; it is kept outside the repository as requested. The subsequent
package consolidation only moves these implementations; class ASTs were checked
to be identical apart from docstrings.

```bash
.venv/bin/python -m benchmarks.feature.sift_scale_space \
  --seq /path/to/graf --expected-checkout "$PWD" --methods pyramid \
  --device cpu --json /tmp/sift-scale-space-cpu.json
```

Run from each measured checkout with an explicit interpreter. Use `--device mps`
for MPS, `--methods patch` for the default pipeline, or `--quality-only` to omit
timing. Both measured revisions already contain the harness.

## End-to-end speed

Mean of six per-image medians, milliseconds per image (lower is better):

| Device | Previous shared pyramid | Specialized detector | Speedup |
| --- | ---: | ---: | ---: |
| CPU | 1486.40 | 1119.63 | 1.33× |
| MPS | 648.26 | 586.31 | 1.11× |

Per-image median / IQR in milliseconds; all runs return 4,096 filled features:

| Image | CPU before | CPU after | MPS before | MPS after |
| --- | ---: | ---: | ---: | ---: |
| 1 | 1484.09 / 18.07 | 1106.50 / 3.96 | 649.52 / 14.40 | 597.92 / 13.12 |
| 2 | 1474.38 / 10.34 | 1109.93 / 8.58 | 661.59 / 10.05 | 579.25 / 22.16 |
| 3 | 1478.44 / 10.19 | 1120.50 / 6.49 | 651.26 / 8.58 | 594.51 / 11.82 |
| 4 | 1482.98 / 4.29 | 1123.10 / 13.00 | 648.92 / 18.22 | 573.80 / 15.17 |
| 5 | 1500.76 / 7.70 | 1124.40 / 3.04 | 633.35 / 12.77 | 585.82 / 15.17 |
| 6 | 1497.76 / 7.18 | 1133.36 / 1.55 | 644.92 / 3.16 | 586.56 / 8.67 |

## CPU matching and homography recovery

| Pair | Before correct / matches | After correct / matches | Before / after precision | Before / after RANSAC inliers | Before / after corner L1 (px) |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1-2 | 1706/1919 | 1655/1839 | 88.90% / 89.99% | 1554 / 1563 | 0.88 / 1.26 |
| 1-3 | 690/1005 | 688/1040 | 68.66% / 66.15% | 617 / 616 | 1.20 / 2.10 |
| 1-4 | 177/365 | 167/342 | 48.49% / 48.83% | 152 / 140 | 2.13 / 1.73 |
| 1-5 | 21/148 | 33/174 | 14.19% / 18.97% | 22 / 22 | 482.11 / 3.46 |
| 1-6 | 3/124 | 3/136 | 2.42% / 2.21% | 12 / 9 | 4605.24 / 396.75 |

## MPS matching and homography recovery

| Pair | Before correct / matches | After correct / matches | Before / after precision | Before / after RANSAC inliers | Before / after corner L1 (px) |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1-2 | 1706/1919 | 1655/1839 | 88.90% / 89.99% | 1555 / 1563 | 0.88 / 1.26 |
| 1-3 | 690/1005 | 688/1040 | 68.66% / 66.15% | 617 / 616 | 1.20 / 2.10 |
| 1-4 | 178/366 | 167/342 | 48.63% / 48.83% | 149 / 140 | 3.15 / 1.73 |
| 1-5 | 22/149 | 33/174 | 14.77% / 18.97% | 21 / 22 | 472.96 / 3.46 |
| 1-6 | 3/124 | 3/136 | 2.42% / 2.21% | 12 / 9 | 4605.24 / 396.75 |

On CPU, the new detector recovers graf 1–5 at 3.46 px corner error versus
482.11 px before, while 1–6 still fails. CPU precision improves on 1–2, 1–4,
and 1–5, but falls on 1–3 and 1–6; correct-match counts decrease on 1–2 through
1–4. The detector changes keypoints and is not numerically equivalent to the
previous generic detector. Graf was used during development and
is not a held-out quality evaluation. Failed homographies remain in the tables.

For context, the earlier pre-PR patch baseline (`5be74dc9f`) measured 2155.72 ms on
CPU; same-revision patch extraction before this detector change measured
2159.27 ms on CPU and 679.72 ms on MPS. These older measurements are not the fresh
before/after comparison above. Patch CPU correct-match counts for pairs 1–2
through 1–6 were 1467, 481, 118, 12, 1, with corner errors 0.98, 1.58, 1.87,
472.69, 617.19 px respectively. The patch implementation remains unchanged.

CUDA, older supported PyTorch versions, and whole-pipeline compilation were not
validated. The sparse descriptor head executes eagerly; optional compilation
covers the specialized pyramid and/or sparse refinement only.
