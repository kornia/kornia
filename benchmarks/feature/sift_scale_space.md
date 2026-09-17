# Dedicated scale-space SIFT on Oxford graf

`SIFTFeatureScaleSpace(descriptor_backend="pyramid")` remains a sparse local
feature pipeline. It uses the same DoG detector, feature budget, masks, responses,
and frame centres as the default patch path. It retains the detector's actual
Gaussian pyramid and octave/layer provenance through feature selection. The
nearest refined Gaussian layer supplies gradients; the continuously refined scale
sets the integration support. No descriptor-owned pyramid is built.

Gradients are computed once per used Gaussian layer and sampled directly for
orientation and description. The SIFT-specific implementation uses a 36-bin
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
