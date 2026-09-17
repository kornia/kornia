# Shared-pyramid SIFT on Oxford graf

`DenseSIFTFeature` is an opt-in approximation for orienting and describing existing
similarity or affine LAFs. It shares DenseSIFT angular histograms across keypoints
at each Gaussian-pyramid octave, samples orientation support from those maps,
rotates the spatial descriptor grid, and remaps angular bins through the LAF's
transpose (including the direction-dependent magnitude for affine frames).

The baseline extracts an image patch for `LAFOrienter(19)`, then extracts an
oriented patch for `LAFDescriptor(SIFTDescriptor(41))`. Both use RootSIFT. The new
path uses 36 source angular bins, a weighted 9×9 orientation sampling grid with
parabolic peak refinement, an 8-pixel triangular pooling window, and 4×4 descriptor
cells with 8 output angular bins. It uses octave selection based on a 19-pixel
support for both stages; the baseline descriptor independently selects its
41-pixel support. Image-space pooled cells approximate a rotated/sheared footprint,
and gradient angular binning is interpolated a second time during affine remapping.
These differences mean descriptor equivalence is not expected.

## Protocol

Original 640×800 Oxford graf PPM images 1–6, grayscale float32, batch one, 4,096
requested detections per image. `SIFTFeatureScaleSpace(upright=True).detector`
provides deterministic fixed detections. The affine experiment additionally
applies `LAFAffineShapeEstimator` before both methods. Image loading, detection,
affine adaptation, and matching are outside the orientation+description timing.
Pyramid construction is inside every dense call; there is no cached pyramid.

CPU uses one thread. Timings use `benchmarks.common.time_us` (warmed median and
IQR), with explicit MPS synchronization. Six images are timed independently.
The summary averages their medians. This measures the changed stages, not
end-to-end feature extraction or matching. No autocast or compilation is used.
CUDA was unavailable on the test machine.

For each pair 1–2 through 1–6, SNN matching uses ratio 0.8. A match is correct if
its forward Euclidean transfer error under the supplied ground-truth homography
is at most 3 pixels. Precision is correct/matched, with zero for no matches.
Correct-match counts accompany precision to expose losses of correspondences.
No RANSAC or estimated homography affects this metric; it differs from the corner
error metric in `graf_benchmark.md`. Graf informed a small pooling-width diagnostic,
so these results are not a held-out quality evaluation.

## Results

Measured 2026-09-17 on Apple M1, macOS 26.5.1, PyTorch 2.14.0, Python 3.11.14.

| Frames / device | Patch ms/image | Dense ms/image | Patch / dense |
| --- | ---: | ---: | ---: |
| Similarity / CPU | 1180.4 | 469.6 | 2.51× |
| Similarity / MPS | 272.0 | 304.9 | 0.89× |
| Affine / CPU | 1204.2 | 484.0 | 2.49× |

Dense is approximately 2.5× faster on CPU for these 4,096-feature workloads,
but 12% slower on MPS. Neither result establishes performance at smaller feature
counts, on CUDA, or for a complete detection-and-matching pipeline.

### Similarity frames (CPU)

| Pair | Patch correct / matches | Patch precision | Dense correct / matches | Dense precision |
| --- | ---: | ---: | ---: | ---: |
| 1-2 | 1467 / 1651 | 88.9% | 1261 / 1419 | 88.9% |
| 1-3 | 481 / 706 | 68.1% | 369 / 585 | 63.1% |
| 1-4 | 118 / 223 | 52.9% | 78 / 171 | 45.6% |
| 1-5 | 12 / 101 | 11.9% | 5 / 87 | 5.7% |
| 1-6 | 1 / 61 | 1.6% | 4 / 40 | 10.0% |

### Affine-adapted frames (CPU)

| Pair | Patch correct / matches | Patch precision | Dense correct / matches | Dense precision |
| --- | ---: | ---: | ---: | ---: |
| 1-2 | 817 / 933 | 87.6% | 569 / 660 | 86.2% |
| 1-3 | 450 / 682 | 66.0% | 268 / 425 | 63.1% |
| 1-4 | 135 / 250 | 54.0% | 55 / 127 | 43.3% |
| 1-5 | 136 / 240 | 56.7% | 40 / 96 | 41.7% |
| 1-6 | 14 / 69 | 20.3% | 4 / 35 | 11.4% |

MPS dense match counts equal the CPU dense counts. The MPS patch baseline differs
by one accepted match on 1–2 and one correct match on 1–4; raw JSON retains those differences.

The CPU speed gain comes with fewer correct correspondences and generally lower
precision. The difficult 1–6 similarity pair has only four correct dense matches
versus one patch match; its higher precision does not demonstrate reliable recovery.
The affine 1–5 pair drops from 136 to 40 correct matches. Keep the patch path as
the default; this dense option is useful only when that CPU speed/quality tradeoff
fits the application.

Raw comparison artifacts:

- [base-affine-cpu](dense_sift_results/base-affine-cpu.json)
- [base-cpu](dense_sift_results/base-cpu.json)
- [base-mps](dense_sift_results/base-mps.json)
- [dense-affine-cpu](dense_sift_results/dense-affine-cpu.json)
- [dense-cpu](dense_sift_results/dense-cpu.json)
- [dense-mps](dense_sift_results/dense-mps.json)

## RANSAC homography evaluation

Added fixed-seed RANSAC runs for both descriptor paths on every pair, using the
same frozen detections, descriptors, and SNN ratio 0.8 as above. Settings match
the existing graf harness: homography model, 2 px inlier threshold, seed 3407,
8,196 hypotheses per batch, up to 10 batches, confidence 0.9999, and default
local refinement. MPS extraction/matching uses CPU RANSAC because the batched
SVD backend is unsupported there.

Error is the mean **L1 distance of the four projected image corners** between
the estimated and ground-truth homographies, in pixels. Consensus inliers are
reported separately: even an incorrect homography can have a consensus. Large
errors are retained; failed estimation or projection is recorded as null.

These supplemental runs use `--quality-only`: no latency measurement is made,
and timing fields are null. RANSAC is outside the earlier extraction-stage
timings, so the CPU speedup above is not a claim about total RANSAC-pipeline speed.

### Similarity frames, CPU

| Pair | Patch inliers | Patch corner L1 px | Dense inliers | Dense corner L1 px |
| --- | ---: | ---: | ---: | ---: |
| 1-2 | 1340 | 0.981 | 1149 | 1.417 |
| 1-3 | 420 | 1.584 | 318 | 2.519 |
| 1-4 | 92 | 1.872 | 66 | 1.995 |
| 1-5 | 18 | 472.688 | 17 | 474.403 |
| 1-6 | 12 | 617.188 | 9 | 578.960 |

[Raw Similarity frames, CPU results](dense_sift_results/ransac-cpu.json).

### Affine-adapted frames, CPU

| Pair | Patch inliers | Patch corner L1 px | Dense inliers | Dense corner L1 px |
| --- | ---: | ---: | ---: | ---: |
| 1-2 | 731 | 1.262 | 523 | 1.890 |
| 1-3 | 395 | 2.664 | 245 | 1.021 |
| 1-4 | 116 | 1.217 | 48 | 0.609 |
| 1-5 | 121 | 4.212 | 35 | 2.866 |
| 1-6 | 12 | 4.117 | 7 | 879.414 |

[Raw Affine-adapted frames, CPU results](dense_sift_results/ransac-affine-cpu.json).

### Similarity frames, MPS extraction + CPU RANSAC

| Pair | Patch inliers | Patch corner L1 px | Dense inliers | Dense corner L1 px |
| --- | ---: | ---: | ---: | ---: |
| 1-2 | 1355 | 1.303 | 1149 | 1.417 |
| 1-3 | 420 | 1.584 | 318 | 2.519 |
| 1-4 | 94 | 3.440 | 66 | 1.995 |
| 1-5 | 18 | 472.688 | 17 | 474.403 |
| 1-6 | 12 | 617.188 | 9 | 578.960 |

[Raw Similarity frames, MPS extraction + CPU RANSAC results](dense_sift_results/ransac-mps.json).

RANSAC changes the quality interpretation: despite fewer matches, dense affine
descriptors give lower corner error on 1–3, 1–4, and 1–5 in this fixed-seed run.
But on affine 1–6, patch SIFT recovers the homography (4.12 px), while dense SIFT
fails (879.41 px). Both similarity pipelines fail on 1–5 and 1–6. This is one
seed on one sequence; the lower errors on individual pairs do not establish a
general accuracy improvement.

Reproduce the supplemental evaluation (replace `--affine` with `--device mps`
for the MPS similarity run; omit both for CPU similarity):

```bash
.venv/bin/python -m benchmarks.feature.dense_sift --seq /tmp/graf --expected-checkout "$PWD" --quality-only --affine --json graf-ransac-affine.json
```

## Usage

```python
import kornia.feature as KF

# Existing similarity or affine detector frames:
oriented_lafs, descriptors = KF.DenseSIFTFeature()(gray_image, lafs)

# Convenience DoG pipeline; the default remains dense_sift=False:
features = KF.SIFTFeature(num_features=4096, dense_sift=True)
lafs, responses, descriptors = features(gray_image)
```

## Reproduce

Download the [Oxford graf archive](https://www.robots.ox.ac.uk/~vgg/research/affine/det_eval_files/graf.tar.gz)
and extract it outside the repository. From the checkout being measured:

```bash
.venv/bin/python -m benchmarks.feature.dense_sift --seq /tmp/graf --expected-checkout "$PWD" --device cpu --json graf-cpu.json
.venv/bin/python -m benchmarks.feature.dense_sift --seq /tmp/graf --expected-checkout "$PWD" --device mps --json graf-mps.json
.venv/bin/python -m benchmarks.feature.dense_sift --seq /tmp/graf --expected-checkout "$PWD" --affine --device cpu --json graf-affine-cpu.json
```

For the base revision, run the same benchmark script via `runpy.run_path` from
stdin in the base worktree, using the primary checkout's explicit interpreter and
`--methods patch`; print/check `kornia.__file__` as the script does. The base was
`5be74dc9f`; the candidate is the uncommitted working tree on that base. Raw JSON
retains the dirty marker, input checksums, software versions, per-image IQRs, and
all five pairs, including failures. Measurements were taken before committing the implementation.

## Validation

Focused CPU float32/float64 descriptor and SIFT integration tests: 43 passed,
2 skipped. New-path CPU float16/bfloat16 tests: 21 passed. MPS float32 tests:
12 passed, 1 skipped (float64 gradcheck). API-surface and benchmark-artifact
validation: 149 passed. Type checking and pre-commit checks passed; type checking
emits a deprecation diagnostic in LightGlue's AMP compatibility code.

Coverage includes affine gradient canonicalization, exact odd-size pyramid
coordinates, finite backward for flat images and invalid frames, mixed LAF dtype,
empty batches, descriptor normalization and a numerical gradient check. A seeded
CPU comparison of the existing DenseSIFTDescriptor against the base revision was
bitwise equal after retaining its original per-bin convolution schedule.
CUDA, older supported PyTorch versions, and whole-pipeline compilation were not
validated in this session. The full documentation image-generation job was not run.

RANSAC follow-up validation: synthetic homography recovery with outliers, the
non-square mean-L1 corner convention, insufficient matches, failed estimates,
nonfinite projections, and benchmark artifact schemas: 7 tests passed.
