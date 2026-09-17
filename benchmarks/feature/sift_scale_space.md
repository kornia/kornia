# Specialized top-K SIFT: correctness review and optimization

For the consolidated device comparison and current figures, see the [SIFT benchmark summary](sift_summary.md).
Raw measurements are linked to commit `efb04dbf`; they are intentionally absent from the PR file diff.

This historical pass compares PR #4638 head `3172b5f1001140f638bfa2c3934633d2be6b9851` with optimized revision `bc7909746f96bf3e9acb61fd368dc95ce0bca37f`. See the [subsequent memory-focused review and optimization](sift_memory.md) for current results. The default patch backend is unchanged. All measurements time the entire public `SIFTFeatureScaleSpace(4096, descriptor_backend="pyramid")` forward, including detection, orientation, and RootSIFT.

## Results

Apple M1, macOS 26.5.1, Python 3.11.14, PyTorch 2.14.0; Oxford graf images 1–6, 640×800, float32, batch one, one CPU thread. Values below are the arithmetic mean of six warmed per-image medians. Every Kornia extraction returns **4,096 valid features**. No contrast threshold or edge rejection is introduced, and the Gaussian support and 19×19/41×41 integration grids are unchanged.

| Device | Reviewed PR ms/image | Optimized ms/image | Speedup |
| --- | ---: | ---: | ---: |
| CPU | 1201.12 | 794.80 | 1.51× |
| MPS | 622.61 | 587.22 | 1.06× |

![Historical Graf quality and extraction runtime](https://raw.githubusercontent.com/kornia/kornia/efb04dbf9c85e4cf71625cc2467bd5243b0c803c/benchmarks/feature/sift_scale_space.svg)

The OpenCV 5.0.0 CPU series is retained from the earlier PR measurement on this machine, using native uint8 SIFT plus NumPy RootSIFT and its default contrast/edge rejection. Its 2,676–4,096 returned features are an equal requested budget, not equal actual work. It is contextual; the speedup table uses only the fresh reviewed/optimized A/B runs. CPU geometry is shown in the figure; failed homographies remain visible.

## What changed

- Large CPU Gaussian images use separable weighted slice accumulation instead of convolution work buffers. Levels below 65,536 elements and direct half-precision calls retain convolution; the production detector already promotes half inputs. This dispatch avoids the small-image regression seen with an unconditional slice path.
- Four axial comparisons identify possible extrema before one sparse gather checks all 26 neighbours. The same gather supplies refinement finite differences. Flattened indexing avoids a dense 27-fold unfolded gradient buffer in backward.
- Each descriptor sample writes its two nonzero angular votes directly. CPU chunks of 128 keep temporaries smaller; accelerators retain chunks of 1,024. MPS uses contiguous angular-bin/sample matrices for histogram multiplication.
- The generic `SIFTDescriptorFromPyramid` skips unused upper-octave histograms and preserves zero gradients for all-invalid frames. Its historical speed/quality report is [separate](dense_sift.md); this benchmark measures the specialized pipeline.
- In-place pyramid compilation preserves eager checkpoint keys. Full-module serialization discards transient compiled callables and restores eager methods on load.

Separable pooling for the generic backend was rejected after a public CPU A/B regression. Gaussian transpose, conv1d, and channels-last experiments were also rejected. No new runtime dependencies were added.

## Matching and homography quality

SNN ratio 0.8; a correct match has forward ground-truth Euclidean transfer error ≤3 px. Homography RANSAC: seed 3407, 2 px inlier threshold, 10 batches of 8,196 hypotheses, confidence 0.9999, default local refinement. Corner error is the mean L1 ground-truth transfer discrepancy at four image corners. Matching/RANSAC are outside extraction timing; MPS uses CPU RANSAC.

CPU accumulation order changes slightly, so feature ordering and RANSAC samples can change. Correct-match counts stay within one of the reviewed PR, but corner errors improve on some pairs and regress on others. Graf 1–6 still fails. Graf informed development and is not a held-out quality evaluation.

| Pair | Correct matches before / after | Precision before / after | RANSAC inliers before / after | Corner L1 px before / after |
| --- | ---: | ---: | ---: | ---: |
| 1-2 | 1655 / 1656 | 89.99% / 90.00% | 1563 / 1570 | 1.265 / 2.035 |
| 1-3 | 688 / 687 | 66.15% / 66.06% | 616 / 616 | 2.096 / 1.253 |
| 1-4 | 167 / 166 | 48.83% / 48.68% | 140 / 141 | 1.729 / 1.824 |
| 1-5 | 33 / 32 | 18.97% / 18.50% | 22 / 23 | 3.457 / 5.637 |
| 1-6 | 3 / 3 | 2.21% / 2.19% | 9 / 9 | 396.751 / 444.140 |

MPS matching counts, precision, RANSAC inliers, and corner errors are unchanged in these runs. All raw CPU/MPS quality rows are retained.

## Per-image timings

Each cell is median ± IQR in milliseconds. IQR is timing spread, not a confidence interval. Other applications on the machine were left untouched; aggregate load metrics are recorded with every run.

| Image | CPU before | CPU after | MPS before | MPS after |
| --- | ---: | ---: | ---: | ---: |
| 1 | 1203.24 ± 111.83 | 783.74 ± 9.31 | 550.85 ± 6.32 | 592.27 ± 19.37 |
| 2 | 1164.74 ± 33.47 | 786.41 ± 6.71 | 550.64 ± 9.54 | 591.96 ± 10.56 |
| 3 | 1448.08 ± 159.97 | 790.09 ± 10.25 | 647.21 ± 23.04 | 586.44 ± 11.74 |
| 4 | 1143.44 ± 119.76 | 800.61 ± 5.84 | 658.54 ± 15.89 | 579.37 ± 8.78 |
| 5 | 1097.02 ± 44.39 | 801.96 ± 11.49 | 672.21 ± 14.32 | 588.78 ± 12.16 |
| 6 | 1150.18 ± 27.88 | 806.02 ± 13.67 | 656.19 ± 11.04 | 584.47 ± 10.00 |

## Reproduction and provenance

Raw files are archived at the pre-cleanup commit in [`sift_scale_space_results/`](https://github.com/kornia/kornia/tree/efb04dbf9c85e4cf71625cc2467bd5243b0c803c/benchmarks/feature/sift_scale_space_results/). Each includes source and input SHA256 hashes, versions, aggregate load, counts, timings, and matching/RANSAC rows. Before runs are from the exact reviewed head; after runs record its dirty working-tree revision plus the hashes of the measured implementation. Those source hashes identify the historical measured implementations; the current follow-up is linked above. Report/plot edits happen after timing.

The harness verifies `kornia.__file__` against `--expected-checkout`; run as a module from each checkout with the same explicit interpreter. It uses `benchmarks.common.time_us`, at least max(1 second, five warm-call durations) of repeated timing, and MPS synchronization inside the timed call. CUDA, large batches, held-out datasets, and compiled end-to-end speed have not been benchmarked.

```bash
# In the reviewed checkout, then repeat in the optimized checkout:
/path/to/the/same/python -m benchmarks.feature.sift_scale_space \
  --seq /path/to/graf --expected-checkout "$PWD" --methods pyramid \
  --device cpu --json /tmp/sift-before-cpu.json
# Repeat with --device mps and distinct output paths.

# Restore historical inputs outside the checkout before plotting:
git archive efb04dbf9c85e4cf71625cc2467bd5243b0c803c benchmarks/feature/sift_scale_space_results | tar -x -C /tmp
.venv/bin/python benchmarks/feature/plot_sift_runtime.py --scale-space \
  --inputs /tmp/benchmarks/feature/sift_scale_space_results/before-cpu.json \
           /tmp/benchmarks/feature/sift_scale_space_results/after-cpu.json \
           /tmp/benchmarks/feature/sift_scale_space_results/opencv-cpu.json \
           /tmp/benchmarks/feature/sift_scale_space_results/before-mps.json \
           /tmp/benchmarks/feature/sift_scale_space_results/after-mps.json \
  --labels "Reviewed PR" "Optimized" "OpenCV" "Reviewed PR" "Optimized" \
  --cpu-label "Apple M1" --output /tmp/sift-optimized
```

## Correctness validation

Regression tests independently pin trilinear bin layout/wraparound, sparse-neighbour values and backward accumulation, strict diagonal/scale ties, Gaussian values/gradients against convolution (including tiny images and asymmetric kernels), masks/padding, and eager/compiled checkpoint roundtrips. CPU float16/bfloat16/float32/float64, MPS float32, and minimum-supported PyTorch 2.5.1 are covered. Native padding capability skips apply only to direct private-helper tests on unsupported dtype/backend combinations; public half inputs retain coverage. Focused compiler tests cover the pyramid, fullgraph CPU slice filter, and sparse refinement. Repository validation is recorded in the PR review.
