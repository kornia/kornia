# Benchmark Results — ScaleSpaceDetector — Oxford Affine / graf

Sequence `graf`: `img1` vs `img2`–`img6` (5 pairs). NF=4096.
RANSAC `inl_th`=2 px, `max_iter`=10, `confidence`=0.9999, `seed`=3407.
Matching: `match_snn` threshold=0.85 (consistent across both versions).
CUDA device: RTX series (see machine). CPU: host CPU.

- **error [px]**  — nanmean corner reprojection error vs ground-truth homography (**lower is better**)
- **inliers [#]** — mean RANSAC inlier count (**higher is better**)
- **det [ms]**    — detection-only wall-clock (ScaleSpaceDetector forward, CUDA, one image)
- **total [ms]**  — detect both images + describe + match wall-clock (**lower is better**)

✨ = new subpix mode, not available in v0.8.2.

> **Note on variance** — only 5 image pairs; RANSAC is stochastic.
> Differences < ~30% in error or < ~10 inliers should be treated as noise.

---

## Matching Quality (CUDA only)

All scalespace configs use `--desc hardnet --ori orinet --aff affnet` — the best
available descriptor + orientation + affine-shape estimator.
Learned models (DISK, DeDoDe, ALIKED) are end-to-end; KeyNet uses its built-in
detector paired with AffNet + HardNet.

`det` and `total` columns are from the **new branch**.
nan pairs are excluded from the error mean (RANSAC found no valid homography).

| config | err v0.8.2 | err new↓ | inl v0.8.2 | inl new↑ | det [ms] | total [ms] |
|--------|----------:|--------:|-----------:|--------:|---------:|-----------:|
| `dog+soft+hardnet+orinet+affnet` | 122.5‡ | **2.1** | 260 | 523 | 107 | 204 |
| `dog+conv+hardnet+orinet+affnet` | 179.5‡ | **1.6** | 242 | **532** | 196 | 289 |
| `dog+iterative+hardnet+orinet+affnet` ✨ | — | **1.6** | — | **532** | 241 | 332 |
| `dog+adaptive+hardnet+orinet+affnet` ✨ | — | **1.6** | — | **532** | 179 | **272** |
| `hessian+soft+hardnet+orinet+affnet` | 108.9‡ | 2.9 | 271 | 511 | **114** | 206 |
| `hessian+conv+hardnet+orinet+affnet` | 131.7‡ | 2.4 | 334 | 509 | 194 | 287 |
| `hessian+iterative+hardnet+orinet+affnet` ✨ | — | 2.4 | — | 509 | 261 | 355 |
| `hessian+adaptive+hardnet+orinet+affnet` ✨ | — | 2.4 | — | 509 | 197 | 288 |
| `DISK` | 7.0 | 7.6 | 127 | 155 | — | 29 |
| `DeDoDe` | 126867† | 187.7† | 3 | 125 | — | 79 |
| `ALIKED` | N/A | 4.3 | N/A | 245 | — | 30 |
| `KeyNet+AffNet+HardNet` | 2.6 | 2.5 | 426 | 360 | — | 158 |

† DeDoDe uses `L-upright` + `B-upright` weights; strong viewpoint change (pairs 5–6) causes
large errors even in the new branch. In v0.8.2 the descriptor had dtype/sampling bugs causing
total matching failure.

‡ v0.8.2 scalespace: pair 1–6 consistently produces a degenerate homography (large error);
the ConvSoftArgmax3d and NMS regressions cause poor localization on harder pairs.

**Takeaway**: the new branch fixes fundamental localization bugs in scalespace detectors,
reducing mean error from ~100–180 px to 1.6–2.9 px. DISK and KeyNet are largely unaffected
by the scalespace changes. ALIKED is new in this branch.

---

## Detection Speed (ms per call†)

†Scalespace: detection only (ScaleSpaceDetector forward). DISK, ALIKED, DeDoDe: full
model forward (detect + describe). KeyNet detection only (batching >1 N/A in v0.8.2).

| config | CPU 1 v0.8.2 | CPU 1 new | CPU 4 v0.8.2 | CPU 4 new | CPU 8 v0.8.2 | CPU 8 new | GPU 1 v0.8.2 | GPU 1 new | GPU speedup¹ |
|--------|-------------:|----------:|-------------:|----------:|-------------:|----------:|-------------:|----------:|-------------:|
| `dog+soft` | 2597 | 2442 | 4174 | 4346 | 6258 | 6831 | 811 | **42.7** | **~19×** |
| `dog+conv` | 1702 | 361 | 6407 | 1591 | 12487 | 3544 | 131 | 82.7 | ~1.6× |
| `dog+iterative` ✨ | — | 349 | — | 1431 | — | 3365 | — | 109.3 | — |
| `dog+adaptive` ✨ | — | **322** | — | 1571 | — | 3365 | — | 83.7 | — |
| `hessian+soft` | 441 | 2281 | 914 | 4868 | 1369 | 8190 | 122 | 50.8 | ~2.4× |
| `hessian+conv` | 290 | 578 | 1246 | 2613 | 2210 | 5435 | 66 | 98.8 | 0.7ד |
| `hessian+iterative` ✨ | — | 512 | — | 2278 | — | 4784 | — | 111.6 | — |
| `hessian+adaptive` ✨ | — | 487 | — | 2270 | — | 4521 | — | 87.6 | — |
| `DISK` (full fwd) | 575 | 521 | 1933 | 1928 | 4105 | 3998 | **12.7** | 13.1 | ~1.0× |
| `ALIKED` (full fwd) | N/A | 342 | N/A | 1749 | N/A | 3471 | N/A | **12.1** | — |
| `DeDoDe` (full fwd) | 4823 | 4604 | 17853 | 17789 | 33496 | 44882 | 69 | 37.7 | ~1.8× |

¹ GPU speedup = v0.8.2 GPU bs=1 / new GPU bs=1. Negative entries mean new branch is slower.
† `hessian+conv` new-branch regression on GPU (new 98.8 ms vs v0.8.2 66 ms); quality is
still dramatically better (2.4 px vs 131 px mean error).

**Takeaways**:
- `dog+soft` GPU: **~19× faster** — ConvSoftArgmax3d rewrite removes a major bottleneck.
- `dog+soft` CPU: essentially unchanged (both ~2.4–2.6 s/img); the rewrite targeted GPU paths.
- `dog+conv` CPU: **~4.7× faster** — NMS/topk refactor helps CPU batch throughput.
- `hessian+soft` CPU: **slower** in new branch — the DoG-specific multi-scale pipeline was
  restructured; hessian sees a regression in CPU single-image time.
- `hessian+conv` GPU: slight regression vs v0.8.2 (quality is massively better regardless).
- `DeDoDe` GPU: ~1.8× faster in new branch (descriptor dtype fix + forward pass cleanup).
- **ALIKED** is the fastest model overall on GPU for single-image use (new only).
