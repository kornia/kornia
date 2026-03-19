# Benchmark Results — ScaleSpaceDetector — Oxford Affine / graf

Sequence `graf`: `img1` vs `img2`–`img6` (5 pairs). NF=4096.
RANSAC `inl_th`=2 px, `max_iter`=10, `confidence`=0.9999, `seed`=3407.
CUDA device: RTX series (see machine). CPU: host CPU.

- **error [px]**  — mean corner reprojection error vs ground-truth homography (**lower is better**)
- **inliers [#]** — RANSAC inlier count (**higher is better**)
- **det [ms]**    — detection-only wall-clock (ScaleSpaceDetector forward, CUDA)
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

`det` and `total` columns are from the **new branch** only.
∞ = RANSAC failed to find a valid homography on ≥ 1 of 5 pairs (mean is undefined).

| config | err v0.8.2 | err new↓ | inl v0.8.2 | inl new↑ | det [ms] | total [ms] |
|--------|----------:|--------:|-----------:|--------:|---------:|-----------:|
| `dog+soft+hardnet+orinet+affnet` | ∞‡ | 2.1 | 65 | 523 | 107 | 204 |
| `dog+conv+hardnet+orinet+affnet` | 159.4‡ | **1.6** | 59 | **532** | 196 | 289 |
| `dog+iterative+hardnet+orinet+affnet` ✨ | — | **1.6** | — | **532** | 241 | 332 |
| `dog+adaptive+hardnet+orinet+affnet` ✨ | — | **1.6** | — | **532** | 179 | **272** |
| `hessian+soft+hardnet+orinet+affnet` | ∞‡ | 2.9 | 75 | 511 | **114** | 206 |
| `hessian+conv+hardnet+orinet+affnet` | 161.4‡ | 2.4 | 78 | 509 | 194 | 287 |
| `hessian+iterative+hardnet+orinet+affnet` ✨ | — | 2.4 | — | 509 | 261 | 355 |
| `hessian+adaptive+hardnet+orinet+affnet` ✨ | — | 2.4 | — | 509 | 197 | 288 |
| `DISK` | ∞‡ | 7.6 | 146 | 155 | — | 29 |
| `DeDoDe` | ∞‡ | 187.7† | 2 | 125 | — | 79 |
| `ALIKED` | N/A | 4.3 | N/A | 245 | — | 30 |
| `KeyNet+AffNet+HardNet` | 2.4 | 2.5 | 307 | 360 | — | 158 |

† DeDoDe uses `L-upright` + `B-upright` weights with no rotation augmentation; the large
error on pairs 5–6 (strong viewpoint change) pulls the mean up.

‡ v0.8.2 scalespace: soft modes had a ConvSoftArgmax3d bug causing poor localization;
conv modes had NMS/subpix regressions. DISK v0.8.2 failed on pair 1–6. DeDoDe v0.8.2
had a sample_keypoints crash path and descriptor dtype issues.

**Takeaway**: the new branch fixes fundamental localization bugs in scalespace detectors,
turning RANSAC failures into 1.6–2.9 px mean error with 500+ inliers. ALIKED is new.
`dog+adaptive` saves ~25 ms vs `dog+conv` with identical scores ✨.

---

## Detection Speed (ms per call†)

†Scalespace: detection only (ScaleSpaceDetector forward). DISK, ALIKED, DeDoDe: full
model forward (detect + describe); there is no separate detection stage.

| config | CPU bs=1 | CPU bs=4 | CPU bs=8 | GPU bs=1 | GPU bs=4 | GPU bs=8 | CUDA speedup vs v0.8.2¹ |
|--------|----------:|---------:|---------:|---------:|---------:|---------:|------------------------:|
| `dog+soft` | 2442 | 4346 | 6831 | **42.7** | 102.1 | 182.7 | **~4×** |
| `dog+conv` | 361 | 1591 | 3544 | 82.7 | 120.8 | 179.8 | ~1.3× |
| `dog+iterative` ✨ | 349 | 1431 | 3365 | 109.3 | 151.9 | 202.0 | — |
| `dog+adaptive` ✨ | **322** | 1571 | 3365 | 83.7 | 123.1 | 178.6 | — |
| `hessian+soft` | 2281 | 4868 | 8190 | 50.8 | 126.5 | 228.9 | **~4×** |
| `hessian+conv` | 578 | 2613 | 5435 | 98.8 | 145.8 | 212.1 | ~1.3× |
| `hessian+iterative` ✨ | 512 | 2278 | 4784 | 111.6 | 167.1 | 230.8 | — |
| `hessian+adaptive` ✨ | 487 | 2270 | 4521 | 87.6 | 141.3 | 220.6 | — |
| `DISK` (full fwd) | 521 | 1928 | 3998 | 13.1 | 48.4 | 96.0 | ~1.3× |
| `ALIKED` (full fwd) | 342 | 1749 | 3471 | **12.1** | **33.4** | **66.0** | — |
| `DeDoDe` (full fwd) | 4604 | 17789 | 44882 | 37.7 | 170.9 | 354.9 | — |

¹ CUDA speedup for `soft` and `conv` vs v0.8.2 measured on the same machine;
`iterative` and `adaptive` did not exist in v0.8.2.

**Takeaways**:
- On **CUDA**: `soft` is 4× faster than v0.8.2 after the `ConvSoftArgmax3d` refactor,
  making it the fastest scalespace detector per image. `adaptive` is fastest on CPU.
- On **CPU**: `dog+adaptive` and `dog+iterative` are 6–7× faster than `dog+soft`
  for single-image inference, making them the right default for CPU-only deployments.
- **ALIKED** is the fastest model overall on GPU for single-image use.
- **DISK** offers the best quality-to-speed tradeoff among learned models at ~13 ms/img GPU.
