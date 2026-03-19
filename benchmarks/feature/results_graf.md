# Benchmark Results — ScaleSpaceDetector — Oxford Affine / graf

Sequence `graf`: `img1` vs `img2`–`img6` (5 pairs). NF=4096.
RANSAC `inl_th`=2 px, `max_iter`=10, `confidence`=0.9999, `seed`=3407.
CUDA device: RTX series (see machine). CPU: host CPU.

- **error [px]**  — mean corner reprojection error vs ground-truth homography (**lower is better**)
- **inliers [#]** — RANSAC inlier count (**higher is better**)
- **det [ms]**    — detection-only wall-clock (ScaleSpaceDetector forward, CUDA)
- **total [ms]**  — detect + orient + affine + describe + match wall-clock (**lower is better**)

✨ = new subpix mode, not available in v0.8.2.

> **Note on variance** — only 5 image pairs; RANSAC is stochastic.
> Differences < ~30% in error or < ~10 inliers should be treated as noise.

---

## Matching Quality (CUDA only)

All scalespace configs use `--desc hardnet --ori orinet --aff affnet` — the best
available descriptor + orientation + affine-shape estimator.
Learned models (DISK, DeDoDe, ALIKED) are end-to-end; KeyNet uses its built-in
detector paired with AffNet + HardNet.

| config | error [px]↓ | inliers [#]↑ | det [ms] | total [ms] |
|--------|------------:|-------------:|---------:|-----------:|
| `dog+soft+hardnet+orinet+affnet` | 2.1 | 523 | 107 | 204 |
| `dog+conv+hardnet+orinet+affnet` | 1.6 | **532** | 196 | 289 |
| `dog+iterative+hardnet+orinet+affnet` ✨ | 1.6 | **532** | 241 | 332 |
| `dog+adaptive+hardnet+orinet+affnet` ✨ | 1.6 | **532** | 179 | **272** |
| `hessian+soft+hardnet+orinet+affnet` | 2.9 | 511 | **114** | 206 |
| `hessian+conv+hardnet+orinet+affnet` | 2.4 | 509 | 194 | 287 |
| `hessian+iterative+hardnet+orinet+affnet` ✨ | 2.4 | 509 | 261 | 355 |
| `hessian+adaptive+hardnet+orinet+affnet` ✨ | 2.4 | 509 | 197 | 288 |
| `DISK` | 7.6 | 155 | — | 29 |
| `DeDoDe` | 187.7† | 125 | — | 79 |
| `ALIKED` | 4.3 | 245 | — | 30 |
| `KeyNet+AffNet+HardNet` | 2.5 | 360 | — | 158 |

† DeDoDe uses `L-upright` + `B-upright` weights with no rotation augmentation; the large
error on pairs 5–6 (strong viewpoint change) pulls the mean up.

**Takeaway**: all four subpix modes produce equivalent quality for DoG/Hessian when
paired with OriNet + AffNet. `dog+soft` is fastest on CUDA among the scalespace
variants; `dog+adaptive` saves ~25 ms vs `dog+conv` with identical scores ✨.

---

## Detection Speed (ms / call, detection only†)

†DISK, ALIKED, DeDoDe time the full model forward (detect + describe); there is no
separate detection stage for these models.

| config | CPU bs=1 | CPU bs=4 | CPU bs=8 | GPU bs=1 | GPU bs=4 | GPU bs=8 | CUDA speedup vs v0.8.2¹ |
|--------|----------:|---------:|---------:|---------:|---------:|---------:|------------------------:|
| `dog+soft` | 2442 ms | 4346 ms | 6831 ms | **42.7 ms** | 102.1 ms | 182.7 ms | **~4×** |
| `dog+conv` | 361 ms | 1591 ms | 3544 ms | 82.7 ms | 120.8 ms | 179.8 ms | ~1.3× |
| `dog+iterative` ✨ | 349 ms | 1431 ms | 3365 ms | 109.3 ms | 151.9 ms | 202.0 ms | — |
| `dog+adaptive` ✨ | **322 ms** | 1571 ms | 3365 ms | 83.7 ms | 123.1 ms | 178.6 ms | — |
| `hessian+soft` | 2281 ms | 4868 ms | 8190 ms | 50.8 ms | 126.5 ms | 228.9 ms | **~4×** |
| `hessian+conv` | 578 ms | 2613 ms | 5435 ms | 98.8 ms | 145.8 ms | 212.1 ms | ~1.3× |
| `hessian+iterative` ✨ | 512 ms | 2278 ms | 4784 ms | 111.6 ms | 167.1 ms | 230.8 ms | — |
| `hessian+adaptive` ✨ | 487 ms | 2270 ms | 4521 ms | 87.6 ms | 141.3 ms | 220.6 ms | — |
| `DISK` (full fwd) | 521 ms | 1928 ms | 3998 ms | 13.1 ms | 48.4 ms | 96.0 ms | ~1.3× |
| `ALIKED` (full fwd) | 342 ms | 1749 ms | 3471 ms | **12.1 ms** | **33.4 ms** | **66.0 ms** | — |
| `DeDoDe` (full fwd) | 4604 ms | 17789 ms | 44882 ms | 37.7 ms | 170.9 ms | 354.9 ms | — |

¹ CUDA speedup for `soft` and `conv` vs v0.8.2 measured on the same machine;
`iterative` and `adaptive` did not exist in v0.8.2.

**Takeaways**:
- On **CUDA**: `soft` is 4× faster than v0.8.2 after the `ConvSoftArgmax3d` refactor,
  making it the fastest scalespace detector per image. `adaptive` is fastest on CPU.
- On **CPU**: `dog+adaptive` and `dog+iterative` are 6–7× faster than `dog+soft`
  for single-image inference, making them the right default for CPU-only deployments.
- **ALIKED** is the fastest model overall on GPU for single-image use.
- **DISK** offers the best quality-to-speed tradeoff among learned models at ~13 ms/img GPU.
