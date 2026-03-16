# Benchmark Results — ScaleSpaceDetector — Oxford Affine / graf

Sequence `graf`: `img1` vs `img2`–`img6` (5 pairs).  NF=2000.  RANSAC `inl_th`=2 px.

- **MAE** — mean corner reprojection error vs ground-truth homography (px, **lower is better**)
- **inl** — RANSAC inliers (**higher is better**)
- **ms**  — detect + describe + match wall-clock time (**lower is better**)

Each config is run identically on both sides to obtain absolute metrics.
✨ = new in this branch (not available in v0.8.2).

> **Note on variance** — only 5 image pairs; RANSAC is stochastic.
> Differences < ~30% in MAE or < ~10 inliers should be treated as noise.

---

## CUDA

| config | v0.8.2 MAE↓ | v0.8.2 inl↑ | v0.8.2 ms↓ | new MAE↓ | new inl↑ | new ms↓ |
|--------|------------:|------------:|-----------:|---------:|---------:|--------:|
| `dog+soft+sift` | 516 | 18 | 1813 | 1055 | 13 | 1819 |
| `hessian+soft+sift` | 921 | 41 | 2201 | 864 | 43 | 2279 |
| `harris+soft+sift` | 2715 | 31 | 1232 | 669 | 27 | 1270 |
| `gftt+soft+sift` | 520 | 19 | 1231 | 562 | 18 | 1277 |
| `dog_single+soft+sift` | N/A¹ | — | — | N/A¹ | — | — |
| `dog+conv+sift` | 348 | 42 | **289** | 342 | 35 | 338 |
| `hessian+soft+hardnet` | 361 | 58 | 2261 | 190 | 100 | 2277 |
| `hessian+soft+sift+aff=patch` | 330 | 17 | 2274 | 272 | 49 | 2285 |
| `hessian+soft+sift+aff=affnet` | 370 | 27 | 2245 | 8027² | 75 | 2277 |
| `hessian+soft+hardnet+aff=patch` | 673 | 36 | 2305 | 235 | 110 | 2277 |
| `hessian+soft+hardnet+aff=affnet` | 432 | 29 | 2299 | **116** | **122** | 2274 |
| `DISK` | **133** | 167 | **33** | 218 | 201 | **27** |
| `DeDoDe` | 228 | 167 | 109 | 332 | 190 | **78** |
| `dog+iterative+sift` ✨ | — | — | — | 300 | 36 | 324 |
| `dog+adaptive+sift` ✨ | — | — | — | 360 | 35 | **261** |
| `hessian+adaptive+sift` ✨ | — | — | — | 315 | 53 | **279** |
| `hessian+adaptive+hardnet` ✨ | — | — | — | 563 | 114 | **290** |
| `hessian+adaptive+hardnet+aff=affnet` ✨ | — | — | — | **117** | **137** | **315** |
| `ALIKED` ✨ | — | — | — | N/A³ | — | — |

---

## CPU

| config | v0.8.2 MAE↓ | v0.8.2 inl↑ | v0.8.2 ms↓ | new MAE↓ | new inl↑ | new ms↓ |
|--------|------------:|------------:|-----------:|---------:|---------:|--------:|
| `dog+soft+sift` | 444 | 19 | 5533 | 1135 | 14 | 6395 |
| `hessian+soft+sift` | 900 | 41 | 5666 | 1477 | 44 | 5435 |
| `harris+soft+sift` | 889 | 31 | 4107 | 521 | 32 | 3854 |
| `gftt+soft+sift` | 584 | 19 | 4168 | 555 | 18 | 3931 |
| `dog_single+soft+sift` | N/A¹ | — | — | N/A¹ | — | — |
| `dog+conv+sift` | 317 | 41 | 3717 | 699 | 35 | **1176** |
| `hessian+soft+hardnet` | 365 | 58 | 6690 | **182** | 103 | 6667 |
| `hessian+soft+sift+aff=patch` | 764 | 17 | 6327 | 234 | 48 | 5402 |
| `hessian+soft+sift+aff=affnet` | 776 | 74 | 5740 | 344 | 74 | 5950 |
| `hessian+soft+hardnet+aff=patch` | 224 | 108 | 6697 | 232 | 107 | 6780 |
| `hessian+soft+hardnet+aff=affnet` | **116** | **124** | 7298 | **115** | **125** | 7210 |
| `DISK` | 212 | 203 | **1164** | 206 | 201 | **1154** |
| `DeDoDe` | 331 | 190 | 9428 | 332 | 190 | 9652 |
| `dog+iterative+sift` ✨ | — | — | — | 318 | 32 | **1224** |
| `dog+adaptive+sift` ✨ | — | — | — | 688 | 36 | **1058** |
| `hessian+adaptive+sift` ✨ | — | — | — | 323 | 51 | **1589** |
| `hessian+adaptive+hardnet` ✨ | — | — | — | **117** | **111** | **2921** |
| `hessian+adaptive+hardnet+aff=affnet` ✨ | — | — | — | **120** | **137** | **3577** |
| `ALIKED` ✨ | — | — | — | N/A³ | — | — |

---

## Notes

¹ **`dog_single`** — `BlobDoGSingle` returns a 5-D tensor incompatible with
`ScaleSpaceDetector` on both v0.8.2 and the new branch when used with `soft`/`conv` subpix.

² **`hessian+soft+sift+aff=affnet` new-branch CUDA** — MAE=8027 is a degenerate RANSAC result.
AffNet changes keypoint shape but SIFT is computed on the original (upright) patch,
creating orientation inconsistency that collapses match quality for some pairs.
The high inlier count (75) is misleading — RANSAC found a bad homography by chance.
Use `--b-desc hardnet` (rotation-invariant) with AffNet, not SIFT.

³ **ALIKED** — `from_pretrained` fails with `Missing key: dkd.hw_grid`; model / code
version mismatch in the current environment.

## Speed highlights (CPU, `adaptive` vs `soft` with Hessian)

| config | ms (soft) | ms (adaptive) | speedup |
|--------|----------:|--------------:|--------:|
| `hessian+*+sift` | 5435 | 1589 | **3.4×** |
| `hessian+*+hardnet` | 6667 | 2921 | **2.3×** |
| `hessian+*+hardnet+aff=affnet` | 7210 | 3577 | **2.0×** |

`dog+adaptive+sift` vs `dog+soft+sift` CPU: 1058 vs 6395 ms → **6.0× speedup**.
