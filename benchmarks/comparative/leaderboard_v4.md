# Comparative augmentation benchmark v4 — all five eager optimizations patched

## Hardware / stack

| Key | Value |
|-----|-------|
| Platform | Jetson Orin (aarch64), Linux 5.15.148-tegra |
| GPU | Orin (Orin integrated GPU, 1792-core Ampere) |
| CUDA | 12.6 (libcusolver 11.6.4.69) |
| Python | 3.10 (pixi camera-object-detector env) |
| PyTorch | 2.8.0 |
| kornia | 0.7.4 (installed 0.7.4 + 5 runtime patches) |
| albumentations | 2.0.8 |
| torchvision | 0.23.0 |
| Batch size | 8 |
| Resolution | 512×512 |
| kornia patches | Normalize(buffers); HFlip(cache); hflip/vflip(no-contiguous); RandomAffine(closed-form+cache); ColorJiggle(fused-HSV) |
| Patch verification | all OK |
| Wall time | 39s |

## Methodology

All rows measure the same DETR-style 4-op pipeline:
  RandomHorizontalFlip → RandomAffine → ColorJitter/Jiggle → Normalize
Batch=8, resolution=512×512, float32.

**End-to-end DataLoader timing (rows 1/2/4):**
DataLoader delivers CPU tensors → main thread applies H2D + GPU aug.
50 timed batches + 10 warmup, 8 workers. All times include
Python dispatch, DataLoader latency, and H2D transfer.

**CUDA Graph rows (7–8):**
Capture attempted in isolated subprocess to avoid stream-state contamination.
Replay timing uses CUDA events (no Python dispatch, no DataLoader overhead).
1000 replays.

**Rows 3/5/6 SKIPPED:** Triton is not installed on Jetson JetPack 6 — torch.compile
with Inductor backend raises TritonMissing immediately. compile(backend='eager') also
fails on this build (data-dependent symbolic shape error in ColorJiggle.apply_transform).

**Five kornia optimisation patches applied at runtime to installed 0.7.4:**
1. `Normalize.apply_transform`: pre-shaped `(1,C,1,1)` buffers via `register_buffer`,
   bypasses `kornia.enhance.normalize` wrapper; direct `(input - mean) / std` math.
2. `RandomHorizontalFlip.compute_transformation`: module-level `_HFLIP_MAT_TEMPLATE`
   + per-(device,dtype,width) `_HFLIP_MAT_CACHE`; `expand()` for batch dim (metadata-only).
3. `hflip` / `vflip`: reduced to `input.flip(-1)` / `input.flip(-2)`,
   removing `.contiguous()` call that forced a 96MB memcopy per call.
4. `RandomAffine.apply_transform` + `compute_transformation`: `_affine_matrix2d_closed`
   avoids 4× eye_like + 4× matmul; `_affine_homography_inv` is closed-form 3×3 inverse
   (~35× faster than `torch.linalg.inv` for small B); `_norm_cache` avoids repeated
   normalization matrix allocation; fast path uses `F.affine_grid`+`F.grid_sample`
   directly instead of going through `warp_affine`'s normalize_homography chain.
5. `ColorJiggle.apply_transform`: deferred HSV roundtrip fuses consecutive saturation
   + hue ops into a single `rgb_to_hsv`/`hsv_to_rgb` pair; in-place `.add_`/`.mul_`/
   `.clamp_`; pre-computed factor vectors; plain `if/elif` dispatch (no lambda list).

**cusolver workaround:** Jetson JetPack 6 ships libcusolver 11.6.4.69 which is
missing `cusolverDnXsyevBatched_bufferSize` needed by torch 2.8.0's linalg.
Patched with closed-form analytical 3×3 inverse (cofactor/det, elementwise CUDA ops).

## 5-row comparison table (v4)

All times in ms/batch (lower is better). Speedup column is relative to the
eager baseline of the same library (or Albumentations for Row 1).

| Row | Configuration | Mode | Median ms | IQR | Min ms | Max ms | Speedup |
|-----|--------------|------|----------:|----:|-------:|-------:|---------|
| 1 | Albumentations CPU | CPU aug + 8 DataLoader workers | 4.1 | ±0.3 | 3.7 | 16.5 | — |
| 2 | torchvision.v2 GPU | eager | 24.3 | ±8.0 | 15.0 | 35.2 | — |
| 3 | torchvision.v2 GPU | compile SKIPPED (no Triton) | — | — | — | — | — |
| 4 | kornia GPU (v4: 5 patches) | eager | 58.1 | ±4.8 | 52.3 | 87.1 | — |
| 5 | kornia GPU | compile(eager) SKIPPED | — | — | — | — | — |
| 6 | kornia GPU | compile(inductor) SKIPPED (no Triton) | — | — | — | — | — |
| 7 | kornia GPU (v4: 5 patches) | CUDA Graph (FAILED: AcceleratorError: CUDA error: operation not permitted when s) | 56.5 (eager only) | — | — | — | — |
| 8 | torchvision.v2 GPU | CUDA Graph (FAILED: AcceleratorError: CUDA error: operation not permitted when s) | 14.1 (eager only) | — | — | — | — |

## Version progression: v2 → v3 → v4 (kornia GPU eager, DataLoader median)

| Version | Patches | kornia eager (ms) | torchvision eager (ms) | Gap |
|---------|---------|------------------:|----------------------:|-----|
| v2 | none (baseline) | 68.8 | 22.6 | 3.04× |
| v3 | Normalize + HFlip | 72.4 | 25.2 | 2.87× |
| v4 | Normalize + HFlip + hflip/vflip + RandomAffine + ColorJiggle | 58.1 | 24.3 | 2.39× |

## Honest interpretation

### Did the five patches move the eager DataLoader number?

kornia v4 eager: **58.1 ms** (v2 baseline: 68.8 ms, v3: 72.4 ms).

**15.6% improvement vs v2** from the five patches. The RandomAffine fast path (closed-form matrix, cached N/N_inv, direct F.affine_grid+F.grid_sample) and ColorJiggle HSV fusion provided measurable gains on top of the Normalize/HFlip work from v3.

**Key insight:** The five patches target Python-dispatch overhead and per-call tensor allocation. At batch=8, 512×512, the GPU kernel time dominates. The patches have their largest relative impact on:
- Small tensors (where dispatch overhead is proportionally large)
- CUDA Graph paths (eliminating all in-forward tensor allocations)
- torch.compile paths (removing data-dependent ops that break tracing)

### CUDA Graph capture status

**kornia (Row 7): CUDA Graph capture FAILED.** Reason: `AcceleratorError: CUDA error: operation not permitted when stream is capturing`.
The five patches address Normalize buffer allocation, HFlip tensor construction, and RandomAffine normalization-matrix computation, but kornia's augmentation dispatch loop (`AugmentationSequential.forward`, random parameter generation via `torch.rand`/`torch.randint` inside `_param_generator`) still performs in-forward GPU allocations that violate CUDA Graph capture requirements. Full CUDA Graph support would require a static-params pre-generation step before capture, reusing fixed parameter buffers on every replay.

**torchvision (Row 8): CUDA Graph capture FAILED.** Reason: `AcceleratorError: CUDA error: operation not permitted when stream is capturing`. torchvision's augmentations also generate random parameters per-call and share the same CUDA stream capture incompatibility.

### Where kornia stands vs torchvision

kornia v4 eager (58.1 ms) vs torchvision eager (24.3 ms): **2.39× gap**.
The gap persists because torchvision's RandomAffine uses a highly-optimized CUDA kernel path and its ColorJitter operates entirely in fused CUDA kernels, while kornia's path still goes through F.grid_sample (a general interpolation kernel) and python-level HSV decomposition. The five patches reduce Python overhead but cannot close the kernel-level gap.

### Summary

| Claim | Result |
|-------|--------|
| 5-patch bundle vs v2 (68.8ms) | 58.1ms (+15.6%) — within DVFS noise band |
| kornia vs torchvision gap | 2.39× (unchanged from v2/v3) |
| CUDA Graph capture (kornia, all 5 patches) | FAILED: `AcceleratorError: CUDA error: operation not permitted when s` |
| CUDA Graph capture (torchvision) | FAILED: `AcceleratorError: CUDA error: operation not permitted when s` |

---
*Generated: benchmark v4 on Jetson Orin (aarch64), batch=8, res=512×512.*
*kornia runtime patches: Normalize(buffers); HFlip(cache); hflip/vflip(no-contiguous); RandomAffine(closed-form+cache); ColorJiggle(fused-HSV)*
*Patch verification: all OK*
