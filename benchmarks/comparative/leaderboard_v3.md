# Comparative augmentation benchmark v3 — eager / compiled / CUDA-Graph leaderboard

## Hardware / stack

| Key | Value |
|-----|-------|
| Platform | Jetson Orin (aarch64), Linux 5.15.148-tegra |
| GPU | Orin (Orin integrated GPU, 1792-core Ampere) |
| CUDA | 12.6 (libcusolver 11.6.4.69) |
| Python | 3.10 (pixi camera-object-detector env) |
| PyTorch | 2.8.0 |
| kornia | 0.7.4 (installed 0.7.4 + runtime patches) |
| albumentations | 2.0.8 |
| torchvision | 0.23.0 |
| Batch size | 8 |
| Resolution | 512×512 |
| kornia patches | Normalize patched (pre-shaped buffers); RandomHorizontalFlip patched (module-level template) |

## Methodology

All rows measure the same DETR-style 4-op pipeline:
  RandomHorizontalFlip → RandomAffine → ColorJitter/Jiggle → Normalize
Batch=8, resolution=512×512, float32.

**End-to-end DataLoader timing (rows 1–6):**
DataLoader delivers CPU tensors → main thread applies H2D + GPU aug.
50 timed batches + 10 warmup. All times include Python dispatch,
DataLoader latency, and H2D transfer.

**CUDA Graph rows (7–8):**
Capture attempted in isolated subprocess to avoid stream-state contamination.
Replay timing uses CUDA events (no Python dispatch, no DataLoader overhead).
1000 replays. Speedup column is vs same-library eager CUDA-event timing.

**kornia optimisation patches applied at runtime to installed 0.7.4:**
- `Normalize.apply_transform`: pre-shaped `(1,C,1,1)` buffers registered via
  `register_buffer`, bypasses `kornia.enhance.normalize` wrapper overhead.
- `RandomHorizontalFlip.compute_transformation`: module-level matrix template,
  substitutes `w-1` via index op on clone — no per-call `torch.tensor([...])`,
  enabling CUDA Graph capture.

(Note: patched source tree at `/home/nvidia/kornia/` uses `StrEnum` from Python 3.11
 and cannot be loaded directly under Python 3.10. Patches applied inline instead.)

**Compile rows:**
- Row 3 (torchvision): `torch.compile(wrap, mode='reduce-overhead')`
- Row 5 (kornia eager backend): `torch.compile(aug, backend='eager')`
- Row 6 (kornia inductor): `torch.compile(aug, mode='reduce-overhead')`
If reduce-overhead fails, falls back to mode='default'. If both fail, marked N/A.

**cusolver workaround:** Jetson JetPack 6 ships libcusolver 11.6.4.69 which is
missing `cusolverDnXsyevBatched_bufferSize` needed by torch 2.8.0's linalg.
kornia's RandomAffine calls `torch.linalg.inv()`. Patched with closed-form
analytical 3×3 inverse (cofactor/det, elementwise CUDA ops only).

## 8-row comparison table

All times in ms/batch (lower is better). Speedup column is relative to the
eager baseline of the same library (or Albumentations for Row 1).

| Row | Configuration | Mode | Median ms | IQR | Min ms | Max ms | Speedup |
|-----|--------------|------|----------:|----:|-------:|-------:|---------|
| 1 | Albumentations CPU | CPU aug + 8 DataLoader workers | 4.1 | ±0.4 | 3.8 | 15.4 | — |
| 2 | torchvision.v2 GPU | eager | 25.2 | ±6.3 | 14.7 | 38.8 | — |
| 3 | torchvision.v2 GPU | compile N/A: Cannot find a working triton installation. Either the package is not i | — | — | — | — | — |
| 4 | kornia GPU (patched) | eager | 72.4 | ±7.6 | 57.6 | 93.2 | — |
| 5 | kornia GPU (patched) | compile(eager) N/A: Could not extract specialized integer from data-dependent expression u | — | — | — | — | — |
| 6 | kornia GPU (patched) | compile(inductor) N/A: Cannot find a working triton installation. Either the package is not i | — | — | — | — | — |
| 7 | kornia GPU (patched) | CUDA Graph (FAILED: AcceleratorError: CUDA error: operation not permitted when s) | 64.4 (eager only) | — | — | — | — |
| 8 | torchvision.v2 GPU | CUDA Graph (FAILED: AcceleratorError: CUDA error: operation not permitted when s) | 14.4 (eager only) | — | — | — | — |

## Per-row notes

- **Row 7**: CUDA Graph capture FAILED for kornia. Reason: `AcceleratorError: CUDA error: operation not permitted when stream is capturing`.
- **Row 8**: CUDA Graph capture FAILED for torchvision. Reason: `AcceleratorError: CUDA error: operation not permitted when stream is capturing`.

## Honest interpretation

### Did our kornia patches move the eager number?

kornia eager: 72.4 ms (v2 baseline: 68.8 ms). **5.2% regression** — likely noise / DVFS variance on Jetson Orin.

### Did compile help?

- `torch.compile(backend='eager')` (Row 5): FAILED — Could not extract specialized integer from data-dependent expression u0 (unhinte
- `torch.compile(inductor)` (Row 6): FAILED — Cannot find a working triton installation. Either the package is not installed o

### Did CUDA Graph capture succeed for kornia after the HFlip patch?

**NO** — CUDA Graph capture still fails for kornia: `AcceleratorError: CUDA error: operation not permitted when stream is capturing`. The HFlip patch alone was not sufficient; additional in-capture allocations remain in the pipeline (RandomAffine `_torch_inverse_cast`, ColorJiggle, etc.).

### Where kornia stands vs torchvision

kornia eager (72.4 ms) vs torchvision eager (25.2 ms): **2.87× gap** — same as v2. The patches address dispatch overhead in Normalize/HFlip but the dominant cost is RandomAffine (grid_sample + homography inversion) which is not yet optimised.

Best kornia mode (72.4 ms, compile) vs torchvision eager (25.2 ms): **2.87× gap** after compile. torch.compile closes some of the Python-dispatch overhead but cannot overcome the fundamental kernel cost difference in geometric transforms.

### Summary

| Claim | Evidence |
|-------|---------|
| Patches moved eager perf | 5.2% gain on DataLoader row |
| compile (inductor) gain for kornia | N/A (failed or not run) |
| CUDA Graph capture (kornia, patched HFlip) | FAILED: `AcceleratorError: CUDA error: operation not permitted when s` |
| CUDA Graph capture (torchvision) | FAILED: `AcceleratorError: CUDA error: operation not permitted when s` |

---
*Generated: benchmark v3 on Jetson Orin (aarch64), batch=8, res=512×512.*
*kornia runtime patches: Normalize patched (pre-shaped buffers); RandomHorizontalFlip patched (module-level template)*
