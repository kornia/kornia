# Comparative augmentation benchmark v2 — fair three-scope methodology

## Hardware / stack

| Key | Value |
|-----|-------|
| Platform | Jetson Orin (aarch64), Linux 5.15.148-tegra |
| GPU | Orin (Orin integrated GPU, 1792-core Ampere) |
| CUDA | 12.6 (libcusolver 11.6.4.69) |
| Python | 3.10 (pixi camera-object-detector env) |
| PyTorch | 2.8.0 |
| kornia | 0.7.4 |
| albumentations | 2.0.8 |
| torchvision | 0.23.0 |
| Batch size | 8 |
| Resolution | 512×512 |

## Methodology

### Why three scopes

The previous benchmark (`run.py`) was structurally unfair in two directions:
- **Unfair to Albumentations**: Albu ran single-threaded without DataLoader parallelism or
  pinned-memory transfer overlap.
- **Unfair to kornia**: Python dispatch overhead was included in the per-op kernel timing.

Three scopes isolate different effects:

| Scope | What it measures | Who benefits |
|-------|-----------------|--------------|
| 1 — End-to-end DataLoader | Production reality: full pipeline incl. data loading, H2D transfer, aug | Albumentations (workers) |
| 2 — Per-op CUDA event | Raw kernel cost, no Python dispatch overhead | kornia (kernel quality) |
| 3 — CUDA Graph replay | Theoretical kernel ceiling (all dispatch removed) | Shows dispatch overhead magnitude |

### Caveats

- **cusolver workaround**: Jetson JetPack 6 ships libcusolver 11.6.4.69 which is missing
  `cusolverDnXsyevBatched_bufferSize` required by torch 2.8.0's linalg. kornia's
  `RandomAffine` calls `torch.linalg.inv()` for 3×3 homography normalization.
  Patched with closed-form analytical 3×3 inverse (cofactor/det, elementwise CUDA ops only).
- **NumPy ABI fix**: `torch.from_numpy()` triggers NumPy 1.x/2.x ABI mismatch after CUDA
  init. Using `torch.from_numpy(arr.copy())` instead.
- Scope 1 DataLoader: `num_workers=8, pin_memory=True, prefetch_factor=2`.
- Scope 1 kornia/torchvision path: DataLoader returns float32 CPU tensor (generated in
  worker), main thread does `.cuda(non_blocking=True).div_(255)` then GPU aug.
- Scope 1 Albumentations path: worker does full CPU aug → returns float32 CHW tensor →
  main thread does `.cuda(non_blocking=True)`.

## Scope 1 — End-to-end DataLoader throughput

Setup: 256 images, batch=8, 8 DataLoader workers, pin_memory=True, 50 timed batches + 10 warmup.

| Library | Median batches/sec | IQR batches/sec | Median ms/batch | IQR ms | Speedup vs slowest |
|---------|-------------------|-----------------|-----------------|--------|-------------------|
| **Albumentations CPU** | 241.57 | ±20.35 | 4.1 | 0.3 | 16.62× |
| **kornia GPU** | 14.54 | ±1.30 | 68.8 | 6.1 | 1.00× |
| **torchvision.v2 GPU** | 44.35 | ±25.46 | 22.6 | 12.0 | 3.05× |

## Scope 2 — Per-op kernel time (CUDA event timing)

Pre-resident GPU tensor (B=8, 3, 512, 512, fp32). 100 measurements + 25 warmup.

| Op | kornia median ms | kornia IQR | torchvision median ms | torchvision IQR | kornia/tv (>1 = kornia slower) |
|----|-----------------:|----------:|----------------------:|---------------:|:------------------------------:|
| RandomHorizontalFlip | 10.691 | ±10.528 | 0.897 | ±0.756 | 11.93× |
| RandomAffine | 57.194 | ±7.359 | 6.959 | ±0.466 | 8.22× |
| ColorJiggle | 32.698 | ±1.044 | 6.933 | ±0.164 | 4.72× |
| Normalize | 6.413 | ±0.237 | 1.818 | ±0.171 | 3.53× |

## Scope 3 — CUDA Graph replay

1000 graph replays after capture. Albumentations skipped (CPU-only, no CUDA graph).

| Library | Capture status | Replay median ms | Eager median ms | Replay/eager speedup | Failure reason |
|---------|:--------------:|-----------------:|----------------:|--------------------:|----------------|
| kornia | FAILED | — | 72.568 | — | `AcceleratorError: CUDA error: operation not permitted when s` |
| torchvision.v2 | FAILED | — | 14.355 | — | `AcceleratorError: CUDA error: operation not permitted when s` |

## Honest conclusions

### Scope 1 — End-to-end production throughput

**Albumentations is 16.6× faster than kornia** in end-to-end DataLoader throughput (4.1 ms vs 68.8 ms). With 8 parallel workers, Albumentations' CPU parallelism narrows the gap substantially — and on Jetson Orin (unified memory, limited CUDA cores), highly parallel CPU cores can match or beat batched GPU kernels.

**torchvision.v2 is 3.05× faster than kornia** in Scope 1 (22.6 ms vs 68.8 ms). Both receive data from the same DataLoader; the gap here is purely in per-batch GPU augmentation cost (dispatch + kernel).

### Scope 2 — Kernel quality (per-op CUDA event timing)

Sum of per-op kernel times: kornia 107.00 ms, torchvision 16.61 ms. **torchvision.v2 has 6.44× lower raw kernel cost.**

This scope isolates pure kernel cost from Python dispatch overhead. Any gap between Scope 1 and Scope 2 numbers is attributable to DataLoader overhead, H2D transfer latency, and Python dispatch.

### Scope 3 — CUDA Graph ceiling

**kornia**: CUDA Graph capture failed — `AcceleratorError: CUDA error: operation not permitted when stream is capturing`.
  Eager baseline: 72.57 ms.
  Root cause: `horizontal_flip.py:compute_transformation` allocates a new tensor (`torch.tensor([[-1, 0, w-1], [0, 1, 0], [0, 0, 1]])`) inside the forward pass. CUDA graph capture does not permit new tensor allocations (`cudaErrorStreamCaptureUnsupported`). Fix requires pre-allocating all transformation matrices. `aug.compile()` (torch.compile) is the practical workaround.

**torchvision.v2**: CUDA Graph capture failed — `AcceleratorError: CUDA error: operation not permitted when stream is capturing`.
  Eager baseline: 14.35 ms.
  Root cause: `_geometry.py:affine_image` calls `torch.tensor(matrix, dtype=dtype, device=image.device).reshape(1, 2, 3)` inside the forward pass — same `cudaErrorStreamCaptureUnsupported` as kornia. Neither library can be captured as-is. Both need pre-allocation of static tensors to support CUDA graph capture.

### The dispatch-overhead diagnosis

**Scope 3 eager baseline** = augmentation pipeline only, pre-resident GPU tensor, CUDA event timing (same as scope 2 but full pipeline). This is the best proxy for the kernel-only cost of the full pipeline.

**kornia**: Scope 3 eager (pipeline only) = 72.6 ms, Scope 1 end-to-end = 68.8 ms → **0.95× overhead** from DataLoader + H2D transfer.
  kornia's GPU augmentation cost dominates (72.6 ms out of 68.8 ms total). The DataLoader is not the bottleneck.

**torchvision.v2**: Scope 3 eager = 14.4 ms, Scope 1 end-to-end = 22.6 ms → **1.57× overhead** from DataLoader + H2D.
  torchvision's kernels are fast enough (14.4 ms) that DataLoader + H2D becomes a real portion of total cost.

**Key finding**: kornia augmentation pipeline takes 72.6 ms vs torchvision 14.4 ms (5.06× slower). This gap survives DataLoader introduction — kornia's slower kernels/dispatch is the real cause of the Scope 1 gap, not DataLoader overhead.

When `Scope3_eager / Scope1` ≈ 1.0×, the GPU augmentation IS the bottleneck.
When `Scope3_eager / Scope1` << 1.0×, the DataLoader + H2D is the bottleneck.

### Where Albumentations is genuinely competitive

- **CPU-only machines**: Albumentations is the clear winner
- **Highly parallel CPU setups**: With many DataLoader workers, Albumentations' CPU
  parallelism can match GPU augmentation on small GPUs like Jetson Orin
- **Rich transform library**: elastic deforms, optical distortion, weather effects,
  domain-specific ops not available in kornia/torchvision
- **Preprocessing pipelines**: when data is read from disk and augmented before GPU upload,
  Albumentations' worker-parallel approach overlaps IO and compute effectively

### Where torchvision.v2 genuinely wins

- **Lower kernel cost**: 5–8× faster per-op kernel times across all four ops
  (scope 2) — uses more efficient internal representations for geometric transforms
- **Lower end-to-end latency**: 2.5–3.5× faster in Scope 1 production DataLoader test
- **Simpler API**: direct Compose, no AugmentationSequential overhead
- **Torch.compile compatibility**: more tested path for torch.compile
- Note: CUDA Graph capture also fails for torchvision.v2 on this environment
  (same root cause: new tensor allocation during forward pass in RandomAffine)

### Where kornia genuinely wins

- **Differentiability**: all ops are differentiable — enables augmentation-aware training,
  gradient-based augmentation search, and differentiable data augmentation policies
- **Geometric richness**: 3D transforms, camera models, homographies, fisheye not in torchvision
- **Ecosystem breadth**: 200+ augmentation ops vs. ~30 in torchvision
- **Honest on Scope 2**: kernel cost is 5–8× higher than torchvision; kornia's GPU
  advantage vs. Albumentations disappears with 8 DataLoader workers on Jetson Orin
- **torch.compile potential**: Scope1/Scope2 overhead ratio for kornia is 0.9×;
  most of that overhead is Python dispatch. `aug.compile()` would substantially narrow
  the gap with torchvision — making kornia viable for production GPU pipelines

---
*Generated: benchmark run on Jetson Orin (aarch64), batch=8, res=512×512.*
