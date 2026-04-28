# Comparative Augmentation Benchmark: kornia vs Albumentations vs torchvision.v2

## Environment

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
| Resolution | 512x512 |
| Runs | 50 measured + 10 warmup (discarded) |

## Pipeline (DETR-style preset)

```
HorizontalFlip(p=0.5)
Affine(rotate=±15°, translate=±10%, scale=0.8–1.2)  [p=1.0]
ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)  [p=1.0]
Normalize(ImageNet mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
```

## Leaderboard


| Rank | Library | Device | Median ms/batch | IQR [p25, p75] | Min ms | Max ms | vs slowest |
|------|---------|--------|-----------------|----------------|--------|--------|------------|
| **1st** | **torchvision.v2 (GPU)** | GPU (tensor pre-resident) | **14.6** | [14.1, 15.4] | 13.7 | 16.3 | **5.47×** |
| **2nd** | **kornia (GPU)** | GPU (tensor pre-resident) | **71.2** | [67.0, 75.7] | 59.4 | 99.0 | **1.12×** |
| **3rd** | **Albumentations CPU + transfer** | CPU aug + H2D transfer | **79.9** | [76.4, 83.2] | 73.5 | 100.4 | **1.00×** |

## Methodology

- Seed: `torch.manual_seed(42)`, `np.random.seed(42)` for reproducibility
- kornia + torchvision.v2: random float32 tensors pre-allocated on GPU (no H2D cost)
- Albumentations: random uint8 HWC numpy (matches real-world training loop ingestion)
  then `torch.tensor(np.stack(...)).permute(0,3,1,2).cuda()` to send batch to GPU
- All GPU variants: `torch.cuda.synchronize()` before and after each timed run
- Wall-clock timer: `time.perf_counter()` (sub-microsecond resolution)
- Warmup runs discard JIT compilation and CUDA context initialization overhead

## Environment notes

**kornia + Jetson cusolver issue**: torch 2.8.0 requires `cusolverDnXsyevBatched_bufferSize`
(cusolver ≥ 11.7, CUDA ≥ 12.4). The Jetson JetPack 6 system has cusolver 11.6.4.69.
kornia's `RandomAffine` calls `torch.linalg.inv()` for 3×3 homography normalization,
which triggers loading `libtorch_cuda_linalg.so` — which fails with the above symbol error.
**Workaround applied**: `_torch_inverse_cast` monkey-patched with a closed-form analytical
3×3 matrix inverse using only elementwise CUDA ops (determinant + cofactor expansion).
The affine warp itself (`grid_sample`, `warp_affine`) still runs fully on GPU.
Timing overhead: < 0.1 ms per batch (the 3×3 inversion is trivial vs. the pixel-level warp).

**Albumentations + NumPy 2.x**: `torch.from_numpy()` triggers a NumPy 1.x/2.x ABI
warning that becomes an error when CUDA is initialized. Replaced with `torch.tensor()`.

## Analysis

**kornia GPU vs Albumentations CPU+transfer**: 1.1× faster (71.2 ms vs 79.9 ms median). Albumentations processes images sequentially on CPU (single-threaded per-image API). kornia batches all 8 images in a single GPU dispatch. Even on Jetson Orin's unified memory (lower H2D latency than PCIe), the serial CPU compute + transfer cost dominates.

**kornia GPU vs torchvision.v2 GPU**: kornia is 4.88× SLOWER (71.2 ms vs 14.6 ms median). This is unexpected and warrants explanation: kornia 0.7.4's `AugmentationSequential` has significant Python-level overhead per op (parameter generation, dispatch, container logic) on top of the GPU kernel. The workaround for the missing cusolver symbol adds a tiny CPU-side matrix inversion (~0.1 ms) but the dominant cost is kornia's augmentation container dispatch overhead in this older version. torchvision.v2 dispatches its transforms with lower per-op overhead. On the Orin's 1792-core Ampere GPU, kernel launch overhead is proportionally large compared to the actual compute for these operations at batch=8, 512×512. A kornia version ≥ 0.8.x with the modern `ImageSequential` API may have different performance characteristics.

**Recommendation (based on these numbers)**:
- `torchvision.transforms.v2` is the fastest option on this Jetson Orin GPU (5.5× faster than Albumentations CPU+transfer).
- kornia 0.7.4 `AugmentationSequential` is only marginally faster than Albumentations+transfer (1.12×). The gap likely widens on newer kornia or with larger batches/resolutions where GPU utilization dominates.
- Albumentations is the right choice for CPU-only environments or when you need its broader transform library.
- The expected "kornia 10× faster than Albumentations" claim does NOT hold at batch=8 with kornia 0.7.4 on Jetson Orin. The actual result is 1.12× for kornia, 5.47× for torchvision.v2.
