# Comparative augmentation benchmark v5 -- Path A lightweight forward fast path

## Hardware / stack

| Key | Value |
|-----|-------|
| Date | 2026-04-27 |
| Platform | Jetson Orin (aarch64), Linux 5.15.148-tegra |
| GPU | Orin (Orin integrated GPU, 1792-core Ampere) |
| CUDA | 12.6 (libcusolver 11.6.4.69) |
| Python | 3.10 (pixi camera-object-detector env) |
| PyTorch | 2.8.0 |
| kornia | 0.7.4 (installed 0.7.4 + v4 + Path A runtime patches) |
| torchvision | 0.23.0 |
| albumentations | 2.0.8 |
| Batch size | 8 |
| Resolution | 512x512 |
| Timing | 25 warmup + 100 CUDA-event runs |
| Wall time | 39.2s |

## Path A monkey-patch verification

v4 patch status: `Normalize(buffers); Denormalize(buffers); HFlip(cache); hflip/vflip(no-contiguous); RandomAffine(closed-form+cache); ColorJiggle(fused-HSV)`

Per-class fast-path opt-in installation:

| Component | Install status |
|-----------|----------------|
| base.forward gate | OK |
| RandomHorizontalFlip | OK |
| RandomVerticalFlip | OK |
| CenterCrop | OK |
| RandomGrayscale | OK |
| RandomInvert | OK |
| RandomSolarize | OK |
| RandomPosterize | OK |
| Normalize | OK |
| Denormalize | OK |

Numerical equivalence (fast-path output vs standard-path output, atol=1e-5):

| Transform | Equivalent? | max abs diff |
|-----------|-------------|--------------|
| RandomHorizontalFlip | YES | 0.00e+00 |
| RandomVerticalFlip | YES | 0.00e+00 |
| CenterCrop | YES | 0.00e+00 |
| RandomGrayscale | YES | 0.00e+00 |
| RandomInvert | YES | 0.00e+00 |
| RandomSolarize | YES | 0.00e+00 |
| RandomPosterize | YES | 0.00e+00 |
| Normalize | YES | 0.00e+00 |
| Denormalize | YES | 0.00e+00 |

## Scope A -- per-op CUDA event timing (10 transforms)

Input: pre-resident GPU tensor B=8, 3, 512, 512, fp32. 25 warmup + 100 CUDA-event iterations. Median ms.

| Transform | k fast (ms) | k std (ms) | tv (ms) | fast / std speedup | fast / tv ratio | std / tv ratio |
|-----------|------------:|-----------:|--------:|-------------------:|----------------:|---------------:|
| CenterCrop | 2.301 | 11.826 | 0.147 | 5.14x | 15.65x | 80.41x |
| RandomHorizontalFlip | 6.577 | 6.147 | 1.124 | 0.93x | 5.85x | 5.47x |
| RandomVerticalFlip | 11.222 | 8.089 | 0.777 | 0.72x | 14.44x | 10.41x |
| RandomGrayscale | 7.351 | 8.876 | 0.810 | 1.21x | 9.07x | 10.96x |
| RandomInvert | 4.860 | 5.835 | 0.609 | 1.20x | 7.98x | 9.58x |
| RandomSolarize | 14.583 | 18.246 | 1.745 | 1.25x | 8.36x | 10.46x |
| RandomPosterize | 14.037 | 15.763 | 2.154 | 1.12x | 6.52x | 7.32x |
| Normalize | 5.665 | 6.563 | 2.238 | 1.16x | 2.53x | 2.93x |
| Denormalize | 5.708 | 7.671 | n/a | 1.34x | n/a | n/a |

## Scope B -- DETR-style 4-op pipeline (kornia)

Pipeline: HFlip(p=0.5) -> Affine(deg=15, t=0.1, s=0.8-1.2) -> ColorJiggle(0.2, 0.2, 0.2) -> Normalize(ImageNet)

| Configuration | Median ms | IQR | Min ms | Max ms | Delta vs v4 (58.1ms) |
|---------------|----------:|----:|-------:|-------:|---------------------:|
| Fast path ENABLED (Path A active) | 48.82 | 1.82 | 46.16 | 54.35 | -9.28ms |
| Fast path DISABLED (forces v4 standard chain) | 49.30 | 1.99 | 45.45 | 60.51 | -8.80ms |

## Honest interpretation

### Did the CenterCrop CPU 33x speedup translate to CUDA?

Kornia CenterCrop on CUDA: fast=2.301ms, std=11.826ms, speedup=5.14x.
vs torchvision CenterCrop (0.147ms): fast/tv=15.65x, std/tv=80.41x.
**Verdict: the CUDA speedup is well below the CPU 33x projection.** On CUDA, CenterCrop's standard path costs 11.83ms while the fast path costs 2.30ms (5.14x). The fast path eliminates parameter generation and the `crop_by_indices` wrapper but still dispatches a contiguous slice + transform-matrix construction every call.  In absolute ms the win is 9.52ms / call -- meaningful on a per-call basis but much smaller than the CPU multiplicative win because GPU kernels are fundamentally fast and the standard path's overhead-per-batch is amortized over a 512x512 batch.

### Per-op REGRESSIONS (fast path slower than standard)

- **RandomVerticalFlip**: fast=11.22ms, std=8.09ms (0.72x). The fast-path overhead (per-call `torch.as_tensor`, eager param generation, transform_matrix construction) exceeds the dispatch savings on this op at B=8, 512x512.  These are the cases where the Path A opt-in is a *no-op* or a minor net loss on this hardware.
- **RandomHorizontalFlip**: fast=6.58ms, std=6.15ms (0.93x). The fast-path overhead (per-call `torch.as_tensor`, eager param generation, transform_matrix construction) exceeds the dispatch savings on this op at B=8, 512x512.  These are the cases where the Path A opt-in is a *no-op* or a minor net loss on this hardware.

### Per-op fast vs standard speedups (top 3)
- **CenterCrop**: 5.14x faster than the standard path
- **Denormalize**: 1.34x faster than the standard path
- **RandomSolarize**: 1.25x faster than the standard path

### Where does kornia fast-path stand vs torchvision?

Ratio = kornia_time / torchvision_time. <1.0x means kornia faster.

| Transform | std / tv (before Path A) | fast / tv (after Path A) | gap closed |
|-----------|-------------------------:|-------------------------:|-----------:|
| CenterCrop | 80.41x | 15.65x | +64.76x |
| RandomHorizontalFlip | 5.47x | 5.85x | -0.38x |
| RandomVerticalFlip | 10.41x | 14.44x | -4.03x |
| RandomGrayscale | 10.96x | 9.07x | +1.88x |
| RandomInvert | 9.58x | 7.98x | +1.60x |
| RandomSolarize | 10.46x | 8.36x | +2.10x |
| RandomPosterize | 7.32x | 6.52x | +0.80x |
| Normalize | 2.93x | 2.53x | +0.40x |
| Denormalize | n/a | n/a | n/a |

### DETR pipeline -- did fast-path drop below v4's 58.1ms?

vs the v4 reference run (58.1ms) the v5 fast-path run is 9.28ms FASTER. **However, this cross-run delta is dominated by DVFS state on the Jetson Orin and cannot be attributed to Path A alone:** the fast-OFF run (which forces the v4-equivalent standard chain) also clocks in at 49.30ms in this v5 session, i.e. roughly equally far from the v4 reference. The Orin governor drifted between the two sessions.

**The within-bench fast-on vs fast-off delta is -0.48ms (noise band).** In this DETR pipeline only HFlip and Normalize are eligible for the fast path; Affine and ColorJiggle still go through the full forward chain. With both p=0.5 (HFlip) and the DETR mix dominated by `F.grid_sample` and the HSV roundtrip, Path A's per-call dispatch savings round to zero against the GPU kernel budget.

---
*Generated: benchmark v5 on Orin, batch=8, res=512x512.*
