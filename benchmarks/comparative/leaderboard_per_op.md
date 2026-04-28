# Per-op eager benchmark -- kornia (patched) vs torchvision.v2 vs albumentations

**Date:** 2026-04-27
**GPU:** Orin
**PyTorch:** 2.8.0
**kornia:** 0.7.4 (5 eager patches applied)
**torchvision:** 0.23.0
**albumentations:** 2.0.8
**Input:** B=8, 512x512, fp32 GPU (kornia/TV); uint8 CPU (Alb)
**Timing:** 25 warmup + 100 CUDA-event iterations
**Total elapsed:** 252.2s

> Note: albumentations times are CPU-only (per-image loop over uint8 HWC numpy). GPU vs CPU comparisons are informational only -- not apples-to-apples.

## Geometric

| Op | kornia ms | torchvision ms | albumentations ms (CPU) | k/tv ratio | winner |
|---|---:|---:|---:|---:|---|
| HorizontalFlip | 16.87 | 0.92 | 2.32 *(CPU)* | 18.38x | torchvision |
| VerticalFlip | 17.40 | 0.92 | 0.96 *(CPU)* | 19.01x | torchvision |
| Rotation | 57.26 | 6.64 | 14.01 *(CPU)* | 8.63x | torchvision |
| Affine | 51.47 | 7.06 | 14.38 *(CPU)* | 7.29x | torchvision |
| ResizedCrop | 16.09 | 1.37 | 4.23 *(CPU)* | 11.74x | torchvision |
| CenterCrop | 11.84 | 0.15 | 0.32 *(CPU)* | 80.47x | torchvision |
| Resize | 9.43 | 1.39 | 4.75 *(CPU)* | 6.78x | torchvision |
| Perspective | 62.01 | 9.05 | 16.68 *(CPU)* | 6.85x | torchvision |

## Intensity (color / brightness)

| Op | kornia ms | torchvision ms | albumentations ms (CPU) | k/tv ratio | winner |
|---|---:|---:|---:|---:|---|
| ColorJitter | 52.29 | 23.19 | 34.06 *(CPU)* | 2.25x | torchvision |
| Brightness | 17.98 | 1.63 | 3.71 *(CPU)* | 11.00x | torchvision |
| Contrast | 21.04 | 2.62 | 3.32 *(CPU)* | 8.03x | torchvision |
| Saturation | 28.73 | 2.65 | 12.11 *(CPU)* | 10.85x | torchvision |
| Hue | 31.55 | 16.83 | 9.84 *(CPU)* | 1.87x | torchvision |
| Grayscale | 19.89 | 0.84 | 1.10 *(CPU)* | 23.78x | torchvision |
| Solarize | 18.14 | 1.74 | 3.60 *(CPU)* | 10.40x | torchvision |
| Posterize | 19.42 | 2.15 | 3.27 *(CPU)* | 9.01x | torchvision |
| Equalize | 53.54 | 7.93 | 17.02 *(CPU)* | 6.75x | torchvision |
| Invert | 8.05 | 0.81 | 0.98 *(CPU)* | 9.95x | torchvision |
| Sharpness | 20.35 | 5.98 | 25.02 *(CPU)* | 3.40x | torchvision |

## Intensity (blur / noise)

| Op | kornia ms | torchvision ms | albumentations ms (CPU) | k/tv ratio | winner |
|---|---:|---:|---:|---:|---|
| GaussianBlur | 32.89 | 4.60 | 21.72 *(CPU)* | 7.15x | torchvision |
| GaussianNoise | 12.17 | 2.94 | 65.23 *(CPU)* | 4.14x | torchvision |
| MotionBlur | 33.05 | SKIP | 15.09 *(CPU)* | -- | kornia-only |
| BoxBlur | 28.04 | SKIP | 6.28 *(CPU)* | -- | kornia-only |
| MedianBlur | 375.29 | SKIP | 1.67 *(CPU)* | -- | kornia-only |

## Erasing

| Op | kornia ms | torchvision ms | albumentations ms (CPU) | k/tv ratio | winner |
|---|---:|---:|---:|---:|---|
| RandomErasing | 38.38 | 2.88 | 3.36 *(CPU)* | 13.32x | torchvision |

## Normalize

| Op | kornia ms | torchvision ms | albumentations ms (CPU) | k/tv ratio | winner |
|---|---:|---:|---:|---:|---|
| Normalize | 9.98 | 2.56 | 8.75 *(CPU)* | 3.90x | torchvision |
| Denormalize | 10.16 | SKIP | SKIP | -- | kornia-only |

## Mix

| Op | kornia ms | torchvision ms | albumentations ms (CPU) | k/tv ratio | winner |
|---|---:|---:|---:|---:|---|
| MixUp | 34.20 | 2.92 | SKIP | 11.70x | torchvision |
| CutMix | 82.66 | 1.94 | SKIP | 42.55x | torchvision |
| Mosaic | 36.32 | SKIP | SKIP | -- | kornia-only |

## Kornia-only ops

| Op | kornia ms (GPU) |
|---|---:|
| RandomRain | 40.20 |
| RandomSnow | 42.77 |
| RandomChannelDropout | 7.35 |
| RandomChannelShuffle | 8.05 |
| RandomRGBShift | 9.78 |
| RandomPlanckianJitter | 11.43 |
| RandomCLAHE | 163.84 |

## Summary

- Total transforms attempted: 37
- kornia OK: 37
- torchvision OK: 25
- albumentations OK: 26

### kornia wins vs torchvision (k/tv < 0.9, lower is better for kornia)
  - (none)

### torchvision wins vs kornia (k/tv > 1.1)
  - CenterCrop: 80.47x (torchvision 7947% faster)
  - CutMix: 42.55x (torchvision 4155% faster)
  - Grayscale: 23.78x (torchvision 2278% faster)
  - VerticalFlip: 19.01x (torchvision 1801% faster)
  - HorizontalFlip: 18.38x (torchvision 1738% faster)
  - RandomErasing: 13.32x (torchvision 1232% faster)
  - ResizedCrop: 11.74x (torchvision 1074% faster)
  - MixUp: 11.70x (torchvision 1070% faster)
  - Brightness: 11.00x (torchvision 1000% faster)
  - Saturation: 10.85x (torchvision 985% faster)
  - Solarize: 10.40x (torchvision 940% faster)
  - Invert: 9.95x (torchvision 895% faster)
  - Posterize: 9.01x (torchvision 801% faster)
  - Rotation: 8.63x (torchvision 763% faster)
  - Contrast: 8.03x (torchvision 703% faster)
  - Affine: 7.29x (torchvision 629% faster)
  - GaussianBlur: 7.15x (torchvision 615% faster)
  - Perspective: 6.85x (torchvision 585% faster)
  - Resize: 6.78x (torchvision 578% faster)
  - Equalize: 6.75x (torchvision 575% faster)
  - GaussianNoise: 4.14x (torchvision 314% faster)
  - Normalize: 3.90x (torchvision 290% faster)
  - Sharpness: 3.40x (torchvision 240% faster)
  - ColorJitter: 2.25x (torchvision 125% faster)
  - Hue: 1.87x (torchvision 87% faster)

### Tied (within 10%)
  - (none)

> albumentations runs on CPU; times are not directly comparable to GPU kornia/torchvision times.
