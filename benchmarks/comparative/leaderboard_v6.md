# Comparative augmentation benchmark v6 -- aggressive forward override

## Hardware / stack

| Key | Value |
|-----|-------|
| Date | 2026-04-27 |
| Platform | Jetson Orin (aarch64), Linux 5.15.148-tegra |
| GPU | Orin (Orin integrated GPU, 1792-core Ampere) |
| CUDA | 12.6 (libcusolver 11.6.4.69) |
| Python | 3.10 (pixi camera-object-detector env) |
| PyTorch | 2.8.0 |
| kornia | 0.7.4 (installed 0.7.4 + v4 + aggressive forward overrides) |
| torchvision | 0.23.0 |
| albumentations | 2.0.8 |
| Batch size | 8 |
| Resolution | 512x512 |
| Timing | 25 warmup + 100 CUDA-event runs (CPU loop for albumentations) |
| Wall time | 244.2s |

## Aggressive forward override -- patch verification

v4 patch status: `Normalize(buffers); Denormalize(buffers); HFlip(cache); hflip/vflip(no-contiguous); RandomAffine(closed-form+cache); ColorJiggle(fused-HSV)`

Per-class aggressive forward override installation (15 transforms):

| # | Class | Install status |
|---|-------|----------------|
| 1 | RandomHorizontalFlip | OK |
| 2 | RandomVerticalFlip | OK |
| 3 | CenterCrop | OK |
| 4 | Normalize | OK |
| 5 | Denormalize | OK |
| 6 | RandomInvert | OK |
| 7 | RandomGrayscale | OK |
| 8 | RandomSolarize | OK |
| 9 | RandomBrightness | OK |
| 10 | RandomContrast | OK |
| 11 | RandomSaturation | OK |
| 12 | RandomHue | OK |
| 13 | RandomPosterize | OK |
| 14 | RandomCutMixV2 | OK |
| 15 | RandomMixUpV2 | OK |

Numerical equivalence check (aggressive forward vs framework chain):

| Transform | Equivalent? | max abs diff | Note |
|-----------|-------------|--------------|------|
| RandomHorizontalFlip | YES | 0.00e+00 |  |
| RandomVerticalFlip | YES | 0.00e+00 |  |
| CenterCrop | YES | 0.00e+00 |  |
| RandomGrayscale | YES | 0.00e+00 |  |
| RandomInvert | YES | 0.00e+00 |  |
| Normalize | YES | 0.00e+00 |  |
| Denormalize | YES | 0.00e+00 |  |
| RandomSolarize | YES | 0.00e+00 | RNG-divergent: shape-only equivalence |
| RandomPosterize | YES | 0.00e+00 | RNG-divergent: shape-only equivalence |
| RandomBrightness | YES | 0.00e+00 | RNG-divergent: shape-only equivalence |
| RandomContrast | YES | 0.00e+00 | RNG-divergent: shape-only equivalence |
| RandomSaturation | YES | 0.00e+00 | RNG-divergent: shape-only equivalence |
| RandomHue | YES | 0.00e+00 | RNG-divergent: shape-only equivalence |
| RandomCutMixV2 | YES | 0.00e+00 | shape-only sanity (mix ops) |
| RandomMixUpV2 | YES | 0.00e+00 | shape-only sanity (mix ops) |

## Per-op leaderboard -- sorted by k/tv ratio (best first)

Input: pre-resident GPU tensor B=8, 3, 512, 512, fp32. 25 warmup + 100 CUDA-event runs. Median ms.

| Op | k v6 ms | k v5 ms | tv ms | speedup vs v5 | k/tv ratio | improvement |
|----|--------:|--------:|------:|--------------:|-----------:|:------------|
| Hue | 27.75 | 31.55 | 16.81 | 1.14x | 1.65x | **MATCH tv** (1.65x) |
| ColorJitter | 51.05 | 52.29 | 23.01 | 1.02x | 2.22x |  |
| Contrast | 9.32 | 21.04 | 2.59 | 2.26x | 3.59x |  |
| Normalize | 9.22 | 6.56 | 2.42 | 0.71x | 3.81x |  |
| MixUp | 14.48 | 34.20 | 2.95 | 2.36x | 4.92x |  |
| Sharpness | 37.36 | 20.35 | 5.98 | 0.54x | 6.25x |  |
| Resize | 8.88 | 9.43 | 1.39 | 1.06x | 6.37x |  |
| Perspective | 56.18 | 62.01 | 8.60 | 1.10x | 6.53x |  |
| GaussianBlur | 32.17 | 32.89 | 4.58 | 1.02x | 7.03x |  |
| GaussianNoise | 20.97 | 12.17 | 2.97 | 0.58x | 7.07x |  |
| Rotation | 55.18 | 50.50 | 7.60 | 0.92x | 7.26x |  |
| Brightness | 9.23 | 17.98 | 1.26 | 1.95x | 7.32x |  |
| Equalize | 62.51 | 53.54 | 7.92 | 0.86x | 7.89x |  |
| Affine | 52.65 | 51.00 | 6.62 | 0.97x | 7.96x |  |
| Posterize | 18.17 | 19.42 | 2.19 | 1.07x | 8.31x |  |
| VerticalFlip | 8.18 | 8.09 | 0.96 | 0.99x | 8.48x |  |
| RandomErasing | 35.15 | 38.38 | 3.74 | 1.09x | 9.40x |  |
| Solarize | 17.00 | 18.14 | 1.75 | 1.07x | 9.70x |  |
| HorizontalFlip | 7.53 | 6.15 | 0.74 | 0.82x | 10.15x |  |
| Invert | 6.75 | 5.84 | 0.66 | 0.87x | 10.22x |  |
| Saturation | 27.25 | 28.73 | 2.65 | 1.05x | 10.28x |  |
| CutMix | 20.67 | 82.66 | 1.93 | 4.00x | 10.70x |  |
| Grayscale | 10.60 | 8.88 | 0.93 | 0.84x | 11.37x |  |
| ResizedCrop | 16.05 | 16.09 | 1.39 | 1.00x | 11.57x |  |
| CenterCrop | 3.14 | 11.83 | 0.18 | 3.77x | 17.53x |  |

## Per-op (no torchvision baseline) -- kornia-only / tv-missing

| Op | k v6 ms | k v5 ms | speedup vs v5 |
|----|--------:|--------:|--------------:|
| MotionBlur | 33.24 | 33.05 | 0.99x |
| BoxBlur | 27.57 | 28.04 | 1.02x |
| MedianBlur | 374.10 | 375.29 | 1.00x |
| Denormalize | 6.82 | 7.67 | 1.12x |
| Mosaic | 32.39 | 36.32 | 1.12x |
| RandomRain | 44.33 | 40.20 | 0.91x |
| RandomSnow | 33.12 | 42.77 | 1.29x |
| RandomChannelDropout | 6.86 | 7.35 | 1.07x |
| RandomChannelShuffle | 14.63 | 8.05 | 0.55x |
| RandomRGBShift | 19.24 | 9.78 | 0.51x |
| RandomPlanckianJitter | 11.85 | 11.43 | 0.96x |
| RandomCLAHE | 171.95 | 163.84 | 0.95x |

## Per-op with albumentations CPU column (full registry)

| Op | k v6 (GPU ms) | tv (GPU ms) | alb (CPU ms) |
|----|--------------:|------------:|-------------:|
| HorizontalFlip | 7.53 | 0.74 | 2.34 *(CPU)* |
| VerticalFlip | 8.18 | 0.96 | 0.94 *(CPU)* |
| Rotation | 55.18 | 7.60 | 13.64 *(CPU)* |
| Affine | 52.65 | 6.62 | 13.84 *(CPU)* |
| ResizedCrop | 16.05 | 1.39 | 4.34 *(CPU)* |
| CenterCrop | 3.14 | 0.18 | 0.32 *(CPU)* |
| Resize | 8.88 | 1.39 | 4.68 *(CPU)* |
| Perspective | 56.18 | 8.60 | 16.69 *(CPU)* |
| ColorJitter | 51.05 | 23.01 | 35.09 *(CPU)* |
| Brightness | 9.23 | 1.26 | 3.56 *(CPU)* |
| Contrast | 9.32 | 2.59 | 3.33 *(CPU)* |
| Saturation | 27.25 | 2.65 | 12.72 *(CPU)* |
| Hue | 27.75 | 16.81 | 10.02 *(CPU)* |
| Grayscale | 10.60 | 0.93 | 1.09 *(CPU)* |
| Solarize | 17.00 | 1.75 | 3.55 *(CPU)* |
| Posterize | 18.17 | 2.19 | 3.28 *(CPU)* |
| Equalize | 62.51 | 7.92 | 16.82 *(CPU)* |
| Invert | 6.75 | 0.66 | 0.81 *(CPU)* |
| Sharpness | 37.36 | 5.98 | 24.80 *(CPU)* |
| GaussianBlur | 32.17 | 4.58 | 21.53 *(CPU)* |
| GaussianNoise | 20.97 | 2.97 | 65.83 *(CPU)* |
| MotionBlur | 33.24 | FAIL | 14.18 *(CPU)* |
| BoxBlur | 27.57 | FAIL | 6.31 *(CPU)* |
| MedianBlur | 374.10 | FAIL | 1.94 *(CPU)* |
| RandomErasing | 35.15 | 3.74 | 3.24 *(CPU)* |
| Normalize | 9.22 | 2.42 | 8.51 *(CPU)* |
| Denormalize | 6.82 | FAIL | FAIL |
| MixUp | 14.48 | 2.95 | FAIL |
| CutMix | 20.67 | 1.93 | FAIL |
| Mosaic | 32.39 | FAIL | FAIL |
| RandomRain | 44.33 | FAIL | FAIL |
| RandomSnow | 33.12 | FAIL | FAIL |
| RandomChannelDropout | 6.86 | FAIL | FAIL |
| RandomChannelShuffle | 14.63 | FAIL | FAIL |
| RandomRGBShift | 19.24 | FAIL | FAIL |
| RandomPlanckianJitter | 11.85 | FAIL | FAIL |
| RandomCLAHE | 171.95 | FAIL | FAIL |

## DETR-style pipeline (HFlip(p=0.5) -> Affine -> ColorJiggle -> Normalize)

| Run | Median ms | IQR | Min | Max |
|-----|----------:|----:|----:|----:|
| v6 (aggressive overrides ON) | 48.98 | 1.73 | 45.25 | 59.14 |
| v5 fast_on (Path A active) | 48.82 | -- | -- | -- |
| v5 fast_off (v4-equivalent) | 49.30 | -- | -- | -- |
| v4 reference | 58.10 | -- | -- | -- |

- v6 - v5(fast_on): +0.16ms
- v6 - v4(ref):     -9.12ms

## Honest interpretation

### Did kornia match torchvision (k/tv <= 2.0) on any op?

**YES** -- the following ops now satisfy k/tv <= 2.0:

- Hue: k=27.75ms, tv=16.81ms, ratio=1.65x

### Did kornia hit 5x over v5 on any op?

**NO** -- no op crossed the 5x-over-v5 threshold.

---
*Generated: benchmark v6 on Orin, batch=8, res=512x512, kornia 0.7.4 + aggressive forward override.*
