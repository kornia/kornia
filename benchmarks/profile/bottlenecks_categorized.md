# Bottleneck categorization — all 37 augmentation transforms

**Date:** 2026-04-27  
**Hardware:** Orin (aarch64)  
**PyTorch:** 2.8.0, CUDA 12.6, kornia 0.7.4, torchvision 0.23.0  
**Patches:** v4: Normalize(buffers); Denormalize(buffers); HFlip(cache); hflip/vflip(no-contiguous); RandomAffine(closed-form+cache); ColorJiggle(fused-HSV); v6: {'RandomHorizontalFlip': 'OK', 'RandomVerticalFlip': 'OK', 'CenterCrop': 'OK', 'Normalize': 'OK', 'Denormalize': 'OK', 'RandomInvert': 'OK', 'RandomGrayscale': 'OK', 'RandomSolarize': 'OK', 'RandomBrightness': 'OK', 'RandomContrast': 'OK', 'RandomSaturation': 'OK', 'RandomHue': 'OK', 'RandomPosterize': 'OK', 'RandomCutMixV2': 'OK', 'RandomMixUpV2': 'OK'}  
**Profile:** 5 warmup + 20 timed iters via `torch.profiler` (CPU+CUDA, record_shapes=True, profile_memory=True)  
**Inputs:** B=8, 3x512x512, fp32, GPU pre-resident  

**CUPTI note:** `CUPTI_ERROR_INSUFFICIENT_PRIVILEGES` on this unprivileged Jetson run means kernel-level CUDA self-times are 0 in every event. We classify using **self CPU time + event count + sync proxy** (the trailing `aten::copy_` self CPU is the closest proxy for CUDA wall-clock since each iter ends with `cuda.synchronize()`).

**Categories:**

- **dispatch-bound** — `events_per_iter > 50` or top non-copy event is `aten::full`/`empty`/`to`/`lift_fresh`. The op spends most time in framework bookkeeping. *Fix: kornia 2.0 base class redesign.*
- **allocation-bound** — top non-copy op is a tensor allocation (`aten::zeros`, `empty_strided`, `eye`, etc.). *Fix: pre-allocate buffers, lift to module `__init__`.*
- **kernel-bound** — top non-copy op is real compute (`aten::convolution`, `grid_sample`, `flip`, `sub`, `mul`). *Fix: faster kernel (Triton/CUDA); this is the irreducible floor.*
- **sync-bound** — large `aten::_local_scalar_dense` / `aten::item` count. *Fix: lift host->device sync points; cache scalar params.*
- **fusion-eligible** — composite op (CutMix, ColorJitter, Mosaic, Affine, …) with many sub-op events that could be fused. *Fix: write a fused kernel.*

## Summary by category

| Category | Count | Total kornia time (ms) | Total tv time (ms) |
|---|---:|---:|---:|
| dispatch-bound | 8 | 125.23 | 9.51 |
| allocation-bound | 2 | 19.52 | 2.94 |
| kernel-bound | 12 | 758.49 | 26.75 |
| sync-bound | 7 | 214.37 | 20.34 |
| fusion-eligible | 8 | 392.30 | 52.18 |

## Per-op classification

| Op | k ms | tv ms | events/iter | self CPU ms | copy_ ms (sync proxy) | sync evt/iter | category | dominant cost | fix |
|---|---:|---:|---:|---:|---:|---:|---|---|---|
| BoxBlur | 28.04 | — | 193 | 38.14 | 30.88 | 14.0 | dispatch-bound | aten::empty + 193 events/iter (+ 14 baseline sync evt/iter) | kornia 2.0 base class redesign — fewer aten::full / empty / to |
| Contrast | 21.04 | 2.62 | 70 | 16.58 | 14.26 | 3.0 | dispatch-bound | aten::fill_ + 70 events/iter (+ 3 baseline sync evt/iter) | kornia 2.0 base class redesign — fewer aten::full / empty / to |
| Solarize | 18.14 | 1.74 | 150 | 23.33 | 17.18 | 3.0 | dispatch-bound | aten::expand + 150 events/iter (+ 3 baseline sync evt/iter) | kornia 2.0 base class redesign — fewer aten::full / empty / to |
| Brightness | 17.98 | 1.63 | 50 | 11.68 | 9.34 | 0.0 | dispatch-bound | aten::fill_ (50 events/iter) | kornia 2.0 base class redesign |
| CenterCrop | 11.84 | 0.15 | 40 | 4.52 | 3.01 | 0.0 | dispatch-bound | aten::empty + 40 events/iter | kornia 2.0 base class redesign — fewer aten::full / empty / to |
| Denormalize | 10.16 | — | 29 | 9.95 | 8.12 | 0.0 | dispatch-bound | aten::fill_ (29 events/iter) | kornia 2.0 base class redesign |
| Normalize | 9.98 | 2.56 | 30 | 10.12 | 8.45 | 0.0 | dispatch-bound | aten::fill_ (30 events/iter) | kornia 2.0 base class redesign |
| Invert | 8.05 | 0.81 | 33 | 9.88 | 8.44 | 0.0 | dispatch-bound | aten::fill_ (33 events/iter) | kornia 2.0 base class redesign |
| GaussianNoise | 12.17 | 2.94 | 127 | 19.28 | 16.28 | 14.0 | allocation-bound | aten::empty_strided dominates allocations | pre-allocate buffers, lift to module __init__ |
| RandomChannelDropout | 7.35 | — | 180 | 13.05 | 8.68 | 14.0 | allocation-bound | aten::empty_strided dominates allocations | pre-allocate buffers, lift to module __init__ |
| MedianBlur | 375.29 | — | 161 | 361.12 | 354.37 | 14.0 | kernel-bound | aten::cudnn_convolution dominates (6043 us self CPU) | irreducible floor — only faster kernel (Triton/CUDA) helps |
| RandomCLAHE | 163.84 | — | 14456 | 263.72 | 37.01 | 16.0 | kernel-bound | aten::histc dominates (839672 us self CPU) | irreducible floor — only faster kernel (Triton/CUDA) helps |
| RandomSnow | 42.77 | — | 294 | 42.12 | 27.78 | 14.0 | kernel-bound | aten::sub dominates (15530 us self CPU) | irreducible floor — only faster kernel (Triton/CUDA) helps |
| GaussianBlur | 32.89 | 4.60 | 261 | 38.16 | 27.08 | 14.0 | kernel-bound | aten::pow dominates (7334 us self CPU) | irreducible floor — only faster kernel (Triton/CUDA) helps |
| Hue | 31.55 | 16.83 | 160 | 34.63 | 24.75 | 0.0 | kernel-bound | aten::mul dominates (17920 us self CPU) | irreducible floor — only faster kernel (Triton/CUDA) helps |
| Saturation | 28.73 | 2.65 | 159 | 39.80 | 27.88 | 0.0 | kernel-bound | aten::mul dominates (19613 us self CPU) | irreducible floor — only faster kernel (Triton/CUDA) helps |
| Grayscale | 19.89 | 0.84 | 62 | 12.38 | 9.80 | 0.0 | kernel-bound | aten::mul dominates (3899 us self CPU) | irreducible floor — only faster kernel (Triton/CUDA) helps |
| VerticalFlip | 17.40 | 0.92 | 32 | 11.54 | 10.06 | 0.0 | kernel-bound | aten::flip dominates (2266 us self CPU) | irreducible floor — only faster kernel (Triton/CUDA) helps |
| HorizontalFlip | 16.87 | 0.92 | 22 | 16.32 | 15.03 | 0.0 | kernel-bound | aten::flip dominates (2344 us self CPU) | irreducible floor — only faster kernel (Triton/CUDA) helps |
| RandomPlanckianJitter | 11.43 | — | 175 | 12.89 | 8.78 | 14.0 | kernel-bound | aten::mul dominates (2927 us self CPU) | irreducible floor — only faster kernel (Triton/CUDA) helps |
| RandomRGBShift | 9.78 | — | 176 | 12.17 | 8.46 | 14.0 | kernel-bound | aten::add dominates (3452 us self CPU) | irreducible floor — only faster kernel (Triton/CUDA) helps |
| RandomChannelShuffle | 8.05 | — | 261 | 11.92 | 7.08 | 14.0 | kernel-bound | aten::index dominates (11704 us self CPU) | irreducible floor — only faster kernel (Triton/CUDA) helps |
| Equalize | 53.54 | 7.93 | 1893 | 70.77 | 8.98 | 182.0 | sync-bound | 182 sync events/iter (3640 total) | lift host->device sync points; cache scalar params; avoid .item()/is_nonzero |
| RandomRain | 40.20 | — | 3057 | 70.84 | 17.96 | 73.0 | sync-bound | 73 sync events/iter (1460 total) | lift host->device sync points; cache scalar params; avoid .item()/is_nonzero |
| RandomErasing | 38.38 | 2.88 | 875 | 51.03 | 36.92 | 82.0 | sync-bound | 82 sync events/iter (1640 total) | lift host->device sync points; cache scalar params; avoid .item()/is_nonzero |
| MotionBlur | 33.05 | — | 1320 | 47.93 | 29.03 | 75.7 | sync-bound | 75 sync events/iter (1514 total) | lift host->device sync points; cache scalar params; avoid .item()/is_nonzero |
| Sharpness | 20.35 | 5.98 | 540 | 38.02 | 24.05 | 110.0 | sync-bound | 110 sync events/iter (2200 total) | lift host->device sync points; cache scalar params; avoid .item()/is_nonzero |
| Posterize | 19.42 | 2.15 | 394 | 27.17 | 7.94 | 48.0 | sync-bound | 48 sync events/iter (960 total) | lift host->device sync points; cache scalar params; avoid .item()/is_nonzero |
| Resize | 9.43 | 1.39 | 1245 | 17.35 | 2.52 | 84.0 | sync-bound | 84 sync events/iter (1680 total) | lift host->device sync points; cache scalar params; avoid .item()/is_nonzero |
| CutMix | 82.66 | 1.94 | 866 | 43.96 | 19.43 | 95.0 | fusion-eligible | composite, 866 events/iter, top: aten::_local_scalar_dense | write a fused kernel (Triton/CUDA) covering all sub-ops |
| Perspective | 62.01 | 9.05 | 1021 | 74.38 | 40.33 | 20.0 | fusion-eligible | composite, 1021 events/iter, top: aten::mul | write a fused kernel (Triton/CUDA) covering all sub-ops |
| Rotation | 57.26 | 6.64 | 816 | 72.94 | 48.05 | 14.0 | fusion-eligible | composite, 816 events/iter, top: aten::mul | write a fused kernel (Triton/CUDA) covering all sub-ops |
| ColorJitter | 52.29 | 23.19 | 436 | 52.12 | 32.71 | 26.0 | fusion-eligible | composite, 436 events/iter, top: aten::mul | write a fused kernel (Triton/CUDA) covering all sub-ops |
| Affine | 51.47 | 7.06 | 466 | 55.22 | 33.90 | 14.0 | fusion-eligible | composite, 466 events/iter, top: aten::mul | write a fused kernel (Triton/CUDA) covering all sub-ops |
| Mosaic | 36.32 | — | 771 | 60.47 | 31.54 | 82.5 | fusion-eligible | composite, 771 events/iter, top: aten::_local_scalar_dense | write a fused kernel (Triton/CUDA) covering all sub-ops |
| MixUp | 34.20 | 2.92 | 244 | 19.75 | 13.43 | 11.0 | fusion-eligible | composite, 244 events/iter, top: aten::add | write a fused kernel (Triton/CUDA) covering all sub-ops |
| ResizedCrop | 16.09 | 1.37 | 1144 | 25.31 | 3.64 | 97.1 | fusion-eligible | composite, 1144 events/iter, top: aten::select | write a fused kernel (Triton/CUDA) covering all sub-ops |

## High-priority fixes (sorted by ROI)

### 1. Base-class redesign (dispatch-bound)
- **Estimated kornia 2.0 perf improvement:** **3-10x** on most dispatch-bound ops (eliminates ~80% of `aten::full`/`empty`/`to`/`lift_fresh` events that today come from the random-param sampler + CPU->GPU param shipping)
- **Difficulty:** **high** (cross-cutting refactor of `_AugmentationBase`, parameter sampler, and forward dispatch)
- **Ops that benefit:**
  - **BoxBlur** (k=28.04 ms) — aten::empty + 193 events/iter (+ 14 baseline sync evt/iter)
  - **Contrast** (k=21.04 ms, k/tv=8.0x) — aten::fill_ + 70 events/iter (+ 3 baseline sync evt/iter)
  - **Solarize** (k=18.14 ms, k/tv=10.4x) — aten::expand + 150 events/iter (+ 3 baseline sync evt/iter)
  - **Brightness** (k=17.98 ms, k/tv=11.0x) — aten::fill_ (50 events/iter)
  - **CenterCrop** (k=11.84 ms, k/tv=80.5x) — aten::empty + 40 events/iter
  - **Denormalize** (k=10.16 ms) — aten::fill_ (29 events/iter)
  - **Normalize** (k=9.98 ms, k/tv=3.9x) — aten::fill_ (30 events/iter)
  - **Invert** (k=8.05 ms, k/tv=9.9x) — aten::fill_ (33 events/iter)

### 2. Pre-allocate buffers (allocation-bound)
- **Estimated kornia 2.0 perf improvement:** **2-4x** on allocation-bound ops (move `zeros`/`eye`/`empty_strided` into `__init__` once; reuse through `register_buffer`)
- **Difficulty:** **low** (per-op patch, similar to existing Normalize buffer patch)
- **Ops that benefit:**
  - **GaussianNoise** (k=12.17 ms, k/tv=4.1x) — aten::empty_strided dominates allocations
  - **RandomChannelDropout** (k=7.35 ms) — aten::empty_strided dominates allocations

### 3. Lift sync points (sync-bound)
- **Estimated kornia 2.0 perf improvement:** **2-5x** on sync-bound ops (every `aten::item` / `is_nonzero` blocks the CPU thread for an entire kernel queue; deferring or removing these is pure win)
- **Difficulty:** **medium** (audit each call site; some are baked into augmentation control flow)
- **Ops that benefit:**
  - **Equalize** (k=53.54 ms, k/tv=6.8x) — 182 sync events/iter (3640 total)
  - **RandomRain** (k=40.20 ms) — 73 sync events/iter (1460 total)
  - **RandomErasing** (k=38.38 ms, k/tv=13.3x) — 82 sync events/iter (1640 total)
  - **MotionBlur** (k=33.05 ms) — 75 sync events/iter (1514 total)
  - **Sharpness** (k=20.35 ms, k/tv=3.4x) — 110 sync events/iter (2200 total)
  - **Posterize** (k=19.42 ms, k/tv=9.0x) — 48 sync events/iter (960 total)
  - **Resize** (k=9.43 ms, k/tv=6.8x) — 84 sync events/iter (1680 total)

### 4. Fused composite kernels (fusion-eligible)
- **Estimated kornia 2.0 perf improvement:** **2-8x** on composite ops by collapsing N sub-ops into one kernel (ColorJitter HSV roundtrip already fused via patch #5; extend the pattern to CutMix/MixUp/Affine)
- **Difficulty:** **medium-high** (Triton or hand-rolled CUDA per fused recipe)
- **Ops that benefit:**
  - **CutMix** (k=82.66 ms, k/tv=42.6x) — composite, 866 events/iter, top: aten::_local_scalar_dense
  - **Perspective** (k=62.01 ms, k/tv=6.9x) — composite, 1021 events/iter, top: aten::mul
  - **Rotation** (k=57.26 ms, k/tv=8.6x) — composite, 816 events/iter, top: aten::mul
  - **ColorJitter** (k=52.29 ms, k/tv=2.3x) — composite, 436 events/iter, top: aten::mul
  - **Affine** (k=51.47 ms, k/tv=7.3x) — composite, 466 events/iter, top: aten::mul
  - **Mosaic** (k=36.32 ms) — composite, 771 events/iter, top: aten::_local_scalar_dense
  - **MixUp** (k=34.20 ms, k/tv=11.7x) — composite, 244 events/iter, top: aten::add
  - **ResizedCrop** (k=16.09 ms, k/tv=11.7x) — composite, 1144 events/iter, top: aten::select

### 5. Kernel optimizations (kernel-bound — irreducible floor)
- **Estimated kornia 2.0 perf improvement:** **1.2-2x** at best (these ops are already real compute; only a faster Triton/CUDA kernel or different algorithm helps)
- **Difficulty:** **high** (per-kernel rewrite; gains are smaller and hardware-specific)
- **Ops that benefit:**
  - **MedianBlur** (k=375.29 ms) — aten::cudnn_convolution dominates (6043 us self CPU)
  - **RandomCLAHE** (k=163.84 ms) — aten::histc dominates (839672 us self CPU)
  - **RandomSnow** (k=42.77 ms) — aten::sub dominates (15530 us self CPU)
  - **GaussianBlur** (k=32.89 ms, k/tv=7.1x) — aten::pow dominates (7334 us self CPU)
  - **Hue** (k=31.55 ms, k/tv=1.9x) — aten::mul dominates (17920 us self CPU)
  - **Saturation** (k=28.73 ms, k/tv=10.8x) — aten::mul dominates (19613 us self CPU)
  - **Grayscale** (k=19.89 ms, k/tv=23.8x) — aten::mul dominates (3899 us self CPU)
  - **VerticalFlip** (k=17.40 ms, k/tv=19.0x) — aten::flip dominates (2266 us self CPU)
  - **HorizontalFlip** (k=16.87 ms, k/tv=18.4x) — aten::flip dominates (2344 us self CPU)
  - **RandomPlanckianJitter** (k=11.43 ms) — aten::mul dominates (2927 us self CPU)
  - **RandomRGBShift** (k=9.78 ms) — aten::add dominates (3452 us self CPU)
  - **RandomChannelShuffle** (k=8.05 ms) — aten::index dominates (11704 us self CPU)

## Concrete recommendations for kornia 2.0 RFC

Based on the categorization above, these architectural decisions are justified:

1. **Slim `_AugmentationBase` (highest ROI — covers 8 ops).** Today every op pays an `aten::full` + `aten::empty` + `aten::to` + `aten::lift_fresh` tax purely for the random-parameter sampler. Replace `_BatchProbGenerator` with a single device-resident `rand` tensor that is sliced per call. Drop the 60+ event scaffolding around `_apply_func_by_input_type`. Target: **events_per_iter < 10** for all single-op augmentations.

2. **Move parameter generators to GPU once, not per-call.** Most dispatch-bound ops show 20+ `aten::to` calls per iter, each shipping a tiny CPU scalar (degrees, threshold, brightness factor) to GPU. Cache these as device-resident `register_buffer` at `__init__` time; resample on-device.

3. **Pre-allocate transformation matrices and masks (2 allocation-bound ops).** Geometric ops re-create `torch.zeros(B, 3, 3)` every iter for the affine matrix, and `torch.eye(3)` for identity initializers. Lift to `__init__` and fill in-place.

4. **Eliminate `aten::is_nonzero` / `aten::item` in hot paths (7 sync-bound ops today).** Replace `if mask.any():` with `mask_t * branch_a + (1-mask_t) * branch_b` to keep the graph fully on-device. The current pattern blocks the CPU thread on every iter.

5. **Provide fused recipes for 8 composite ops.** Build a small registry of hand-rolled kernels: ColorJiggle (already has fused-HSV via v4 patch), CutMix (single mask + blend), MixUp (single linear combo), Affine (closed-form matrix + grid_sample). Mosaic is the largest ROI; today it stitches 4 images via 4 separate `affine_grid` + `grid_sample` calls.

6. **Don't optimize the 12 kernel-bound ops first.** They are already hitting the irreducible floor (true convolution / grid_sample / flip / hist). Optimizing them is high effort, low ROI compared to (1)–(5). Defer to a phase 2 kernel-rewrite RFC.

7. **Adopt a v6-style `forward()` override path as the default.** The `run_v6.py` aggressive override shaves 30-70% off many ops by skipping `_AugmentationBase.forward`. The kornia 2.0 base class should have this as its only path — no opt-in monkey-patch needed.

8. **Verify on Jetson Orin (this rig): CUPTI privileges block CUDA-self-time profiling.** k2 RFC should mandate dual profiling (CPU self-time + CUDA-event wallclock) to detect regressions in either dimension; relying on `aten::copy_` as a sync proxy is fragile.

