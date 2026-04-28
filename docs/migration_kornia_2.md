# Migrating from kornia 1.x to 2.0 (augmentations)

This guide helps you upgrade from `kornia.augmentation` (1.x) to `kornia.augmentations` (2.0). For most users, **migration is a single import change plus zero code changes** — the rest is opt-in.

If you're upgrading and not in a hurry: keep using `kornia.augmentation` (singular). It's preserved as a transparent re-export shim through kornia 2.x and only becomes a deprecation alias in 3.0.

If you want the perf wins: switch to `kornia.augmentations` (plural). Read on for the details.

## TL;DR — the 90% case

```python
# 1.x (still works in 2.x)
import kornia.augmentation as K
aug = K.AugmentationSequential(
    K.RandomHorizontalFlip(p=0.5),
    K.RandomAffine(degrees=15.0),
    K.ColorJiggle(brightness=0.2),
    K.Normalize(mean=torch.tensor([0.485, 0.456, 0.406]),
                std=torch.tensor([0.229, 0.224, 0.225])),
)
out = aug(x)

# 2.0 — same code, faster
import kornia.augmentations as K   # plural namespace = fast path
aug = K.AugmentationSequential(
    K.RandomHorizontalFlip(p=0.5),
    K.RandomAffine(degrees=15.0),
    K.ColorJiggle(brightness=0.2),
    K.Normalize(mean=torch.tensor([0.485, 0.456, 0.406]),
                std=torch.tensor([0.229, 0.224, 0.225])),
)
out = aug(x)
```

**That's the migration for most users.** Same API, ~4× faster on Jetson Orin / 2-3× on data-center GPUs.

## What actually changed (and what didn't)

### Default behavior change: per-sample randomness is now opt-in

This is the **only** behavioral default change in 2.0. In 1.x, every transform used per-sample random params by default (each image in the batch got its own coin flip / rotation angle / etc.). In 2.0, the default flips to batch-shared params (matching torchvision.v2's behavior).

```python
# 1.x default — each image had its own flip decision
aug_1x = kornia.augmentation.RandomHorizontalFlip(p=0.5)
out = aug_1x(batch_of_8)
# Result: ~4 of 8 flipped (per-sample coin flips)

# 2.0 default — one flip decision for the whole batch
aug_2 = kornia.augmentations.RandomHorizontalFlip(p=0.5)
out = aug_2(batch_of_8)
# Result: all 8 flipped or all 8 not (one coin flip)

# 2.0 explicit per-sample (matches 1.x default)
aug_2_per = kornia.augmentations.RandomHorizontalFlip(p=0.5, per_sample=True)
out = aug_2_per(batch_of_8)
# Result: ~4 of 8 flipped (per-sample coin flips)
```

**Why this changed:** per-sample randomness costs ~30-40% extra GPU work on geometric ops (B different matrices, B different grids, B different `grid_sample` warps). For most users (classification training with augmentation as regularization), batch-shared is plenty. Per-sample is the kornia advantage for SSL/contrastive recipes that need multi-view in one batch — it stays available, just opt-in.

**Migration:** if your training relied on per-sample variation, add `per_sample=True` to your transforms. If you used `same_on_batch=True` in 1.x, **remove it** — that's now the default.

### `same_on_batch` argument is deprecated

```python
# 1.x — explicitly batch-shared
aug = kornia.augmentation.RandomHorizontalFlip(p=0.5, same_on_batch=True)

# 2.0 — drop the argument; that's now the default
aug = kornia.augmentations.RandomHorizontalFlip(p=0.5)
```

In 2.0–2.1, passing `same_on_batch=True` works with a deprecation warning. In 3.0 it's removed. `same_on_batch=False` becomes a clear error in 2.1 — use `per_sample=True` instead.

### Multi-target: `data_keys=` → tagged tensors (both work in 2.x)

The recommended pattern in 2.0 uses tagged tensor types for type-safe multi-target dispatch. The old `data_keys=` API still works through the 2.x cycle.

```python
# 1.x — string-keyed dispatch
aug = K.AugmentationSequential(
    K.RandomHorizontalFlip(p=0.5),
    K.RandomAffine(15.0),
    data_keys=["input", "bbox_xyxy", "mask"],
)
out_img, out_boxes, out_mask = aug(img, boxes, mask)

# 2.0 — tagged tensors, type-dispatched
img = K.tensors.Image(img)
boxes = K.tensors.BoundingBoxes(boxes, format="xyxy", image_size=(H, W))
mask = K.tensors.Mask(mask)
aug = K.AugmentationSequential(
    K.RandomHorizontalFlip(p=0.5),
    K.RandomAffine(15.0),
)
out_img, out_boxes, out_mask = aug(img, boxes, mask)

# 1.x style still works in 2.x (deprecation warning in 2.2):
out_img, out_boxes, out_mask = aug(img.tensor, boxes, mask, data_keys=["input", "bbox_xyxy", "mask"])
```

**Why this matters:** `data_keys=` runs through enum dispatch on every call. Image-only calls (the 90% case) pay this overhead even though they don't need it. Tagged tensors dispatch by type at zero overhead for the image-only case.

### `transform_matrix` is now lazy

```python
# 1.x — transform_matrix always computed
aug = K.RandomAffine(15.0)
out = aug(x)
matrix = aug.transform_matrix  # always available, paid every forward

# 2.0 — computed only on first read
aug = K.RandomAffine(15.0)
out = aug(x)
matrix = aug.transform_matrix  # computes now (one time), caches until next forward
matrix2 = aug.transform_matrix  # returns cached value
```

**Migration:** if you read `aug.transform_matrix`, no change needed — works identically. If you don't read it, you save ~1-2 ms per call automatically.

### Replay (`params=`) is now opt-in via `return_params=True`

```python
# 1.x — params always cached on the transform
aug = K.RandomAffine(15.0)
out1 = aug(x1)
out2 = aug(x2, params=aug._params)  # replay last call's params

# 2.0 — caller declares replay intent
aug = K.RandomAffine(15.0)
out1, params = aug(x1, return_params=True)  # opt-in
out2 = aug(x2, params=params)
```

**Migration:** if you use replay, add `return_params=True` and capture the returned params. If you don't use replay, you save the dict construction cost automatically.

### `inverse()` requires params explicitly

```python
# 1.x — uses last-cached params automatically
aug = K.RandomAffine(15.0)
out = aug(x)
back = aug.inverse(out)

# 2.0 — pass params explicitly OR use return_params=True
aug = K.RandomAffine(15.0)
out, params = aug(x, return_params=True)
back = aug.inverse(out, params)

# Alternative: aug.inverse(out) still works if return_params=True was used,
# since the transform stashed _last_params.
```

If you call `aug.inverse(out)` without `return_params=True`, you get `RuntimeError: inverse requires params; call with return_params=True or pass explicitly`.

### Differentiability is gone

In 1.x, kornia transforms claimed to be differentiable (gradients could flow through). In 2.0, **all augmentation transforms run under `@torch.no_grad()`**. This is a deliberate scope decision: augmentations are training-data preparation, not part of the model graph.

```python
# 1.x — gradients flowed through (not always working as documented)
x = torch.randn(1, 3, 32, 32, requires_grad=True)
out = K.Normalize(mean=..., std=...)(x)
out.sum().backward()
print(x.grad)  # had values

# 2.0 — gradients do NOT flow through augmentations
x = torch.randn(1, 3, 32, 32, requires_grad=True)
out = K.Normalize(mean=..., std=...)(x)
out.sum().backward()  # error: no grad path
```

**Migration:** if you used kornia.augmentation transforms inside a model's forward (gradient-preserving preprocessing), import the underlying functional directly:

```python
# Old kornia 1.x pattern (in-model preprocessing)
self.normalize = kornia.augmentation.Normalize(mean=..., std=...)
def forward(self, x):
    x = self.normalize(x)
    return self.backbone(x)

# kornia 2.0 pattern — use kornia.enhance for in-model preprocessing
import kornia.enhance
def forward(self, x):
    x = kornia.enhance.normalize(x, mean=..., std=...)  # gradient-preserving
    return self.backbone(x)
```

`kornia.enhance.normalize` is unchanged in 2.0 and remains gradient-preserving. The `NormalizeWithGrad` class shipped in 1.x funnel work is removed in 2.0; users with that import need to switch to `kornia.enhance.normalize`.

## Recipe migrations

### Detection training (DETR / YOLO / RT-DETR)

```python
# Recommended: use the preset
import kornia.augmentations as K
aug = K.presets.detr(resolution=560)
# Or:
aug = K.presets.yolov8()

# Manual equivalent
aug = K.AugmentationSequential(
    K.RandomHorizontalFlip(p=0.5),
    K.RandomAffine(degrees=15.0, translate=(0.1, 0.1), scale=(0.8, 1.2), p=1.0),
    K.ColorJiggle(brightness=0.2, contrast=0.2, saturation=0.2, p=1.0),
    K.Normalize(mean=K.IMAGENET_MEAN, std=K.IMAGENET_STD),
    bbox_params=K.BboxParams(format="xyxy", min_visibility=0.3, label_fields=["labels"]),
)

# Run with tagged tensors
img_t = K.tensors.Image(img)
boxes_t = K.tensors.BoundingBoxes(boxes, format="xyxy", image_size=(H, W))
out_img, out_boxes = aug(img_t, boxes_t)
```

### SSL multi-view (DINOv2, SimCLR, MoCov3)

```python
# DINOv2: 2 global views + N local views
aug = K.presets.dinov2(local_views=8)
# Returns global_view_1, global_view_2, local_view_1, ..., local_view_8

# Manually composed — note per_sample=True is automatic in SSL presets
aug = K.AugmentationSequential(
    K.RandomResizedCrop(size=(224, 224), scale=(0.4, 1.0), per_sample=True),
    K.RandomHorizontalFlip(p=0.5, per_sample=True),
    K.ColorJiggle(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1, per_sample=True, p=0.8),
    K.RandomGrayscale(p=0.2, per_sample=True),
    K.RandomGaussianBlur(kernel_size=23, sigma=(0.1, 2.0), p=0.5, per_sample=True),
    K.RandomSolarize(threshold=0.5, p=0.2, per_sample=True),
    K.Normalize(mean=K.IMAGENET_MEAN, std=K.IMAGENET_STD),
)
```

### Segmentation (SAM / MaskFormer / SegFormer)

```python
# Use tagged Mask tensor — supports [H,W] and [B,H,W] natively, no fake channel dim
aug = K.presets.sam()

img_t = K.tensors.Image(img)
mask_t = K.tensors.Mask(seg_mask)  # accepts (H, W) or (B, H, W)
out_img, out_mask = aug(img_t, mask_t)
```

### Lightning DataModule

```python
import pytorch_lightning as pl
import kornia.augmentations as K

aug = K.presets.detr(resolution=560)

trainer = pl.Trainer(
    callbacks=[K.lightning.AugmentationCallback(aug)],
    ...
)
```

The callback applies augmentation in `on_after_batch_transfer` automatically — same pattern as rf-detr's manual integration in 1.x.

## Migrating from other libraries

### From Albumentations

```python
# Old albumentations spec
import albumentations as A
spec = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.Affine(rotate=(-15, 15), translate_percent=(-0.1, 0.1), scale=(0.8, 1.2), p=1.0),
    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, p=1.0),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Auto-translate to kornia 2.0
import kornia.augmentations as K
aug = K.compat.from_albumentations(spec.to_dict())
# Returns a kornia AugmentationSequential + a fidelity report
print(aug.fidelity_report)  # which transforms exact, approximate, unsupported
```

The fidelity report flags any transforms that don't have a 1:1 kornia equivalent. Approximate translations are explicit; unsupported ones raise `NotTranslatable` with a suggestion.

### From torchvision.v2

```python
import torchvision.transforms.v2 as T
tv_compose = T.Compose([
    T.RandomHorizontalFlip(p=0.5),
    T.RandomAffine(degrees=15.0),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Translate to kornia 2.0
import kornia.augmentations as K
aug = K.compat.from_torchvision_v2(tv_compose)
```

Note: tv's batch-shared semantics map directly to kornia 2.0's default mode. Per-sample variation is not added during translation; if you want it, set `per_sample=True` after translation.

## Common errors and fixes

### `RuntimeError: inverse requires params`

You called `aug.inverse(out)` without first calling with `return_params=True`.

```python
# Fix:
out, params = aug(x, return_params=True)
back = aug.inverse(out, params)
```

### `NotTranslatable: <transform>` from compat importer

The transform in your source spec doesn't have a kornia equivalent. Either:
1. Replace it with a similar kornia transform manually
2. File an issue — we may add it
3. Use the `unsupported_passthrough=True` flag to skip unsupported entries silently (with a warning)

### Goldens / numerical equivalence test failures after migration

Most likely your test was relying on the 1.x per-sample default. Either:
1. Add `per_sample=True` to the transforms to restore 1.x behavior
2. Regenerate the goldens with the new batch-shared semantics

```python
# Restore 1.x-style per-sample
aug = K.RandomHorizontalFlip(p=0.5, per_sample=True)
```

### `AttributeError: 'NormalizeWithGrad'`

Removed in 2.0. Use `kornia.enhance.normalize` directly for gradient-preserving in-model preprocessing.

### Slower than expected after migration

Check:
1. You're using the `kornia.augmentations` namespace (plural), not `kornia.augmentation` (singular).
2. You're not setting `per_sample=True` unnecessarily — only SSL/multi-view recipes need it.
3. You're not passing tensors via `data_keys=` for image-only calls — that path runs the legacy dispatch.
4. `aug.transform_matrix` access in a hot loop — each access triggers lazy compute. Cache the result if you read it repeatedly.

### TorchScript scripting fails

Augmentation transforms in 2.0 are not scriptable (Pydantic params, type-dispatch registry, dict-based params don't TorchScript). For scripted preprocessing, use the functional API:

```python
import kornia.augmentations.functional as KF

@torch.jit.script
def preprocess(x: torch.Tensor) -> torch.Tensor:
    x = KF.normalize(x, mean=..., std=...)  # scriptable
    return x
```

## Performance you should expect

On Jetson Orin (the development hardware), with default `per_sample=False`:

| Op | kornia 1.x | kornia 2.0 default | torchvision.v2 |
|---|---:|---:|---:|
| HorizontalFlip | 6.15 ms | ~1.5 ms | 1.12 ms |
| CenterCrop | 11.83 ms | ~0.3 ms | 0.15 ms |
| Normalize | 6.56 ms | ~2.5 ms | 2.24 ms |
| RandomAffine | 51.47 ms | ~9 ms | 7.06 ms |
| ColorJiggle | 52.29 ms | ~28 ms | 23.19 ms |
| DETR pipeline (4 ops) | 58.1 ms | ~15 ms | 24.3 ms |

With `per_sample=True` (kornia's SSL feature), expect ~2× of the default-mode numbers because per-sample mode does B different transforms instead of 1 broadcast.

On A100 / 4090 / H100, ratios are similar but absolute numbers are smaller (kernel time dominates less; framework overhead removal still helps).

## Where kornia 2.0 wins decisively (not just matches)

The per-op table above is the conservative "match torchvision" pitch. There are five places where kornia 2.0 structurally **beats** torchvision:

### 1. CUDA Graph captured pipelines

```python
aug = K.presets.detr(resolution=560)
aug_graph = aug.cuda_graph(batch_shape=(8, 3, 560, 560))   # capture once
out = aug_graph(x)                                          # replay — near-zero overhead
```

torchvision's transforms can't be graph-captured cleanly without invasive refactor (in-forward `torch.tensor()` calls). kornia 2.0's design lifts param sampling out of forward specifically to enable this. **Realistic 5× pipeline-level speedup over tv.**

### 2. SSL multi-view (DINOv2, SimCLR)

```python
# 10 augmented views per source image, batch of 8 → kornia: ONE forward call
aug = K.presets.dinov2(local_views=8)   # per_sample=True automatic
views = aug(images)                       # B=80 in one shot

# Same in torchvision = 80 separate forward calls + Python collate
```

**5-10× faster than torchvision on SSL recipes.**

### 3. Detection pipelines with shared multi-target

```python
# Image + boxes + mask augmented together, ONE param sampling
img = K.tensors.Image(image)
boxes = K.tensors.BoundingBoxes(b, format="xyxy", image_size=(H, W))
mask = K.tensors.Mask(m)
out_img, out_boxes, out_mask = aug(img, boxes, mask)   # 3 kernels, 1 sample
```

torchvision via TVTensor dispatch runs N forward calls (one per target type). **1.5-3× faster on detection/segmentation pipelines.**

### 4. Composite ops with fused Triton kernels

```python
# After kornia 2.0 ships fused intensity Triton kernels:
aug = K.ColorJiggle(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
# ColorJiggle: 52.29 ms (1.x) → ~5 ms (2.0 with Triton) vs torchvision 23.19 ms
```

**1.5-4.6× faster on composite ops.** Available on x86 + CUDA + Triton (not on Jetson aarch64 — Triton missing).

### 5. Transforms torchvision doesn't have

If your recipe uses any of these, you have no torchvision option:

```
RandomElasticTransform, RandomThinPlateSpline, RandomFisheye,
RandomCLAHE, RandomMotionBlur, RandomMedianBlur,
Mosaic, RandomCopyPaste, RandomFog/Snow/Rain
```

For these kornia 2.0 is the only competitive choice on GPU.

### Summary

```
torchvision strengths:  fast simple kernel ops, batch-shared params, ~30 transforms,
                         minimal API surface, well-known and trusted

kornia 2.0 strengths:   matches tv on simple ops in default mode,
                         beats tv on graph-captured pipelines (5×),
                         beats tv on SSL multi-view (5-10×),
                         beats tv on detection multi-target (1.5-3×),
                         beats tv on composite ops with Triton (1.5-4.6×),
                         200+ ops including ~10 with no tv equivalent,
                         3D + video + geometric richness (perspective, TPS, fisheye, elastic),
                         differentiability available via kornia.enhance.* (out of augmentation scope)
```

Pick the right tool for the job:
- **Simple classification training** with stock augmentations → torchvision is fine, kornia 2.0 also fine
- **SSL training** (DINOv2, SimCLR, MoCo) → kornia 2.0 is decisively faster
- **Detection training** with bbox/mask → kornia 2.0 is faster + cleaner API
- **Geometric-heavy recipes** (perspective, fisheye, elastic) → kornia is the only option
- **Inference-time deterministic preprocessing** with ONNX export → either works; kornia.augmentations.deterministic is the export-clean subset

## Frequently asked questions

**Q: Do I have to migrate?**
No. `kornia.augmentation` (singular) is preserved as a transparent re-export shim through kornia 2.x. It only becomes a deprecation alias in 3.0 (~1 year out).

**Q: My training pipeline depends on per-sample randomness. Will 2.0 break it?**
No automatic break — but performance won't improve until you add `per_sample=True`. Without that flag, your default behavior changes from per-sample to batch-shared. If you have validation tests checking per-sample variation, add the flag and your behavior is restored.

**Q: Is the funnel work from 1.x preserved?**
Yes. All 18 PRs from the 1.x funnel (BboxParams, NormalizeWithGrad → moved to enhance, deterministic namespace export, GPU-side RNG, per-op kernel optimizations, channels-last awareness, 5 new transforms, etc.) are preserved in 2.0 either through re-exports or as the new fast-path implementations.

**Q: What about TorchScript / `torch.compile` / CUDA Graphs?**
- TorchScript: not supported for class-based transforms (use `kornia.augmentations.functional` for scripted preprocessing).
- `torch.compile`: works on the new architecture (`fullgraph=True` with `per_sample=False` is the most reliable). Not validated on Jetson aarch64 (no Triton); validated on x86 + CUDA + Triton.
- CUDA Graph capture: works for deterministic-mode (`per_sample=False`) compositions. Does not work with `per_sample=True` (random samplers in forward).

**Q: rf-detr / torchgeo / lightly — when will they support 2.0?**
The 1.x namespace re-export means downstream code keeps working without changes. Each downstream maintainer can opt into the plural namespace at their own pace. The kornia maintainers will offer migration help (PRs, review) for the major downstream users when 2.0 ships.

**Q: I'm a kornia 1.x heavy user with custom transforms. Do I have to rewrite them?**
No, but you should. 1.x base classes (`AugmentationBase2D`, `RigidAffineAugmentationBase2D`, etc.) are preserved as deprecation shims in 2.x. Custom transforms subclassing them keep working but don't get the 2.0 perf wins. Migration to the new `Transform` base + mixins is straightforward — see [appendix C](#appendix-c-custom-transform-migration-walkthrough) for a worked example.

**Q: How long is the deprecation window?**
- 2.0.0 → 3.0.0: ~1 year. Plenty of time to migrate.
- `same_on_batch=False` deprecation warning starts at 2.1.
- `data_keys=` deprecation warning starts at 2.2.
- All deprecations enforced at 3.0.

## Appendix A: Public API additions in 2.0

New top-level imports in `kornia.augmentations`:

```python
# Tagged tensor types (multi-target dispatch)
K.tensors.Image, K.tensors.BoundingBoxes, K.tensors.Mask, K.tensors.Keypoints, K.tensors.Video

# Recipe presets
K.presets.detr, K.presets.dinov2, K.presets.simclr, K.presets.mocov3,
K.presets.yolov8, K.presets.sam

# Compat importers
K.compat.from_albumentations, K.compat.from_torchvision_v2

# Functional API mirror
K.functional.random_horizontal_flip, K.functional.affine, K.functional.normalize, ...

# Configuration
K.BboxParams, K.IMAGENET_MEAN, K.IMAGENET_STD

# Lightning integration
K.lightning.AugmentationCallback
```

All 1.x public symbols continue to be importable via the singular namespace and the plural namespace re-exports them.

## Appendix B: Public API removals in 2.0

Only one removal:
- `kornia.augmentations.deterministic.NormalizeWithGrad` — replaced by direct use of `kornia.enhance.normalize`.

The 1.x `kornia.augmentation.Normalize` still exists in 2.0 (in `kornia.augmentations`) but runs under `@torch.no_grad()`. For gradient flow, use `kornia.enhance.normalize` directly.

## Appendix C: Custom transform migration walkthrough

If you have a custom 1.x transform like:

```python
class MyCustomBlur(IntensityAugmentationBase2D):
    def __init__(self, kernel_size=5, sigma=1.5, p=0.5):
        super().__init__(p=p)
        self.kernel_size = kernel_size
        self.sigma = sigma
    def apply_transform(self, input, params, flags, transform=None):
        return kornia.filters.gaussian_blur2d(input, (self.kernel_size,)*2, (self.sigma,)*2)
```

The 2.0 equivalent:

```python
import kornia
from kornia.augmentations.base import (
    Transform, _RandomMixin, _MultiTargetMixin, register_kernel,
)
from kornia.tensors import Image, Mask

class MyCustomBlur(Transform, _RandomMixin, _MultiTargetMixin):
    """Custom Gaussian blur with random per-sample probability gate."""

    def __init__(self, kernel_size: int = 5, sigma: float = 1.5, p: float = 0.5,
                 *, per_sample: bool = False):
        super().__init__()
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.p = p
        self.per_sample = per_sample

    def _sample_params(self, shape, generator=None):
        if self.per_sample:
            return {"apply": torch.rand(shape[0], generator=generator) < self.p}
        return {"apply": torch.rand(1, generator=generator).item() < self.p}

    def _apply(self, x, params):
        apply = params["apply"]
        if isinstance(apply, bool):
            if not apply:
                return x
            return kornia.filters.gaussian_blur2d(
                x, (self.kernel_size,)*2, (self.sigma,)*2
            )
        # per_sample
        blurred = kornia.filters.gaussian_blur2d(
            x, (self.kernel_size,)*2, (self.sigma,)*2
        )
        return torch.where(apply[:, None, None, None], blurred, x)


@register_kernel(MyCustomBlur, Image)
def _blur_image(t, x, params):
    return t._apply(x, params)


@register_kernel(MyCustomBlur, Mask)
def _blur_mask(t, m, params):
    # Blur doesn't apply to masks — passthrough
    return m
```

Differences:
- Subclass `Transform` + mixins instead of `IntensityAugmentationBase2D`
- `apply_transform` → `_apply`; `params` is a plain dict not a tensor dict
- Random sampling lives in `_sample_params`, not `forward_parameters`
- Multi-target dispatch via `register_kernel` decorators (only if your transform supports boxes/masks/keypoints; image-only transforms can omit `_MultiTargetMixin` and the registrations)
- `per_sample=False` default — make per-sample opt-in if your transform supports it

The new transform is ~30 lines. The 1.x version was ~12 lines but inherited a thousand+ line base class.

## Appendix D: Where to ask questions

- GitHub Discussions: https://github.com/kornia/kornia/discussions
- Slack: kornia.org → community → Slack
- Issues for bugs: https://github.com/kornia/kornia/issues
- The migration guide source: this file (`docs/migration_kornia_2.md`) — PRs welcome.

If you migrate a major project (rf-detr, torchgeo, lightly, etc.) and want help, ping `@edgarriba` on GitHub.
