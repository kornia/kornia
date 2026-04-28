---
name: kornia-augmentations
description: Use when working with image/video augmentation pipelines for ML training in PyTorch. Activates for tasks involving data augmentation, GPU augmentation, bbox/mask/keypoint transforms, torchvision-to-kornia or albumentations-to-kornia migration, ONNX or torch.export of preprocessing transforms, authoring new kornia transforms, or composing AugmentationSequential pipelines for detection/segmentation/SSL recipes.
---

# Kornia Augmentations Skill

## Mode Detection

This skill supports two distinct workflows. Identify your context before proceeding.

### USER mode triggers
- Building augmentation pipelines for ML training
- Composing transforms via `AugmentationSequential`
- Migrating from albumentations or torchvision v2
- Exploring available transforms via catalog
- Compiling or optimizing augmentation graphs
- Exporting augmentations to ONNX or torch.export
- Integrating with PyTorch Lightning training loops
- Handling bounding box/mask/keypoint transforms

### DEVELOPER mode triggers
- Writing a new transform class
- Testing transform correctness via contract tests
- Authoring performance tests or benchmarks
- Building presets for published recipes (DETR, YOLO, SAM, etc.)
- Adding to deterministic export namespace
- Contributing to kornia/augmentations core

If ambiguous, ask the user once: "Are you building a pipeline or authoring a new transform?"

---

## USER Mode Workflow

### 1. Discover Transforms via Catalog

Explore available transforms programmatically:

```python
from kornia.augmentations import auto, container

# List all 2D augmentations (100+ transforms)
import kornia.augmentation as K
import inspect

transforms_2d = [
    name for name, obj in inspect.getmembers(K)
    if inspect.isclass(obj) and name.startswith('Random')
]
print(f"Available: {len(transforms_2d)} transforms")

# Example: RandomAffine, RandomBrightness, RandomCrop, RandomGaussianBlur, etc.
```

### 2. Inspect Transform Parameters via Schema

Every transform exposes a Pydantic schema. Inspect before use:

```python
import kornia.augmentation as K

# For any transform T, inspect its parameters
T = K.RandomAffine
schema = T.__init__.__annotations__  # or check docstring

# Example parameters:
aug = K.RandomAffine(
    degrees=(-45.0, 45.0),       # rotation range in degrees
    translate=(0.1, 0.1),         # translation as fraction of image
    scale=(0.8, 1.2),             # scale range
    p=0.5                          # apply probability
)
print(aug)
```

### 3. Compose Pipelines with AugmentationSequential

Build deterministic, GPU-accelerated augmentation chains:

```python
import torch
import kornia.augmentation as K

# Image-only pipeline
aug = K.AugmentationSequential(
    K.RandomAffine(degrees=45, p=0.8),
    K.RandomBrightness((0.0, 1.0), p=0.5),
    K.RandomGaussianBlur((3, 5), (0.1, 2.0), p=0.3),
)

# Apply to batch (B, C, H, W)
images = torch.randn(4, 3, 224, 224)
augmented = aug(images)
```

For detection pipelines, add bounding box support:

```python
from kornia.geometry.boxes import Boxes

aug = K.AugmentationSequential(
    K.RandomAffine(degrees=45, p=1.0),
    K.RandomCrop((128, 128), p=1.0),
    data_keys=["input", "bbox_xyxy"],
)

# Images: (B, C, H, W)
# Boxes: kornia.geometry.boxes.Boxes instance or (B, N, 4) tensor [x1, y1, x2, y2]
images = torch.randn(2, 3, 256, 256)
boxes = Boxes(torch.tensor([
    [[10., 10., 100., 100.]],  # batch 1, object 1
    [[50., 50., 150., 150.]]   # batch 2, object 1
]))

images_aug, boxes_aug = aug(images, boxes)
```

For segmentation, use masks:

```python
aug = K.AugmentationSequential(
    K.RandomAffine(degrees=30, p=1.0),
    K.RandomCrop((128, 128), p=1.0),
    data_keys=["input", "mask"],
)

images = torch.randn(2, 3, 256, 256)
masks = torch.randint(0, 10, (2, 1, 256, 256))  # (B, C, H, W)

images_aug, masks_aug = aug(images, masks)
```

### 4. Migration: Albumentations or Torchvision v2

If a module exists, use the compat import; otherwise, manually port key transforms:

```python
import kornia.augmentation as K
from kornia.augmentation.compat import from_albumentations, from_torchvision_v2

# Albumentations round-trip (if compat module exists)
from albumentations import Compose, HorizontalFlip, RandomBrightnessContrast
alb_aug = Compose([
    HorizontalFlip(p=0.5),
    RandomBrightnessContrast(p=0.2),
])
try:
    kornia_aug = from_albumentations(alb_aug)
except (ImportError, AttributeError):
    # Fallback: manually compose equivalent
    kornia_aug = K.AugmentationSequential(
        K.RandomHorizontalFlip(p=0.5),
        K.RandomBrightness((0.0, 0.3), p=0.2),
    )

# Torchvision v2 round-trip
from torchvision.transforms import v2
tv_aug = v2.Compose([
    v2.RandomAffine(degrees=30),
    v2.GaussianBlur(kernel_size=5),
])
try:
    kornia_aug = from_torchvision_v2(tv_aug)
except (ImportError, AttributeError):
    # Fallback: manual port
    kornia_aug = K.AugmentationSequential(
        K.RandomAffine(degrees=30),
        K.RandomGaussianBlur((5, 5), (0.1, 2.0)),
    )
```

### 5. Recipe Presets

If building detection/segmentation/SSL models, check for published presets:

```python
import kornia.augmentation as K

# Preset names follow recipe convention: presets.{model}
# Check these if they exist in kornia.augmentation:
presets = [
    'detr',      # DETR object detection
    'dinov2',    # DINOv2 SSL
    'yolov8',    # YOLOv8 detection
    'sam',       # Segment Anything Model
]

# If a preset module exists (kornia.augmentation.presets.{name}):
try:
    from kornia.augmentation.presets import detr
    aug = detr.build_augmentation()
except (ImportError, AttributeError):
    # Fallback: compose manually (see recipe papers)
    pass
```

### 6. Performance: Compilation and CUDA Graphs

Optimize augmentation graphs for inference or batch training:

```python
import torch
import kornia.augmentation as K

aug = K.AugmentationSequential(
    K.RandomAffine(degrees=30, p=0.5),
    K.RandomBrightness((0.0, 0.5), p=0.5),
)

# Option A: torch.compile for JIT acceleration
# Use when: eager execution is slow; you have stable shapes; inference or training
try:
    aug_compiled = torch.compile(aug, fullgraph=True)
    # First call incurs compilation overhead; subsequent calls are faster
    out = aug_compiled(images)
except RuntimeError as e:
    print(f"torch.compile failed: {e}. Falling back to eager execution.")
    aug_compiled = aug

# Option B: CUDA graph capture (fastest for fixed batch size)
# Use when: batch size, input shape, and sequence are fixed; inference critical path
if torch.cuda.is_available():
    batch_shape = (32, 3, 224, 224)  # Fixed batch shape
    # Warm up
    dummy = torch.randn(*batch_shape, device='cuda')
    for _ in range(3):
        _ = aug(dummy)

    # Capture
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = aug(dummy)

    # Reuse on same-shaped inputs
    images = torch.randn(*batch_shape, device='cuda')
    images.copy_(images)  # or images_copy[:] = images
    g.replay()
    result = out  # g modified out in-place
```

### 7. PyTorch Lightning Integration

If using Lightning, integrate augmentation as a callback (if module exists):

```python
import pytorch_lightning as pl
import kornia.augmentation as K
from kornia.augmentation.lightning import AugmentationCallback

class MyDataModule(pl.LightningDataModule):
    def setup(self, stage: str):
        # Load datasets
        self.train_ds = ...
        self.val_ds = ...

class MyModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = ...

# Option A: Callback-based (if kornia.augmentations.lightning exists)
try:
    from kornia.augmentations.lightning import AugmentationCallback
    aug = K.AugmentationSequential(
        K.RandomAffine(degrees=30, p=0.8),
        K.RandomBrightness((0.0, 0.5), p=0.5),
    )
    callback = AugmentationCallback(aug)
    trainer = pl.Trainer(callbacks=[callback])
except (ImportError, AttributeError):
    # Fallback: apply augmentation in DataModule or on-the-fly
    trainer = pl.Trainer()
```

Option B: Apply augmentation in DataLoader or training step:

```python
def train_step(batch):
    images, labels = batch
    images_aug = aug(images)
    return images_aug, labels
```

### 8. Export for Inference: Deterministic Namespace and ONNX

Augmentations must be deterministic and export-clean for serving:

```python
import torch
import kornia.augmentation as K

# Normalize and export: no randomness
aug = K.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

# Try torch.export (TorchScript-like graph capture)
try:
    example_input = torch.randn(1, 3, 224, 224)
    exported = torch.export.export(aug, (example_input,))
    print("torch.export successful")
except Exception as e:
    print(f"torch.export failed: {e}. Check for control flow or inplace ops.")

# Fallback: torch.jit.script or traced module
try:
    aug_jit = torch.jit.script(aug)
except Exception:
    aug_jit = torch.jit.trace(aug, torch.randn(1, 3, 224, 224))

# ONNX export (if supported)
try:
    import torch.onnx
    torch.onnx.export(
        aug,
        torch.randn(1, 3, 224, 224),
        "augmentation.onnx",
        opset_version=14,
        input_names=["image"],
        output_names=["output"]
    )
    print("ONNX export successful")
except Exception as e:
    print(f"ONNX export failed: {e}")
```

**Key point**: Not all transforms are export-clean. Geometric transforms (affine, perspective) are; some color jitter or random crops may hit control flow. Test early.

### 9. Common Errors and Resolutions

| Error | Cause | Resolution |
|-------|-------|-----------|
| `RuntimeError: Expect (B, C, H, W). Got (H, W)` | Input shape mismatch | Ensure input is 4D batch `(B, C, H, W)`. Use `transform_tensor()` or batch manually: `img[None, None, ...]` |
| `ValueError: sample vs batch params` | Mixing per-sample and per-batch randomization | Set `same_on_batch=True` if all samples in batch should use the same transform; `False` for independent. Default is `False`. |
| `Mask shape mismatch` | Mask channel dimension differs from image | For masks, ensure `(B, C, H, W)` matches spatial dims. If binary mask, use `(B, 1, H, W)`. |
| `Keypoints out of bounds after crop` | Keypoints drift outside image post-crop | Use `bbox_params` or `keypoints_params` in `AugmentationSequential` to handle dropping invalid points. |
| `torch.compile fails with control flow` | Random ops contain Python conditionals | Use deterministic transforms (`Normalize`, fixed crops) for export; randomized ops cause shape/control variance. |
| ONNX export error | Op not in ONNX opset | Some ops (thin-plate spline, advanced filters) don't have ONNX kernels. Stick to core: affine, brightness, contrast, crop. |

---

## DEVELOPER Mode Workflow

### 1. Verify Kornia Repository Root

Before authoring, confirm you are in the kornia repository:

```bash
# At repo root:
ls -la kornia/augmentation/__init__.py  # should exist
grep -r "AugmentationBase2D" kornia/augmentation/base.py  # verify base class
```

### 2. Scaffold a New Transform

Use the dev module to auto-generate structure (if it exists):

```bash
# If kornia has a scaffolding tool:
python -m kornia.augmentation.dev.scaffold new \
  --name RandomCustomBlur \
  --kind geometric \
  --domain 2d \
  --supports image,bbox,mask,keypoints \
  --invertible false

# Otherwise, follow the canonical pattern (see below).
```

### 3. Canonical Transform Pattern

All new transforms must follow this shape:

```python
from typing import Any, Optional
from pydantic import BaseModel, Field
from kornia.augmentation.base import _AugmentationBase
from kornia.augmentation._2d.base import RigidAffineAugmentationBase2D
import torch
import torch.nn.functional as F

# Step 1: Define Pydantic Params (required for schema and serialization)
class RandomCustomBlurParams(BaseModel):
    kernel_size: int = Field(gt=0, description="Blur kernel size (odd)")
    sigma: torch.Tensor  # Per-sample parameter, shape (B,)

    class Config:
        arbitrary_types_allowed = True

# Step 2: Implement Transform inheriting from Base2D or RigidAffineAugmentationBase2D
class RandomCustomBlur(RigidAffineAugmentationBase2D):
    r"""Apply random custom blur.

    Args:
        kernel_size: blur kernel size (must be odd).
        sigma: standard deviation range (min, max).
        p: probability of applying the augmentation.
        same_on_batch: apply same transformation across batch.
    """

    def __init__(
        self,
        kernel_size: int = 5,
        sigma: tuple[float, float] = (0.1, 2.0),
        p: float = 0.5,
        same_on_batch: bool = False,
        keepdim: bool = False,
    ):
        super().__init__(p=p, same_on_batch=same_on_batch, keepdim=keepdim)
        self.kernel_size = kernel_size
        self.sigma_range = sigma

    # Step 3: sample_params — return Pydantic Params with sampled tensors
    def sample_params(self, shape: torch.Size, device: torch.device, dtype: torch.dtype) -> dict[str, Any]:
        B = shape[0]
        sigma = torch.empty(B, device=device, dtype=dtype).uniform_(*self.sigma_range)
        return {"sigma": sigma}

    # Step 4: apply — core forward logic
    def apply_transform(
        self,
        input: torch.Tensor,
        params: dict[str, Any],
        transform_matrix: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # input: (B, C, H, W)
        sigma = params["sigma"]
        # Reshape for per-sample application
        output = []
        for i in range(input.shape[0]):
            img_i = input[i:i+1]  # (1, C, H, W)
            s_i = sigma[i].item()
            # Apply blur (deterministic per sampled sigma)
            blurred = torch.nn.functional.gaussian_blur(
                img_i,
                kernel_size=self.kernel_size,
                sigma=s_i
            )
            output.append(blurred)
        return torch.cat(output, dim=0)

    # Step 5 (optional): apply_to_mask, apply_to_boxes, apply_to_keypoints
    def apply_to_mask(self, input: torch.Tensor, params: dict[str, Any]) -> torch.Tensor:
        # Masks use same blur as images (or identity)
        return self.apply_transform(input, params)

    def apply_to_boxes(self, input: torch.Tensor, params: dict[str, Any]) -> torch.Tensor:
        # Blur doesn't affect box geometry (identity)
        return input

    def apply_to_keypoints(self, input: torch.Tensor, params: dict[str, Any]) -> torch.Tensor:
        # Blur doesn't affect keypoint coordinates (identity)
        return input

    # Step 6 (optional): inverse — undo transformation if invertible
    # If not invertible, omit this method
```

### 4. Test Discipline

Write tests using `BaseTester`:

```python
from testing.base import BaseTester
import torch
from kornia.augmentation import RandomCustomBlur

class TestRandomCustomBlur(BaseTester):

    def test_smoke(self, device, dtype):
        """Basic run with all device/dtype combos."""
        aug = RandomCustomBlur(p=1.0)
        img = torch.randn(2, 3, 32, 32, device=device, dtype=dtype)
        out = aug(img)
        assert out.shape == img.shape

    def test_cardinality(self, device, dtype):
        """Output shapes match input."""
        aug = RandomCustomBlur(p=1.0)
        for h, w in [(32, 32), (64, 48), (100, 200)]:
            img = torch.randn(4, 3, h, w, device=device, dtype=dtype)
            out = aug(img)
            assert out.shape == img.shape

    def test_exception(self, device, dtype):
        """Invalid inputs raise exceptions."""
        aug = RandomCustomBlur()
        # Wrong shape
        with self.assertRaises(RuntimeError):
            aug(torch.randn(3, 32, 32, device=device, dtype=dtype))  # missing batch

    def test_feature(self, device, dtype):
        """Numerical correctness: blur increases smoothness."""
        aug = RandomCustomBlur(sigma=(1.0, 1.0), p=1.0)  # Fixed sigma for determinism
        img = torch.randn(1, 3, 64, 64, device=device, dtype=dtype)
        # High contrast image should smooth under blur
        img_contrast = torch.cat([torch.ones(1, 3, 32, 32), torch.zeros(1, 3, 32, 32)], dim=2).to(device=device, dtype=dtype)
        out = aug(img_contrast)
        # Edges should be blurred (gradient magnitude decreases)
        assert out.shape == img_contrast.shape

    def test_gradcheck(self, device):
        """Gradients flow correctly."""
        aug = RandomCustomBlur(sigma=(1.0, 1.0), p=1.0)
        img = torch.randn(2, 3, 16, 16, device=device, requires_grad=True, dtype=torch.float64)
        # Use BaseTester.gradcheck
        self.gradcheck(aug, img)

    def test_dynamo(self, device, dtype, torch_optimizer):
        """torch.compile compatibility."""
        aug = RandomCustomBlur(p=1.0)
        img = torch.randn(2, 3, 32, 32, device=device, dtype=dtype)
        try:
            aug_compiled = torch.compile(aug)
            out_compiled = aug_compiled(img)
            # Compare to eager
            out_eager = aug(img)
            self.assert_close(out_compiled, out_eager, atol=1e-3, rtol=1e-3)
        except (RuntimeError, torch._dynamo.exc.Unsupported) as e:
            # torch.compile may fail on certain ops; document and skip
            pass
```

### 5. Run Tests and Verify Public API

Ensure all tests pass and goldens are stable:

```bash
# Run your test
pixi run test tests/augmentation/test_random_custom_blur.py -q

# Check for dtype/device coverage
pixi run test tests/augmentation/test_random_custom_blur.py --dtype=float32,float64 --device=cpu,cuda

# Run quick smoke
pixi run test-quick tests/augmentation/test_random_custom_blur.py
```

### 6. PR Checklist for New Transforms

Paste this checklist into your PR description:

```markdown
## Transform Addition Checklist

- [ ] **Tests Pass**: `pixi run test tests/augmentation/test_MY_TRANSFORM.py --dtype=float32,float64 --device=cpu,cuda` passes locally
- [ ] **Goldens Stable**: No changes to existing `*.golden` files; new goldens generated via script
- [ ] **Public API**: Transform exported in `kornia/augmentation/__init__.py` and `__all__`
- [ ] **Docstring**: Full docstring with Args, Returns, Examples sections
- [ ] **No Legacy Base**: Uses unified `Transform` or `RigidAffineAugmentationBase2D`, not legacy 5-class hierarchy

## Additional Guidance

- Base class choice: Use `RigidAffineAugmentationBase2D` for geometric (affine, homography). Use `AugmentationBase2D` for pixel-wise (color, blur).
- Invertibility: Implement `inverse()` only if transformation is truly reversible (e.g., affine with known matrix). Omit if approximate (e.g., JPEG compression).
- Export-readiness: Deterministic transforms (Normalize, crop at fixed size) can be torch.export-clean if no shape branching.
```

### 7. Performance Authoring Guidelines

Follow these patterns for hot augmentations (used millions of times):

```python
# GOOD: In-place ops, minimal allocations, matches kornia style (see color_jitter.py)
def apply_transform(self, input: torch.Tensor, params: dict[str, Any]) -> torch.Tensor:
    # Vectorize all per-sample ops at once, no loops
    alpha = params["alpha"]  # shape (B,)
    # Reshape and broadcast
    alpha_reshaped = alpha.view(input.shape[0], 1, 1, 1)
    # Single in-place operation
    output = input.mul_(alpha_reshaped)
    return output

# BAD: Python loops, excessive copies
def apply_transform(self, input: torch.Tensor, params: dict[str, Any]) -> torch.Tensor:
    alpha = params["alpha"]
    output = []
    for i in range(input.shape[0]):
        img_i = input[i:i+1].clone()
        img_i = img_i * alpha[i]  # creates new tensor
        output.append(img_i)
    return torch.cat(output, dim=0)  # unnecessary concatenation
```

Do NOT invent new performance patterns. If you need a hot op, examine `kornia/augmentation/_2d/intensity/color_jitter.py` for the reference pattern and mirror it.

### 8. Preset Recipes: When to Add

Only add a preset if:
1. It reproduces a **published recipe** from a paper or official model implementation.
2. You have a **reference** (link to paper, code repo, or model card).
3. The preset **combines 3+ existing transforms** in a specific order.

Example: DETR augmentation for object detection

```python
# kornia/augmentations/presets/detr.py
from kornia.augmentation import (
    AugmentationSequential, RandomHorizontalFlip, RandomAffine,
    Normalize, Resize
)

def build_detr_augmentation(input_size: int = 800):
    """DETR training augmentation (from https://arxiv.org/abs/2005.12677)."""
    return AugmentationSequential(
        RandomHorizontalFlip(p=0.5),
        RandomAffine(degrees=0, translate=(0.1, 0.1), p=0.5),
        Resize((input_size, input_size)),
        Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        data_keys=["input"],
    )
```

### 9. Deterministic Namespace: When to Add

Add a transform to `kornia.augmentations.deterministic.*` if:
1. It has **zero randomness** (no `sample_params`; only static params).
2. It is **export-clean** (torch.export and ONNX compatible).
3. It is **no_grad** throughout OR is `NormalizeWithGrad` (preserves gradients).

Examples:
- `Normalize` — deterministic, no_grad, export-clean. Add to deterministic.
- `NormalizeWithGrad` — deterministic, preserves gradients. Add to deterministic.
- `RandomCrop` — random, even with fixed seed. Never add to deterministic.

### 10. Contract Testing: One-Liner Verification

Every new transform must pass the contract test:

```python
from kornia.augmentations.testing import transform_contract

# Passes iff transform respects shape, dtype, device, batch invariants
transform_contract(RandomCustomBlur)
```

---

## Things This Skill Explicitly Does NOT Do

- **Recommend autograd through augmentations.** Augmentations run `no_grad` by default; only `NormalizeWithGrad` preserves gradients. Do not try to backprop through randomness.
- **Suggest writing custom RandomGenerator subclasses.** The unified `Transform` base's `sample_params()` covers 99% of use cases. Avoid the old `RandomGeneratorBase` unless you need very specific sampling logic.
- **Reach for the legacy 5-class hierarchy** (`RigidAffineAugmentationBase2D`, `IntensityAugmentationBase2D`, etc.) **for new code.** These are for back-compat. Use unified `AugmentationBase2D` or `RigidAffineAugmentationBase2D` for new transforms.
- **Suggest `data_keys=["input","bbox_xyxy"]` for new code.** Tagged tensors (via `Boxes`, `Keypoints`) are the preferred forward path. `data_keys` is for back-compat; removal target is kornia 1.0.
- **Recommend MCP (Model Context Protocol) integration.** There is no kornia MCP server. Use kornia as a library within your Python code or LLM context windows directly.

---

## Quick Reference Table

| Task | API | Notes |
|------|-----|-------|
| **Discover transforms** | `from kornia import augmentation as K` + `inspect.getmembers(K)` | 100+ transforms available; catalog is the source of truth |
| **Inspect schema** | `K.RandomAffine.__init__.__annotations__` or docstring | All public params documented; Pydantic schema available internally |
| **Build pipeline** | `K.AugmentationSequential(T1, T2, ..., data_keys=["input", ...])` | Deterministic chaining; GPU-native; supports bbox/mask/keypoints |
| **Spec round-trip** | Call `aug()`, inspect `aug._params`, reapply with `aug(..., params=aug._params)` | Enables reproducible augmentation; useful for logging/validation splits |
| **Migrate from Alb** | `from_albumentations(spec)` or manual port | Check for compat module; fallback to manual composition |
| **Migrate from TV v2** | `from_torchvision_v2(compose)` or manual port | Check for compat module; fallback to manual composition |
| **Compile** | `torch.compile(aug, fullgraph=True)` | JIT acceleration; stable shapes required; test on target device first |
| **CUDA graph** | Capture with `torch.cuda.graph()` after warm-up | Fastest inference with fixed batch/shape; 1–2x over torch.compile on GPUs |
| **torch.export** | `torch.export.export(aug, (example_input,))` | Export-clean for serving; deterministic transforms only; test early |
| **ONNX export** | `torch.onnx.export(aug, input, "aug.onnx", opset_version=14)` | Core ops (affine, brightness, crop) portable; advanced ops may fail |
| **Scaffold (dev)** | `python -m kornia.augmentations.dev.scaffold new --name X --kind Y ...` | Auto-generates boilerplate; adapt as needed |
| **Contract test (dev)** | `from kornia.augmentations.testing import transform_contract; transform_contract(MyTransform)` | One-liner verification; catches shape/dtype/device issues early |
| **Unit test (dev)** | Inherit from `BaseTester`; implement `test_smoke`, `test_cardinality`, `test_feature`, `test_gradcheck`, `test_dynamo` | Standard test harness; fixtures inject device/dtype; gradcheck via `self.gradcheck()` |
| **Performance test (dev)** | Benchmark on CPU/CUDA; record hardware/commit; compare before/after | Include timing tables and quality metrics in PR description |

---

## Quick Start Examples

### Example 1: Detection Pipeline (YOLO-style)

```python
import torch
import kornia.augmentation as K
from kornia.geometry.boxes import Boxes

# Build YOLO-like training augmentation
aug = K.AugmentationSequential(
    K.RandomHorizontalFlip(p=0.5),
    K.RandomAffine(degrees=30, translate=(0.1, 0.1), scale=(0.8, 1.2), p=0.8),
    K.RandomCrop((416, 416), p=1.0),
    K.RandomBrightness((0.5, 1.5), p=0.3),
    K.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, p=0.3),
    data_keys=["input", "bbox_xyxy"],
)

# Prepare batch
images = torch.randn(4, 3, 640, 640)
boxes = Boxes(torch.tensor([
    [[10., 10., 100., 100.], [200., 200., 300., 300.]],  # batch 0: 2 objects
    [[50., 50., 150., 150.]], [[0., 0., 50., 50.]], [[100., 100., 200., 200.]]  # other batches
]))

# Apply
images_aug, boxes_aug = aug(images, boxes)
print(f"Images: {images_aug.shape}, Boxes: {boxes_aug.shape}")
# Output: Images: torch.Size([4, 3, 416, 416]), Boxes: torch.Size([4, N, 4])
```

### Example 2: Export Normalization for Inference

```python
import torch
import kornia.augmentation as K

# Preprocessing for model serving
preprocess = K.AugmentationSequential(
    K.Resize((224, 224)),
    K.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
)

# Export to torch.export
input_example = torch.randn(1, 3, 480, 640)
exported = torch.export.export(preprocess, (input_example,))
print("Exported successfully; ready for ONNX or edge deployment")

# Verify
output_eager = preprocess(input_example)
output_exported = exported.module()(input_example)
print(f"Eager vs exported match: {torch.allclose(output_eager, output_exported)}")
```

### Example 3: Author a Geometric Transform (Developer)

```python
# File: kornia/augmentation/_2d/geometric/random_perspective_custom.py

import torch
from kornia.augmentation._2d.base import RigidAffineAugmentationBase2D
from kornia.geometry.transform import perspective

class RandomPerspectiveCustom(RigidAffineAugmentationBase2D):
    r"""Apply random perspective transformation with custom params."""

    def __init__(self, distortion_scale: float = 0.5, p: float = 0.5):
        super().__init__(p=p)
        self.distortion_scale = distortion_scale

    def sample_params(self, shape: torch.Size, device: torch.device, dtype: torch.dtype) -> dict:
        B, _, H, W = shape
        # Sample random homography perturbations
        start_points = torch.tensor(
            [[[0., 0.], [W - 1, 0.], [W - 1, H - 1], [0., H - 1]]]
        ).repeat(B, 1, 1).to(device=device, dtype=dtype)

        end_points = start_points + torch.empty_like(start_points).uniform_(
            -self.distortion_scale * H, self.distortion_scale * H
        )

        return {
            "start_points": start_points,
            "end_points": end_points,
        }

    def apply_transform(self, input: torch.Tensor, params: dict, transform_matrix=None) -> torch.Tensor:
        # Use kornia.geometry.transform.perspective
        H_mat = perspective(params["start_points"], params["end_points"])
        # Apply warp
        output = torch.nn.functional.grid_sample(
            input,
            F.affine_grid(H_mat[:, :2], input.shape),
            align_corners=False,
        )
        return output

# Test
if __name__ == "__main__":
    from testing.base import BaseTester

    class TestRandomPerspectiveCustom(BaseTester):
        def test_smoke(self, device, dtype):
            aug = RandomPerspectiveCustom(p=1.0)
            img = torch.randn(2, 3, 32, 32, device=device, dtype=dtype)
            out = aug(img)
            assert out.shape == img.shape

    tester = TestRandomPerspectiveCustom()
    tester.test_smoke('cpu', torch.float32)
    print("✓ Transform works!")
```
