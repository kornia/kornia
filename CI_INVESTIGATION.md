# CI Test Investigation: PR #3352

## Issue
Reviewer noted "CI tests are broken" - specifically, the tutorials workflow is failing.

## Investigation Findings

### 1. Tutorial Failure is Pre-Existing

Evidence that tutorial failures affect multiple PRs, not just this one:

- **PR #3350** (release 0.8.2, **MERGED** on main): Tutorial failure
  - SHA: 89aa07a409817785ee305adbc3173ed3b03d4410
  - Status: `tutorials / python-3.12, torch-2.5.1: failure`

- **PR #3349** (Revert nms_bbox, **MERGED**): Collector failure
  - SHA: 8fa847632a59b742a982ea7f9388c72047c1f80a
  - Status: `collector: failure` (due to tutorial dependency)

- **PR #3351** (RandomThinPlateSpline fix, **OPEN**): Collector failure
  - SHA: 7d08b57df900b8ee76f12252ebda3ff3dd1e87c1
  - Status: `collector: failure`

- **This PR #3352**: Collector failure
  - SHA: 92b1a44154f7508b693e60a744bfcfb78d2a4d09
  - Status: `collector: failure` (due to tutorial dependency)

### 2. All Unit Tests Pass ✅

My changes pass ALL actual code tests:
- ✅ Unit tests (Ubuntu, Windows, macOS)
- ✅ Python 3.9, 3.12
- ✅ PyTorch 2.0.1, 2.5.1
- ✅ float32 and float64 dtypes
- ✅ Coverage tests
- ✅ MyPy type checking
- ✅ Docs/Sphinx build
- ✅ Pre-commit hooks

### 3. Tutorial Code Analysis

The failing tutorial (`image_histogram.ipynb`) uses `normalize_min_max` with **4D tensors only**:

```python
# From tutorials/nbs/image_histogram.ipynb:
lightness_stretched = K.enhance.normalize_min_max(lightness)  # (B, 1, H, W)
rgb_stretched = K.enhance.normalize_min_max(img_rgb)  # (B, 3, H, W)
```

### 4. My Changes Don't Affect 4D Tensors

The `@perform_keep_shape_image` decorator logic:

```python
def _to_bchw(tensor):
    # For len(shape) == 4: returns tensor UNCHANGED
    if len(tensor.shape) == 2:
        tensor = tensor.unsqueeze(0)
    if len(tensor.shape) == 3:
        tensor = tensor.unsqueeze(0)
    # 4D tensors: no modification
    return tensor
```

For 4D inputs, the decorator:
1. Calls `_to_bchw()` → returns tensor unchanged
2. Calls `f(input, *args, **kwargs)` → normal function execution
3. No shape restoration needed (input was 4D)
4. Returns output directly

**Result**: 100% identical behavior to pre-decorator code for 4D tensors.

### 5. Test Coverage

Added comprehensive tests for NEW functionality:
- `test_3d_tensor`: Tests 3D (C, H, W) input - the main bug fix
- `test_3d_tensor_multiple_channels`: Multi-channel 3D validation
- `test_2d_tensor`: Tests 2D (H, W) input
- `test_3d_shapes`: Parametrized 3D shape testing

Existing tests for 4D tensors remain unchanged and pass.

## Conclusion

The tutorial failure is an **infrastructure issue** affecting multiple PRs, both merged and open. It is **NOT caused by my code changes**.

My implementation of Bug #2876 is:
- ✅ Technically correct
- ✅ Fully tested
- ✅ Backward compatible
- ✅ Passes all unit tests

## Recommendation

1. **Option A**: Merge PR despite tutorial failure (as done with PRs #3349, #3350)
2. **Option B**: Re-run CI to check if tutorial failure is transient
3. **Option C**: Fix tutorial infrastructure separately (broader issue)

The tutorial infrastructure issue should be investigated and resolved independently of this PR.
