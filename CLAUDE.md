# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## About Kornia

Kornia is a differentiable computer vision library built on PyTorch. It provides differentiable image processing, geometric vision algorithms, augmentation pipelines, and pre-trained AI models (feature detection/matching, segmentation, etc.).

## Development Environment

The project uses [pixi](https://pixi.sh) for environment management and `uv` for Python package management.

```bash
# Set up development environment (Python 3.11 default)
pixi install

# For specific Python versions
pixi install -e py312
pixi install -e py313

# For CUDA development
pixi run -e cuda install
```

## Common Commands

```bash
# Run all tests
pixi run test

# Run a specific test file
pixi run test tests/test_geometry_transform.py

# Run tests with pytest options (dtype: bfloat16, float16, float32, float64, all; device: cpu, cuda, mps, tpu, all)
pixi run test tests/test_geometry_transform.py --dtype=float32,float64 --device=all

# Run quick tests (excludes jit, grad, nn)
pixi run test-quick

# Linting (ruff via pre-commit)
pixi run lint

# Type checking (uses `ty`)
pixi run typecheck

# Doctests
pixi run doctest

# Build documentation
pixi run build-docs

# Run tests with specific device/dtype via env vars
KORNIA_TEST_DEVICE=cuda KORNIA_TEST_DTYPE=float32 pixi run test
KORNIA_TEST_RUNSLOW=true pixi run test-slow
```

## Code Architecture

The library is structured as submodules under `kornia/`:

- **`kornia/filters/`** — Image filtering (Gaussian, Sobel, Median, Canny, etc.). Imported first in `__init__.py` as it's core.
- **`kornia/geometry/`** — Geometric transformations (affine, homography, camera models, stereo, 3D). Also core, imported first.
- **`kornia/augmentation/`** — Augmentation pipeline (`AugmentationSequential`, `RandomAffine`, etc.)
- **`kornia/color/`** — Color space conversions (RGB, HSV, grayscale, etc.)
- **`kornia/feature/`** — Feature detection and description (SIFT, HardNet, DISK, DeDoDe, LoFTR, LightGlue, etc.)
- **`kornia/enhance/`** — Image enhancement (histogram equalization, CLAHE, gamma correction)
- **`kornia/losses/`** — Loss functions (SSIM, PSNR, Dice, Hausdorff, etc.)
- **`kornia/models/`** — Pre-trained AI models (YuNet face detection, SAM segmentation, etc.)
- **`kornia/morphology/`** — Morphological operations (dilation, erosion, etc.)
- **`kornia/onnx/`** — ONNX export and inference (`ONNXSequential`)
- **`kornia/contrib/`** — Experimental/contributed modules
- **`kornia/core/`** — Base classes (`ImageModule`, `ImageSequential`, `TensorWrapper`, ONNX mixins)
- **`kornia/transpiler/`** — Multi-framework support via ivy (JAX, TensorFlow, NumPy backends)
- **`testing/`** — Test utilities (not tests). `testing/base.py` contains `BaseTester` and `assert_close`.

**Import order matters**: `filters` and `geometry` must be imported before other modules to avoid circular dependencies (see `kornia/__init__.py`).

## Testing Patterns

All tests should inherit from `BaseTester` (from `testing.base`):

```python
from testing.base import BaseTester

class TestMyFunction(BaseTester):
    def test_smoke(self, device, dtype): ...          # Basic run with all arg combinations
    def test_exception(self, device, dtype): ...      # Exception cases
    def test_cardinality(self, device, dtype): ...    # Output shapes
    def test_feature(self, device, dtype): ...        # Correctness / numerical accuracy
    def test_gradcheck(self, device): ...             # Gradient checking via self.gradcheck()
    def test_dynamo(self, device, dtype, torch_optimizer): ...  # torch.compile compat
```

The `device` and `dtype` fixtures are injected automatically. Use `self.assert_close()` for tensor comparisons. Test configurations are driven by env vars `KORNIA_TEST_DEVICE` and `KORNIA_TEST_DTYPE`, or by `--device`/`--dtype` pytest args.

## Coding Standards

- **Python >= 3.11** with `from __future__ import annotations` for non-JIT modules
- **Line length**: 120 characters (ruff enforced)
- **Type hints required** on all function inputs and outputs; use `torch.Tensor` directly (not string annotations for tensor types in JIT-compatible code)
- **Only PyTorch** as a third-party dependency — no other libraries
- **Use existing `kornia` utilities** rather than reimplementing with raw PyTorch
- **Docstrings**: Follow existing codebase style; all public APIs need docstrings
- Every source file must start with the Apache 2.0 license header (managed by `add-license-header`)

## Benchmarks

Scripts under `benchmarks/` measure the speed and/or quality of existing kornia functions, modules, or models. Each benchmark must:

- Report **CPU and CUDA timings** in a table.
- Include **quality metrics** where applicable (see `benchmarks/feature/` for an example with local-feature matching scores).
- Record the **date, hardware description, and git commit hash** being evaluated at the top of the output or in a results file.
- Benchmark **only the public kornia API** — no custom reimplementations or alternative snippets inside the script.

### Workflow for performance PRs

1. Check out `main` (or the relevant release tag) and run the benchmark to establish a baseline.
2. Apply your changes on a new branch and run the same benchmark again.
3. Include both result tables in the PR description so reviewers can compare before and after.

## Pre-commit Hooks

Install hooks with `pre-commit install`. CI enforces ruff formatting, linting, and docformatter.

## Documentation and Visualizations

When adding a new feature detector or descriptor to `kornia/feature/`:
- Add an entry to the `responses` list in `docs/generate_examples.py` with a corresponding `elif` block that produces a heatmap/score visualization (`(B, 3, H, W)` BGR image in `img_in`, `(B, 3, H, W)` response map in `out`).
- See existing entries (`DISK`, `ALIKED`, `XFeat`) for the expected pattern.

## PR Requirements

All PRs must:
- Be linked to a previously discussed GitHub issue or Discord discussion (`Fixes #123`)
- Include pasted local test log output as proof of execution (`pixi run test ...`)
- Reference an algorithm source (PyTorch, OpenCV, scikit-image, paper, etc.) for any new implementation

**Comments**: No redundant or ghost comments (e.g., "this returns the input tensor", or comments explaining deleted code). Violation triggers mandatory manual rewrite request.

See `AI_POLICY.md` for the full contribution policy, including AI usage disclosure requirements.
