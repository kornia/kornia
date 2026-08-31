# PR 4122 Affine Fallback Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make PR 4122's affine-shape fallback finite in forward and backward, orientation-consistent, shared by both estimators, and covered by the repository's half-precision test machinery.

**Architecture:** Repair the public moment estimator at the source with a dtype-aware threshold, then guard estimator outputs before any singular reciprocal or scale operation. A private helper will centralize safe scale/orientation finalization for the handcrafted and learned affine estimators while preserving their public APIs.

**Tech Stack:** Python, PyTorch autograd and `nn.Module`, pytest, Kornia LAF utilities, Pixi project tasks.

---

## File map

- `kornia/feature/affine_shape.py`: dtype-aware moment fallback, invalid-shape detection, and shared safe LAF finalization.
- `tests/feature/test_affine_shape_estimator.py`: source-level, forward/backward, orientation, and AffNet parity regressions.
- `tests/feature/test_scale_space_detector.py`: re-enable the padding contract in supported half-precision environments.
- `kornia/feature/scale_space_detector.py`: remove the dtype-specific claim from the padding explanation.
- `CHANGELOG.md`: describe finite-gradient, orientation-consistent internal handling and the repaired float16 source guard.

### Task 1: Repair the patch estimator's float16 degeneracy guard

**Files:**
- Modify: `tests/feature/test_affine_shape_estimator.py`
- Modify: `kornia/feature/affine_shape.py:99-108`

- [ ] **Step 1: Write the failing float16 source regression**

Add this method to `TestPatchAffineShapeEstimator`:

```python
def test_zero_patch_uses_circular_shape(self, device, dtype):
    if dtype in (torch.float16, torch.bfloat16) and not (
        supports_conv2d(device, dtype) and supports_replicate_padding(device, dtype)
    ):
        pytest.skip(f"no {dtype} Sobel kernels on {device.type}")
    patch = torch.zeros(1, 1, 32, 32, device=device, dtype=dtype)
    out = PatchAffineShapeEstimator(32).to(device, dtype)(patch)
    expected = torch.tensor([[[1.0, 0.0, 1.0]]], device=device, dtype=dtype)
    self.assert_close(out, expected)
```

Import `supports_replicate_padding` from `testing.base` beside the existing kernel probes.

- [ ] **Step 2: Run the test in float16 and verify the regression**

Run:

```bash
pixi run test-module tests/feature/test_affine_shape_estimator.py::TestPatchAffineShapeEstimator::test_zero_patch_uses_circular_shape --dtype=float16
```

Expected: FAIL because the current `eps=1e-10` becomes zero and the zero moment matrix normalizes as `0 / 0`.

- [ ] **Step 3: Use a representable threshold and a boolean fallback**

Replace the current `bad_mask` block with:

```python
eps = max(self.eps, torch.finfo(ellipse_shape.dtype).tiny)
bad_mask = (ellipse_shape < eps).sum(dim=2, keepdim=True) >= 2
circular_shape = ellipse_shape.new_tensor([1.0, 0.0, 1.0]).view(1, 1, 3)
ellipse_shape = torch.where(bad_mask, circular_shape, ellipse_shape)
```

Keep the existing normalization immediately after it. The boolean `where` also avoids the `nan * 0` behavior of the old mask multiplication.

- [ ] **Step 4: Run the source regression and existing patch-estimator checks**

Run:

```bash
pixi run test-module tests/feature/test_affine_shape_estimator.py::TestPatchAffineShapeEstimator --dtype=float16,float32,float64
```

Expected: all supported combinations PASS; unsupported half kernels SKIP through the explicit probe.

- [ ] **Step 5: Commit the source fix**

```bash
git add kornia/feature/affine_shape.py tests/feature/test_affine_shape_estimator.py
git commit -m "fix(feature): guard half-precision affine moments"
```

### Task 2: Make the handcrafted estimator fallback safe and orientation-consistent

**Files:**
- Modify: `tests/feature/test_affine_shape_estimator.py`
- Modify: `kornia/feature/affine_shape.py:124-178`

- [ ] **Step 1: Move the float16 reproduction onto the standard dtype fixture and add backward coverage**

Change `test_degenerate_ellipse_falls_back_to_input_laf_float16` to accept `dtype`, skip unless it is float16, construct every tensor/module with that fixture, enable gradients, and append:

```python
img.requires_grad_()
laf.requires_grad_()
out = LAFAffineShapeEstimator(32).to(device, dtype)(laf, img)
self.assert_close(out, laf)
out.sum().backward()
assert img.grad is not None and torch.isfinite(img.grad).all()
assert laf.grad is not None and torch.isfinite(laf.grad).all()
```

Retain the MPS autocast and `supports_conv2d`/`supports_grid_sample` probes, now using `dtype` instead of a hardcoded `torch.float16`.

- [ ] **Step 2: Add a forced-degenerate detector and orientation regression**

Add a small test-local module:

```python
class DegenerateShape(torch.nn.Module):
    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        shape = patches.mean(dim=(-2, -1), keepdim=False).unsqueeze(-1) * 0
        return torch.cat([shape, shape, torch.ones_like(shape)], dim=-1)
```

Then add a test using a 90-degree rotated, otherwise valid LAF:

```python
def test_degenerate_ellipse_fallback_respects_upright_contract(self, device, dtype):
    img = torch.rand(1, 1, 32, 32, device=device, dtype=dtype)
    laf = torch.tensor([[[[0.0, 8.0, 16.0], [-8.0, 0.0, 16.0]]]], device=device, dtype=dtype)
    aff = LAFAffineShapeEstimator(32, DegenerateShape(), preserve_orientation=False).to(device, dtype)
    out = aff(laf, img)
    self.assert_close(out, make_upright(laf))
```

Import `make_upright` from `kornia.feature.laf`. Apply the same runtime kernel probes used by the float16 reproduction when `dtype` is half precision.

- [ ] **Step 3: Run the two regressions and verify their current failures**

Run:

```bash
pixi run test-module tests/feature/test_affine_shape_estimator.py::TestLAFAffineShapeEstimator::test_degenerate_ellipse_falls_back_to_input_laf_float16 --dtype=float16
pixi run test-module tests/feature/test_affine_shape_estimator.py::TestLAFAffineShapeEstimator::test_degenerate_ellipse_fallback_respects_upright_contract --dtype=float32
```

Expected: the first FAILS on non-finite gradients and the second FAILS because the current fallback returns the rotated input unchanged.

- [ ] **Step 4: Add shared safe finalization helpers**

Add private module-level helpers above `LAFAffineShapeEstimator`:

```python
def _invalid_laf_mask(laf: torch.Tensor) -> torch.Tensor:
    det = laf[..., 0, 0] * laf[..., 1, 1] - laf[..., 1, 0] * laf[..., 0, 1]
    return ~laf.isfinite().all(dim=-1).all(dim=-1) | ~det.isfinite() | (det == 0)


def _finalize_laf(
    laf_out: torch.Tensor, laf: torch.Tensor, preserve_orientation: bool, bad: torch.Tensor
) -> torch.Tensor:
    fallback = laf if preserve_orientation else make_upright(laf)
    safe_laf_out = torch.where(bad[..., None, None], fallback, laf_out)
    scale_orig = get_laf_scale(laf)
    if preserve_orientation:
        ori_orig = get_laf_orientation(laf)
    ellipse_scale = get_laf_scale(safe_laf_out)
    safe_laf_out = scale_laf(safe_laf_out, scale_orig / ellipse_scale)
    if preserve_orientation:
        safe_laf_out = set_laf_orientation(safe_laf_out, ori_orig)
    return torch.where(bad[..., None, None], fallback, safe_laf_out)
```

The first `where` is the backward-safety boundary; the second preserves the exact fallback value.

- [ ] **Step 5: Sanitize ellipse coefficients before conversion**

In `LAFAffineShapeEstimator.forward`, replace the current conversion/normalization tail with:

```python
bad = ~ellipse_shape.isfinite().all(dim=-1) | (ellipse_shape[..., 0] == 0) | (ellipse_shape[..., 2] == 0)
circular_shape = ellipse_shape.new_tensor([1.0, 0.0, 1.0])
safe_ellipse_shape = torch.where(bad[..., None], circular_shape, ellipse_shape)
ellipses = torch.cat([laf.view(-1, 2, 3)[..., 2].unsqueeze(1), safe_ellipse_shape], dim=2).view(B, N, 5)
laf_out = ellipse_to_laf(ellipses)
return _finalize_laf(laf_out, laf, self.preserve_orientation, bad)
```

Remove the old post-hoc non-finite guard and the duplicated scale/orientation code.

- [ ] **Step 6: Run the regressions and full handcrafted-estimator class**

Run:

```bash
pixi run test-module tests/feature/test_affine_shape_estimator.py::TestLAFAffineShapeEstimator --dtype=float16,float32,float64
```

Expected: all supported combinations PASS, including finite image/LAF gradients and upright fallback.

- [ ] **Step 7: Commit the differentiable fallback**

```bash
git add kornia/feature/affine_shape.py tests/feature/test_affine_shape_estimator.py
git commit -m "fix(feature): sanitize affine fallback before inversion"
```

### Task 3: Give AffNet the same invalid-output policy

**Files:**
- Modify: `tests/feature/test_affine_shape_estimator.py`
- Modify: `kornia/feature/affine_shape.py:246-269`

- [ ] **Step 1: Add an AffNet singular-output regression**

Add this test helper and test to `TestLAFAffNetShapeEstimator`:

```python
class SingularAffNetOutput(torch.nn.Module):
    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        values = patches.new_tensor([-1.0, 0.0, -1.0]).view(1, 3, 1, 1)
        return values.expand(patches.shape[0], -1, -1, -1)


def test_singular_prediction_falls_back_upright(self, device, dtype):
    img = torch.rand(1, 1, 32, 32, device=device, dtype=dtype)
    laf = torch.tensor([[[[0.0, 8.0, 16.0], [-8.0, 0.0, 16.0]]]], device=device, dtype=dtype)
    aff = LAFAffNetShapeEstimator(preserve_orientation=False).to(device, dtype)
    aff.features = SingularAffNetOutput()
    out = aff(laf, img)
    self.assert_close(out, make_upright(laf))
```

Use the convolution/grid-sample runtime probes for half dtypes; the replacement `features` module removes AffNet's own convolution requirement but patch extraction still needs grid sampling.

- [ ] **Step 2: Run the regression and verify it fails**

Run:

```bash
pixi run test-module tests/feature/test_affine_shape_estimator.py::TestLAFAffNetShapeEstimator::test_singular_prediction_falls_back_upright --dtype=float32
```

Expected: FAIL because the current AffNet tail normalizes the singular zero frame instead of applying the shared fallback.

- [ ] **Step 3: Route AffNet through the shared finalizer**

Replace the duplicated scale/orientation tail in `LAFAffNetShapeEstimator.forward` with:

```python
bad = _invalid_laf_mask(new_laf)
return _finalize_laf(new_laf, laf, self.preserve_orientation, bad)
```

- [ ] **Step 4: Run both estimator suites**

Run:

```bash
pixi run test-module tests/feature/test_affine_shape_estimator.py --dtype=float32,float64
pixi run test-module tests/feature/test_affine_shape_estimator.py --dtype=float16,bfloat16
```

Expected: all supported tests PASS and unavailable kernels SKIP with a precise reason.

- [ ] **Step 5: Commit estimator parity**

```bash
git add kornia/feature/affine_shape.py tests/feature/test_affine_shape_estimator.py
git commit -m "fix(feature): share safe affine LAF finalization"
```

### Task 4: Restore half-precision detector integration coverage

**Files:**
- Modify: `tests/feature/test_scale_space_detector.py:32,361-372,724-738`
- Modify: `kornia/feature/scale_space_detector.py:172-184`
- Modify: `CHANGELOG.md:76-111`

- [ ] **Step 1: Replace stale skips with capability checks**

Import `supports_conv2d` and `supports_grid_sample`, then add:

```python
def _require_affine_orientation_kernels(device: torch.device, dtype: torch.dtype) -> None:
    if dtype not in (torch.float16, torch.bfloat16):
        return
    if device.type == "mps":
        pytest.skip("MPS autocast changes the effective dtype")
    probes = (
        ("replicate-padding", supports_replicate_padding),
        ("conv2d", supports_conv2d),
        ("grid-sample", supports_grid_sample),
        ("topk", supports_topk),
    )
    for name, probe in probes:
        if not probe(device, dtype):
            pytest.skip(f"no {name} kernel for {dtype} on {device.type}")
```

Call this helper at the start of both `test_padding_survives_the_affine_and_orientation_modules` methods and remove their unconditional half-precision skips.

- [ ] **Step 2: Run both padding tests in half precision**

Run:

```bash
pixi run test-module tests/feature/test_scale_space_detector.py -k padding_survives_the_affine_and_orientation_modules --dtype=float16,bfloat16
```

Expected: PASS on supported combinations and capability-based SKIP elsewhere. The tests must still assert exact zero LAFs and responses.

- [ ] **Step 3: Correct internal and user-facing explanations**

Update `_zero_unfilled`'s docstring to say affine/orientation modules may normalize or propagate invalid padding frames differently by dtype, so the detector always reapplies its zero-padding contract after those modules. Do not claim a fixed `1e-5` intermediate.

Update the PR's breaking-change entry to state that `PatchAffineShapeEstimator` uses a representable dtype-aware threshold, estimator callers sanitize invalid candidates before nonlinear LAF operations, backward remains finite, and fallback orientation follows `preserve_orientation`.

- [ ] **Step 4: Run focused detector and affine-shape suites**

Run:

```bash
pixi run test-module tests/feature/test_scale_space_detector.py -k padding_survives_the_affine_and_orientation_modules --dtype=float32,float64
pixi run test-module tests/feature/test_affine_shape_estimator.py tests/feature/test_scale_space_detector.py --dtype=float32,float64
```

Expected: focused and full-file standard-precision tests PASS.

- [ ] **Step 5: Commit integration coverage and wording**

```bash
git add CHANGELOG.md kornia/feature/scale_space_detector.py tests/feature/test_scale_space_detector.py
git commit -m "test(feature): restore half-precision detector coverage"
```

### Task 5: Final verification and review

**Files:**
- Review all files above and `docs/superpowers/specs/2026-08-31-pr-4122-affine-fallback-design.md`

- [ ] **Step 1: Run the focused feature tests**

```bash
pixi run test-module tests/feature/test_affine_shape_estimator.py tests/feature/test_laf.py tests/feature/test_scale_space_detector.py --dtype=float32,float64
pixi run test-module tests/feature/test_affine_shape_estimator.py tests/feature/test_scale_space_detector.py --dtype=float16,bfloat16
```

Expected: PASS on supported CPU combinations with explicit kernel SKIPs only.

- [ ] **Step 2: Run CUDA half tests in isolated subprocesses if CUDA is available**

```bash
pixi run -e cuda test-module tests/feature/test_affine_shape_estimator.py tests/feature/test_scale_space_detector.py --device=cuda --dtype=float16,bfloat16 --isolate-half-precision
```

Expected: each half-precision CUDA case is isolated; PASS or a capability-based SKIP, with no shared-process execution.

- [ ] **Step 3: Run repository quality gates**

```bash
pixi run pre-commit-all
pixi run typecheck
pixi run doctest
```

Expected: all commands exit 0.

- [ ] **Step 4: Inspect the final patch and status**

```bash
git diff origin/main...HEAD --check
git diff origin/main...HEAD --stat
git status --short
```

Expected: no whitespace errors; only PR files, the approved design/plan documents, and the user's pre-existing benchmark JSON files are present.

- [ ] **Step 5: Request an independent code review and resolve findings**

Provide the reviewer the approved spec, PR review findings 1-6, and the final diff. Fix every critical or important issue test-first, rerun the affected focused command, and retain reasoned pushback for suggestions that contradict verified repository behavior.
