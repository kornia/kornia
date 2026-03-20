# Testing Guide

This document explains how to run the Kornia test suite, what the common pytest flags are, and how to handle recurring classes of test failures.

## Running Tests

Kornia uses [pixi](https://pixi.sh) to manage the test environment.

```bash
# All tests, default device (cpu) and dtype (float32)
pixi run test

# Specific file
pixi run test tests/geometry/test_boxes.py

# Specific device and dtype
pixi run test tests/ --device=cuda --dtype=float32

# All devices and dtypes
pixi run test tests/ --device=all --dtype=all

# Skip slow tests (default); include them with --runslow
pixi run test tests/ --runslow

# Quick tests (excludes jit, grad, nn markers)
pixi run test-quick
```

### CLI Options

| Option | Env var | Default | Description |
|---|---|---|---|
| `--device` | `KORNIA_TEST_DEVICE` | `cpu` | `cpu`, `cuda`, `mps`, `tpu`, or `all` |
| `--dtype` | `KORNIA_TEST_DTYPE` | `float32` | `float32`, `float64`, `float16`, `bfloat16`, or `all` |
| `--runslow` | `KORNIA_TEST_RUNSLOW` | off | Include `@pytest.mark.slow` tests |
| `--tf32` | `KORNIA_TEST_TF32` | off | Enable TF32 mode (see below) |
| `--optimizer` | `KORNIA_TEST_OPTIMIZER` | `inductor` | `torch.compile` backend for dynamo tests |
| `--isolate-half-precision` | `KORNIA_TEST_ISOLATE_HALF` | off | Run float16/bfloat16 CUDA tests each in a fresh `subprocess.run` process (no shared CUDA state) |

## Test Structure

All tests inherit from `testing.base.BaseTester` and implement a standard set of methods:

```python
from testing.base import BaseTester

class TestMyFunction(BaseTester):
    def test_smoke(self, device, dtype): ...        # basic run, all arg combinations
    def test_exception(self, device, dtype): ...    # error paths
    def test_cardinality(self, device, dtype): ...  # output shapes
    def test_feature(self, device, dtype): ...      # correctness / numerical accuracy
    def test_gradcheck(self, device): ...           # gradient checking via self.gradcheck()
    def test_dynamo(self, device, dtype, torch_optimizer): ...  # torch.compile compat
```

The `device` and `dtype` fixtures are injected automatically from the CLI options.  Use `self.assert_close()` for tensor comparisons — it automatically selects tolerances appropriate for the dtype.

## Markers

| Marker | Meaning |
|---|---|
| `@pytest.mark.slow` | Long-running test; skipped unless `--runslow` is passed |
| `@pytest.mark.grad` | Gradient-check test |
| `@pytest.mark.jit` | TorchScript test |
| `@pytest.mark.nn` | Module-level test |
| `@pytest.mark.tf32` | Known to fail under TF32 (see section below); xfail unless `--tf32` |

---

## Known Sources of Test Failures

### 1. TF32 (TensorFloat-32) Precision

**What it is.** `torch.set_float32_matmul_precision("high")` enables TF32 mode for CUDA matrix multiplications (`torch.bmm`, `torch.mm`, etc.). TF32 truncates float32 inputs to a 10-bit mantissa before the multiply-accumulate, giving roughly float16 mantissa precision for those ops. This is the default when `torch.compile` is in use and is enabled by many deep-learning frameworks for throughput.

**Effect on tests.** Numerically sensitive tests that compare float32 outputs of matrix operations against hardcoded expected values (or against a CPU reference) can fail because the accumulated rounding error exceeds the test tolerance.

**In Kornia's test suite.** TF32 is **off by default**. Pass `--tf32` (or set `KORNIA_TEST_TF32=true`) to enable it. Tests that are known to be sensitive to TF32 are marked `@pytest.mark.tf32`; without `--tf32` they are marked `xfail` so the suite stays green.

**Fixing a TF32-sensitive test.** Prefer fixing the test data over relaxing tolerances:

- Use integer-valued coordinates — all integers up to 2047 are exactly representable in TF32 (10-bit mantissa, so $n < 2^{10}$; integers up to 2047 via powers-of-two representation).
- Restrict inputs to ranges that keep intermediate values well within the TF32-exact region (e.g. pixel coordinates within image bounds rather than ±1500).
- For camera/geometry tests, avoid near-zero depth values and use realistic intrinsic matrices (e.g. `fx=500, cx=256`) rather than fully random ones.
- Only relax `atol`/`rtol` **as a last resort**, and only when the new tolerance is still below 0.01.

**Example: 3D box transforms.**  Float coordinates like `z=283.162` fall in the TF32 range [256, 512) where the representable step is 0.25.  Rounding `283.162 → 283.25` then computing `2×283.25+1 = 567.5` instead of the expected `567.324` gives an error of 0.176 — which fails with `atol=1e-4` but is not meaningful.  The fix is to use integer coordinates (`z=284`) which are TF32-exact.

---

### 2. Device-Dependent PRNG

**What it is.** `torch.rand(..., device='cuda')` uses a different random number generator (Philox) than the CPU (Mersenne Twister). Even with the same seed set by `torch.manual_seed(n)`, the two devices produce **different sequences**.

**Effect on tests.** Tests that:
1. Generate random tensors directly on a non-CPU device, AND
2. Compare against hardcoded expected values computed on CPU

will fail on CUDA/MPS even though the code is correct.

**In Kornia's test suite.** This affects augmentation tests with seeded expected values (`TestRandomRGBShift`, `TestRandomMixUpGen`), color roundtrip tests (`TestLuvToRgb`), and any test that runs `torch.rand(..., device=device)` without a device-independent strategy.

**Fixes (in order of preference):**

1. **Generate on CPU, move to device.** This is fully device-agnostic:
   ```python
   torch.manual_seed(42)
   data = torch.rand(3, 4, 5).to(device=device, dtype=dtype)
   ```

2. **Skip for non-CPU when checking specific values.**  Follow the existing pattern used in augmentation generator tests:
   ```python
   if device.type != "cpu":
       pytest.skip("Random number sequences differ between CPU and non-CPU devices")
   ```

3. **Test properties instead of exact values** (e.g., check that output is in [0, 1] rather than matching a specific tensor).

**Note.** `torch.manual_seed` seeds all devices in PyTorch ≥ 1.8, but the sequences still differ per device because the underlying algorithms differ.

---

### 3. CUDA Non-Determinism in Backward

**What it is.** Some CUDA kernels use `atomicAdd` for scatter and reduction operations. The order of floating-point additions is non-deterministic across runs, producing slightly different gradient values each time backward is called.

**Effect on tests.** `torch.autograd.gradcheck` calls backward twice with the same inputs and checks that the results are bit-identical (`nondet_tol=0.0` by default). If the op uses atomics, gradcheck raises `GradcheckError: Backward is not reentrant`.

**Affected operations.** Histogram-based orientation estimators (`LAFOrienter`), ALIKED backbone (scatter/pool ops), and any custom op that uses `torch.scatter_add` or `atomicAdd` on CUDA.

**Fix.** Pass `nondet_tol` to gradcheck:
```python
# In a test using self.gradcheck():
self.gradcheck(fn, inputs, rtol=1e-3, atol=1e-3, nondet_tol=1e-3)

# In a test calling torch.autograd.gradcheck() directly:
torch.autograd.gradcheck(fn, inputs, eps=1e-4, atol=1e-3, rtol=1e-3,
                         fast_mode=True, nondet_tol=1e-3)
```

A value of `1e-3` is usually sufficient; it should not exceed `atol`.

---

### 4. Test-Order Dependencies (Full Suite vs. Isolation)

**What it is.** A test can pass in isolation but fail when preceded by other tests, because some global state was mutated by an earlier test.

**Common sources:**
- **CUDA RNG state**: unseeded `torch.rand(..., device='cuda')` draws from the CUDA RNG, whose state depends on all prior CUDA random operations in the process.
- **`torch.set_float32_matmul_precision`**: if any test (or the conftest warmup) sets this to `"high"`, all subsequent CUDA matmuls use TF32.
- **`torch.use_deterministic_algorithms`**: a test enabling deterministic mode affects all later tests.
- **Model caches / lazy initialisation**: some feature extractors load weights on first call and cache them globally.

**Diagnosis.** If a test fails in the full suite but passes in isolation, run it with `--randomly-seed=last` (if `pytest-randomly` is installed) to reproduce the ordering, or prefix the failing test with the suspected culprit and check if the failure disappears.

**Fix.** Always seed RNG state explicitly in tests that compare against reference values, and prefer generating random data on CPU:
```python
torch.manual_seed(0)
data = torch.rand(B, C, H, W).to(device=device, dtype=dtype)
```

---

### 5. Float32 Numerical Precision in Geometry/Camera Tests

**What it is.** Operations like camera projection, LuV color conversion, and homography estimation involve divisions and non-linear functions. In float32, the roundtrip error can be significant for extreme inputs.

**Common anti-patterns (and fixes):**

| Anti-pattern | Problem | Fix |
|---|---|---|
| Depth in `[-500, 500]` | Near-zero depth → `1/z` blow-up | Restrict depth to `[1, 500]` |
| Pixel coords in `[-1500, 1500]` | Far outside image; large TF32 rounding | Use `[0, W) × [0, H)` |
| Fully random `K` matrix | Unrealistic intrinsics | Use `fx=500, cx=256` or similar |
| Fully random rotation matrix | May not be a valid rotation | Use `axis_angle_to_rotation_matrix` |
| `torch.rand(B, N, 2)` for homography points | Random degenerate configs | Use `create_random_homography` from `testing.geometry.create` |

---

### 6. SVD Numerical Stability (float32 on CUDA)

**What it is.** `torch.linalg.svd` on float32 CUDA tensors can produce inaccurate singular values for ill-conditioned matrices. This affects anything that uses `_torch_svd_cast` internally: stereo camera reprojection, fundamental/essential matrix estimation, etc.

**Fix (implemented in kornia).** `kornia.core.utils._torch_svd_cast` automatically promotes float32 inputs to float64 before SVD (except on MPS where float64 is unsupported), then casts the result back. This matches the existing behaviour of `_torch_solve_cast`.

If you write a new function that calls SVD, use `_torch_svd_cast` rather than calling `torch.linalg.svd` directly.

---

### 7. Half-Precision dtypes (float16 / bfloat16)

**What it is.** float16 and bfloat16 have limited support across PyTorch and kornia:

- **bfloat16**: Many kornia functions explicitly reject it.  In addition, many CUDA kernels lack bfloat16 implementations (`svd_cuda`, `linalg_eigh_cuda`, `cdist_cuda`, `lu_factor_cublas`, `geqrf_cuda`, etc.).
- **float16**: PyTorch's `linalg` routines (`linalg.inv`, `linalg.eigh`, `linalg.svd`, …) do not accept float16 on CPU (`RuntimeError: Low precision dtypes not supported`).  On CUDA, many kernels trigger device-side asserts for float16 inputs.

**Testing strategy: isolated runs.** Half-precision tests live alongside their float32/float64 counterparts in the same directories and files.  They are **not** run in combined (`--dtype=all`) invocations on CUDA; instead, half-precision and standard-precision suites are run as separate, isolated pytest invocations:

```bash
# Standard CI — all devices, float32 and float64 only
pixi run test tests/ --dtype=float32,float64

# Half-precision — run separately, per directory or file
pytest tests/color/           --dtype=float16,bfloat16
pytest tests/geometry/        --dtype=float16,bfloat16 --device=cuda
```

Keeping the runs separate means a half-precision failure or CUDA context corruption cannot affect the float32/float64 results.

**CUDA device-side asserts and test contamination.** CUDA kernel errors are *asynchronous*: a failing kernel logs the error but continues execution until the next host–device synchronisation point.  If that sync happens inside a *different* (passing) test, that test fails spuriously.  Once a device-side assert fires, the CUDA context is permanently broken for the process lifetime.

The root `conftest.py` contains two autouse fixtures to handle this:

- **`skip_half_precision_on_cuda`** — *skips* float16 and bfloat16 tests on CUDA when tests are run in combined mode.  Skipping means no CUDA kernel is launched, so no assert can be triggered.  On CPU/MPS/TPU, tests run as normal (they may fail).

- **`cuda_device_assert_guard`** — synchronises the CUDA device *before* each CUDA test.  If the context is already corrupted by a previous test, the current test is *skipped* rather than allowed to fail spuriously.  After each CUDA test, a second synchronisation drains the queue so that any async error surfaces in teardown of the test that caused it, not at the start of the next one.

**Running half-precision tests across a whole directory.**  Use `--isolate-half-precision`.  Each float16/bfloat16 CUDA test is run in a completely fresh Python process via `subprocess.run`, so a device-side assert in one test cannot affect any other test — there is no shared CUDA state at all:

```bash
# Whole directory, fully isolated — results reported normally (pass/fail per test)
pytest tests/color/     --device=cuda --dtype=bfloat16 --isolate-half-precision
pytest tests/geometry/  --device=cuda --dtype=all      --isolate-half-precision

# Via pixi tasks
pixi run test-half        # float16 + bfloat16, CPU
pixi run test-cuda-half   # float16 + bfloat16, CUDA, with isolation
```

Without `--isolate-half-precision`, float16/bfloat16 CUDA tests are **skipped** (safe default for combined runs).

**See also.** `docs/source/get-started/precision.rst` for the per-module half-precision support table.

---

## Writing Robust Tests

- **Seed the RNG** when the test compares against reference values: `torch.manual_seed(seed)`.
- **Generate random inputs on CPU** then move to device, to avoid device-specific RNG sequences.
- **Use realistic inputs**: positive depths, in-bounds pixel coordinates, valid rotation matrices, and sensible intrinsic matrices.
- **Avoid hardcoding CUDA expected values** computed from CPU runs — they will differ.
- **Use `nondet_tol`** in `gradcheck` for ops that use CUDA atomics.
- **Check the error magnitude** before relaxing tolerances: only relax `atol`/`rtol` if the maximum observed error is well below 0.01, and document why.
