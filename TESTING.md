# Testing Guide

This document explains how to run the Kornia test suite, what the common pytest flags are, and how to handle recurring classes of test failures.

## Environment and validation tasks

Use [the contributor setup](CONTRIBUTING.md#developing-kornia) and Pixi for project
commands. Tasks call `uv`; Python packages under test live in `.venv` or `.venv-py*`,
not `.pixi`. The `py312` and `py313` features select `.venv-py312` and `.venv-py313`.
Dependencies are defined in `pyproject.toml`; after an update run the matching install task.

**CUDA:** Run `pixi run -e cuda install` to install the locked PyTorch as a CUDA build into the project `.venv`,
which is also used by `pixi run -e cuda test-cuda`. PyPI's Linux wheels are already CUDA builds, so on Linux the
task only syncs the lock, reinstalling every package so that removing a stale wheel cannot take files a locked
wheel needs. PyPI's Windows wheels are CPU-only, so on Windows the task then replaces torch with the same version
from the PyTorch CUDA index; a lock update to a new PyTorch has to update that version and index in `pixi.toml`
and the index in `pyproject.toml` too. Pixi's `.pixi/envs/cuda` supplies the tools, not the Python dependencies
under test. Commands run with `-e cuda` do not synchronize `.venv` automatically, so re-run the CUDA install task
after a dependency or lock change. On Windows this is what preserves the CUDA build, and a default-environment
command that synchronizes `.venv` replaces it, so re-run the task after one of those too. For focused CUDA tests,
use `pixi run -e cuda test-module tests/<path> --device=cuda`.

The locked build needs an NVIDIA driver from the R580 branch or newer and a Turing or newer GPU. Check
`torch.cuda.is_available()` after installing: when it is false, a `--device=cuda` run skips its tests and still
exits 0.

`pixi run pre-commit-all` runs every hook; `pixi run lint` runs only Ruff. Use
`pixi run typecheck` for types and `pixi run toml-fmt` for TOML formatting. Install the
Git hook with `pixi run pre-commit-install`. Task definitions live in [pixi.toml](pixi.toml).

`pixi run doctest` skips examples needing an uncached checkpoint. Use
`pixi run doctest-weights` (or `KORNIA_DOCTEST_DOWNLOAD=1`) to exercise them with downloads
(about 838 MB on a cold cache). PR CI uses the restored cache and skips misses; scheduled
runs on `main` at 00:23 and 12:23 UTC download and exercise those examples.

## Running Tests

`test-module` forwards pytest arguments: a file, node ID, or `-k` expression narrows the
run. Start focused and expand for the affected callers. `KORNIA_TEST_OPTIMIZER=inductor`
enables dynamo/compile tests; the conformance options below are separate.

```bash
# All tests, default device (cpu) and dtype (float32)
pixi run test

# Specific file
pixi run test-module tests/geometry/test_boxes.py

# Specific device and dtype
pixi run test-module tests/ --device=cuda --dtype=float32

# All devices and dtypes
pixi run test --device=all --dtype=all

# Skip slow tests (default); include them with --runslow
pixi run test --runslow

# Quick tests (same as `pixi run test`; slow tests are skipped by default)
pixi run test-quick
```

### CLI Options

| Option | Env var | Default | Description |
|---|---|---|---|
| `--device` | `KORNIA_TEST_DEVICE` | `cpu` | `cpu`, `cuda`, `mps`, `tpu`, or `all` |
| `--dtype` | `KORNIA_TEST_DTYPE` | `float32` | `float32`, `float64`, `float16`, `bfloat16`, or `all` |
| `--runslow` | `KORNIA_TEST_RUNSLOW` | off | Include `@pytest.mark.slow` tests |
| `--tf32` | `KORNIA_TEST_TF32` | off | Enable TF32 mode (see below) |
| `--optimizer` | `KORNIA_TEST_OPTIMIZER` | CLI: `inductor`; env: unset | Select the backend. Only setting the environment variable enables dynamo/compile collection; the CLI option alone does not. |
| `--isolate-half-precision` | `KORNIA_TEST_ISOLATE_HALF` | off | Run float16/bfloat16 CUDA tests each in a fresh `subprocess.run` process (no shared CUDA state) |
| `--known-failure-profile=NAME` | none | unset | Select an explicit reproducible profile such as `cpu-float16` |
| `--xfail-known-failures` | none | off | Focus mode: validate selected entries from the chosen profile; without a profile, use the device/dtype known-failure manifest |
| `--verify-known-failures` | none | off | Complete mode: require the full chosen profile and every manifest entry to reproduce exactly |
| `--record-known-failures=PATH` | none | unset | Record a complete profile to a guarded, untracked candidate path |
| `--known-failure-manifest=PATH` | none | unset | Replay a candidate manifest instead of the checked-in profile manifest |
| `--run-device-agnostic` | `KORNIA_TEST_RUN_DEVICE_AGNOSTIC` | off | Run `@pytest.mark.device_agnostic` tests even when CPU is not in the device matrix |
| `--onnx-conformance=PACKAGES` | `KORNIA_TEST_ONNX_CONFORMANCE` | unset | Run the ONNX conformance test for these stable-core packages (`geometry,color`, ...) or `all`; deselected otherwise (see below) |
| `--compile-conformance=PACKAGES` | `KORNIA_TEST_COMPILE_CONFORMANCE` | unset | Run the torch.compile conformance test for these stable-core packages or `all`; deselected otherwise (see below) |

## Test Structure

Use `testing.base.BaseTester` for operator tests. Include smoke, exception, shape,
numerical, gradient, and compile tests where applicable; not every change needs every kind:

```python
from testing.base import BaseTester


class TestMyFunction(BaseTester):
    def test_smoke(self, device, dtype): ...  # basic run, all arg combinations
    def test_exception(self, device, dtype): ...  # error paths
    def test_cardinality(self, device, dtype): ...  # output shapes
    def test_feature(self, device, dtype): ...  # correctness / numerical accuracy
    def test_gradcheck(self, device): ...  # gradient checking via self.gradcheck()
    def test_dynamo(self, device, dtype, torch_optimizer): ...  # torch.compile compat
```

The `device` and `dtype` fixtures are injected automatically from the CLI options.  Use `self.assert_close()` for tensor comparisons — it automatically selects tolerances appropriate for the dtype.

Keep numerical expected values self-contained: prefer a hardcoded literal with the
small generation snippet and source to requiring an optional reference library.
Optional-dependency integration tests, including ONNX tests, may use `pytest.importorskip`.
Preserve device/dtype with the injected fixtures. Keep tests deterministic, independent,
and proportional to the change; test batched behavior and derivatives where applicable.
Reusable helpers belong in `testing/`, not under `tests/`.

## Comparing a branch against another revision

kornia is installed in editable mode, so `import kornia` resolves through the editable finder to
the **primary checkout** regardless of the current directory. A measurement script that lives
outside a worktree therefore imports the primary tree even when you run it from the worktree:
`python /tmp/probe.py` puts `/tmp` first on `sys.path`, nothing shadows the editable install, and
an A/B comparison silently measures the same revision twice and reports "no difference".

Run comparison snippets from the worktree root, as stdin or `-c`, never as a file stored
elsewhere. Only then does the leading empty `sys.path` entry -- the working directory -- shadow
the editable install:

```bash
# From the primary checkout; select .venv-py* here if that is the environment under test.
comparison_python="$PWD/.venv/bin/python"
git worktree add /tmp/base-wt <base-revision>
cd /tmp/base-wt && "$comparison_python" - <<'EOF'
import sys
import kornia
print(sys.executable)
print(kornia.__file__)  # must name the worktree, not the primary checkout
EOF
```

On Windows, use the environment's `Scripts/python.exe` instead of `bin/python`.
Print `kornia.__file__` inside every measurement block and check that it names the tree you
meant. A `cd` into the worktree is not by itself enough, and neither is putting the worktree on
`PYTHONPATH` while the script still lives somewhere else. The same applies to `sys.executable`
when more than one virtualenv is in play: name the interpreter explicitly rather than relying on
`PATH`.

## Gradient guards across PyTorch versions

`torch.clamp` is not a portable gradient guard: its derivative **at the bound** passes the incoming gradient
through (`1.0`) on torch 2.5.1 and 2.9.1 and returns `0.0` on 2.14.0. Wherever the bound is also the singular
point -- `clamp(min=0).sqrt()`, `clamp(-1, 1).acos()`, `clamp(-1, 1).asin()` -- the clamp bounds the value only,
so the unbounded derivative still reaches the backward pass as `inf` or `nan` on the older half of the supported
range while 2.14 masks it. A floor below the dtype's smallest subnormal is no guard either: `clamp(min=1e-12)`
underflows to `0` in `float16`. Substitute a safe argument before evaluating the singular expression;
take any value-only correction from a detached copy. `torch.where` does not protect an
unselected arm's backward pass: both differentiated arms must have finite derivatives.
For example, `safe = torch.where(x > 0, x, torch.ones_like(x))` followed by
`torch.where(x > 0, safe.sqrt(), torch.zeros_like(x))` chooses a zero derivative at zero
without evaluating `sqrt(0)` in autograd; that convention still needs a contract justification.
A CPU float32 run on a recent torch does not exercise the version difference -- the
2.5.1 leg is what discriminates. The measurements and the audit of the pattern are in #4229.

## Repository contract checks

### Public API inventory

`tests/api_surface.json` inventories the public names of stable-core modules; `tests/test_api_surface.py` rejects their removal until the inventory is updated in the same change. Run `tests.test_api_surface.regenerate()` to regenerate it. Keep tracked module keys (use `[]` when removing every name). Every module that declares `__all__` under a recorded package root has its own key; deleting such a module empties its key, and that empty key stays as the record. The `Import Surface` workflow also checks that each inventory removal corresponds to an actual export removal in the same diff. An inventory edit acknowledges an `__all__` removal only for the exact same module and name. For an `__all__` name of a module without its own key, such as a module outside the recorded package roots, add the exact module and name to `tests/api_surface_removals.json`, a JSON object mapping module names to lists of removed names (for example, `{"kornia.feature.dedode": ["DeDoDe"]}`). Only entries newly added relative to the merge base count, and each must match an actual `__all__` removal in that change; an unrelated source edit or a previously recorded acknowledgement does not count. Explicit acknowledgements do not replace updates to existing inventory entries, the deprecation window, or release notes.

### Dynamic-shape check messages

A `KORNIA_CHECK` message that formats a tensor size or an `int` argument specializes it under a dynamic-shape `torch.compile`, which then compiles a graph per value. Compare sizes with `KORNIA_CHECK_SAME_SIZE`, `KORNIA_CHECK_SIZES` or `KORNIA_CHECK_SHAPE`, or pass the message as `lambda: f"..."`. TorchScript rejects a `lambda`, so a function it compiles formats whole shapes instead, such as `{x.shape}`, which do not specialize. `tests/core/test_check_message_call_sites.py` enforces both and lists the functions TorchScript compiles.

### Docstring checks

The authoring rules live in [Docstring style](CONTRIBUTING.md#docstring-style).

What `tests/test_docstring_style.py` checks:

- every package of `kornia` is assigned to in scope or excluded (global);
- an in-scope package has no function or class, exported by an `__all__` or documented on a docs page, without a docstring;
- each public function and class of an in-scope package has a `See Also:` whose entries carry an explanation and name a kornia object, with no self-link or private target; an exception needs a non-empty reason, and an ``Alias of :func:`target`.`` or ``Deprecated: use :func:`target`.`` title line replaces the `See Also:`;
- every Python-domain object role (`:func:`, `:class:`, `:meth:`, `:attr:`, `:data:`, `:const:`, `:exc:`, `:obj:`) in the docstring of such a function or class, and in those of the methods, properties and attributes autodoc documents under such a class, resolves as Sphinx resolves it. It must name either an object registered by the `.rst` pages under `docs/source`, or a torch, Python or NumPy object in the cached intersphinx inventories (`docs/source/_intersphinx`). Registered objects include the data attributes autodoc documents: enum members and dataclass fields under `:undoc-members:`, and variables with a `#:` comment or a docstring. An intersphinx object must be of a type the role links to: a function for `:func:`, a method for `:meth:`, an attribute or property for `:attr:`, and nothing for `:const:`. Only `:func:` and `:meth:` drop a trailing `()`. Roles in a catalog module docstring must import;
- no `kornia/**/*.py` file or `docs/source` page writes a shape tuple as `:math:`, comments included (global); a tuple that reads as a shape but is a value is listed with what it is in `MATH_NON_SHAPES` (`testing/docstring_rollout.py`), and an entry that no longer occurs fails;
- a catalog namespace's module docstring names every export of its `__all__` inside a role or a double-backtick literal, where a qualified name must be a path to that export (`torch.nn.functional.normalize` does not catalog `kornia.enhance.normalize`); the same word in plain prose or in single backticks does not count.

Read the Docs builds with `fail_on_warning: true` (`readthedocs.yml`) and the docs dependencies pinned in `uv.lock`, the same versions as `pixi run build-docs`, so a Sphinx warning fails the docs check. The checks cover every public object of the packages in `IN_SCOPE_PACKAGES` (`testing/docstring_rollout.py`); `EXCLUDED_PACKAGES` names the rest with a reason, and a new top-level package must join one of the two. A namespace with `__all__` is a catalog namespace when some export is not also exported, as the same object, by a catalog namespace above it, or when its module docstring has a catalog anyway (a line starting `Which function to use`, `Which class to use` or the like); `catalog_namespaces` in `testing/docstring_style.py` selects them, and `CATALOG_EXCEPTIONS` leaves one out with a reason. An object that needs no `See Also:` goes into `SEE_ALSO_EXCEPTIONS` with a reason. The helpers are in `testing/docstring_style.py`; its `sphinx_registry` raises on an autodoc option it does not model, so a docs page that adds one (`:no-index:`, `:exclude-members:`, `:inherited-members:`) needs the registry taught first.

## Markers

| Marker | Meaning |
|---|---|
| `@pytest.mark.device_agnostic` | Test runs once in a CPU-containing matrix because its logic does not use the selected accelerator. An accelerator-only run deselects it and says so on a line of its own at the start of the run, because the deselected set includes the ONNX and `torch.export` suites; `--run-device-agnostic` forces it back on |
| `@pytest.mark.slow` | Long-running test; skipped unless `--runslow` is passed |
| `@pytest.mark.tf32` | Known to fail under TF32 (see section below); xfail unless `--tf32` |
| `@pytest.mark.onnx_conformance` | Part of the ONNX conformance test; deselected unless `--onnx-conformance` selects packages |
| `@pytest.mark.compile_conformance` | Part of the torch.compile conformance test; deselected unless `--compile-conformance` selects packages |

### Selecting conformance packages

From the repository root after fetching the target branch, select changed packages and
their importers, including committed, staged, unstaged, and untracked files. Use
`upstream/main` from the contributor setup, or `origin/main` when `origin` is Kornia itself:

```bash
(
  set -eu
  conformance_target=upstream/main
  # For a direct clone of Kornia, use this instead:
  # conformance_target=origin/main
  conformance_base=$(git merge-base "$conformance_target" HEAD)
  conformance_paths=$(
    git diff --no-renames --name-only "$conformance_base" &&
      git ls-files --others --exclude-standard
  )
  printf '%s\n' "$conformance_paths" | pixi run python .github/scripts/conformance_packages.py onnx
  printf '%s\n' "$conformance_paths" | pixi run python .github/scripts/conformance_packages.py compile
)
```

The selectors use Pixi's Python so they can parse the repository's Python syntax.
The subshell stops if the base cannot be resolved; check the selected remote and fetch it
before retrying. `--no-renames` includes both the old and new paths of a moved file, as CI does.
Each command prints the corresponding conformance option's package value (`all`, a
comma-separated list, or empty when no promised operator is affected). Recompute after edits.

### ONNX conformance (#5819)

Before changing export behavior, inspect `kornia/onnx/`, its mixins in `kornia/core/`,
and existing ONNX tests. Development uses optional `onnx`, `onnxruntime`, and
`onnxscript`; verify graph construction/export and runtime behavior when both change.
Use the [package selection](#selecting-conformance-packages) for changed packages and their importers.

`tests/onnx/test_conformance.py` holds kornia's ONNX promise: every *promised* case -- a record of the survey
snapshot `docs/source/_data/export_support.json` with ONNX status `ok` in a stable-core package
(`docs/source/get-started/stability.rst`), without anything built from pretrained weights (registry tag `pretrained`)
or listed in `EXCLUDED` -- must export with `torch.onnx.export(..., dynamo=True)` and match eager execution in
onnxruntime (the same dtype, values within the case's tolerance), and a legacy (TorchScript) export that succeeds must
match as well; one that fails loudly passes. The cases are the survey's own (`docs/export_support/cases_*.py`, joined
by name; `cases_models.py` is never imported) and run through `docs/export_support/harness.py`, so the test and the
survey page cannot drift apart: `test_promised_cases_are_registered` fails when a promised case is missing or skipped
in its registry, and `test_promised_set_is_pinned` when the promised names change or an exclusion leaves nothing out.

The promise itself -- the promised cases, `EXCLUDED` with the reason for each exclusion, `ENFORCED_TORCH_RELEASES` and
the reading of the manifests -- is `testing/conformance_promise.py`, which imports no pytest: the docs build renders
the support statement of the export-support page ("What Kornia promises") from it, and `testing/conformance.py` runs
it as tests.

```bash
pixi run test-module tests/onnx/test_conformance.py --onnx-conformance=all
pixi run test-module tests/onnx/test_conformance.py --onnx-conformance=geometry,color
```

The run takes minutes, so it is opt-in. Pull requests run it on torch 2.15.0 for the packages their changes reach
(`.github/scripts/conformance_packages.py onnx`: a changed module selects its stable-core package and every
stable-core package with a module that imports it, directly or not; `kornia/core`, `kornia/utils`, `kornia/onnx`, the
survey and its snapshot, the stability page, the test and its support, and the dependency pins select `all`); the
scheduled run on `main` checks `all` on torch 2.9.1 and 2.15.0. No promised case needs pretrained weights, and the test
never downloads.

Known failures are strict xfails in `testing/known_failure_xfails/onnx_conformance_torch<major>.<minor>.txt`, one
manifest per enforced torch release, in the `<exception>\t<node ID>` format of the other manifests, each entry under
a comment that opens with its tracking issue (`# #NNNN: why`, or `# owner/repo#NNNN ...` upstream; a comment naming no
issue after an entry or a blank line starts a block without one); loading a manifest rejects an entry without one, and
the support statement lists each entry with that issue. The exception type names the
kind of failure (`STATUS_EXCEPTIONS` in `testing/conformance.py`): `AssertionError` for a result that differs from
eager, `RuntimeError` for a failed export, checker or onnxruntime run (or, in the compile test, a failed compile),
`ValueError` for a failed eager run, so a known failure that changes kind fails the run. Entries of packages a run does
not select are not checked. Delete an entry when its case passes (a strict XPASS fails the run); a new failure of a
promised operator is a regression to fix.

### torch.compile conformance (#5819)

Run this check for changed packages and their importers using the
[package selection](#selecting-conformance-packages).

`tests/test_inductor_conformance.py` holds the torch.compile promise the same way: every record of the survey snapshot
with compile status `ok` in a stable-core package, without anything built from pretrained weights or listed in
`EXCLUDED`, must compile with `torch.compile(..., fullgraph=True)` and the default inductor backend and match eager
execution (the same dtype, values within the case's tolerance). The survey records `ok` only for a case that
`fullgraph=True` compiles and that matches eager (#6194). `EXCLUDED` lists the snapshot's `ok` cases the promise leaves
out, each with its reason; an exclusion must name a case the promise would cover without it, so a snapshot
regeneration that drops one fails `test_promised_set_is_pinned` until its entry is deleted. Each case is called once,
at its registry input shapes, through the survey's own `docs/export_support/probe_compile.py`, with Dynamo reset
around it; a recompile at another shape is not part of the promise. The selection, the drift guard, the pinned
promised set and the manifests (`testing/known_failure_xfails/compile_conformance_torch<major>.<minor>.txt`) work as
for the ONNX test.

```bash
pixi run test-module tests/test_inductor_conformance.py --compile-conformance=all
pixi run test-module tests/test_inductor_conformance.py --compile-conformance=color
```

It has its own option and does not read `KORNIA_TEST_OPTIMIZER`, and its tests are named after inductor because the
root `conftest.py` drops tests whose name contains `compile` or `dynamo` when that variable is unset. Inductor needs a
C++ compiler; onnx and onnxruntime are not needed. Pull requests run it on torch 2.15.0 for the packages
`.github/scripts/conformance_packages.py compile` selects (the same mapping as the ONNX job, with this test file and
its manifests in place of the ONNX ones); the scheduled run on `main` checks `all` on torch 2.9.1 and 2.15.0. Both
conformance jobs run with `--timeout=600`.

---

## Known Sources of Test Failures

### 1. TF32 (TensorFloat-32) Precision

**What it is.** `torch.set_float32_matmul_precision("high")` enables TF32 mode for CUDA matrix multiplications (`torch.bmm`, `torch.mm`, etc.). TF32 truncates float32 inputs to a 10-bit mantissa before the multiply-accumulate, giving roughly float16 mantissa precision for those ops. This is the default when `torch.compile` is in use and is enabled by many deep-learning frameworks for throughput.

**Effect on tests.** Numerically sensitive tests that compare float32 outputs of matrix operations against hardcoded expected values (or against a CPU reference) can fail because the accumulated rounding error exceeds the test tolerance.

**In Kornia's test suite.** TF32 is **off by default** for both matmul and cuDNN convolutions. Pass `--tf32` (or set `KORNIA_TEST_TF32=true`) to enable it. Tests that are known to be sensitive to TF32 are marked `@pytest.mark.tf32`; without `--tf32` they are marked non-strict `xfail`, so a default run reports neither their failure nor their recovery.

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

**Use `self.gradcheck()`.** `BaseTester` supplies `nondet_tol=1e-10` for CUDA inputs
when the caller does not set it; CPU keeps zero. This covers measured rounding from
ordinary CUDA kernels without changing which kernels are tested. If an operation needs
more, measure the discrepancy and justify its tolerance in the test instead of applying
an arbitrary blanket relaxation. See `CUDA_GRADCHECK_NONDET_TOL` in `testing/base.py`.

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
pixi run test --dtype=float32,float64

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
pixi run -e cuda test-cuda-half   # float16 + bfloat16, CUDA, with isolation
```

Without `--isolate-half-precision`, float16/bfloat16 CUDA tests are **skipped** (safe default for combined runs).

**Linux CPU half-precision CI baseline.** Pull requests and scheduled CI run the complete `cpu-float16` and
`cpu-bfloat16` profiles on Linux, Python 3.11, and the workflow-pinned PyTorch release. Selecting a profile sets
the device and dtype before parametrization, then gives each node an isolated, stable Python/NumPy/PyTorch seed.
Manifest entries store an exact setup/call phase, opaque fully qualified exception identity, and node ID. The v1
header enforces the profile, Python major/minor, PyTorch release, OS, architecture, and seed scheme; pytest, NumPy,
and Pillow versions are recorded as provenance warnings.

Focus mode accepts file, node, and `-k` selections and validates only selected manifest entries:

```bash
KORNIA_TEST_OPTIMIZER= pixi run test-module tests/geometry/liegroup/test_so3.py \
  --xfail-known-failures --known-failure-profile=cpu-float16
```

Complete mode is the blocking CI contract. It requires the canonical full `tests/` collection with no partial
selector, optimizer, slow suite, xdist, early-stop, last-failed, failed-first, or collect-only option:

```bash
KORNIA_TEST_OPTIMIZER= pixi run test-module tests/ \
  --verify-known-failures --known-failure-profile=cpu-float16
```

A recorded entry must reproduce with the same phase and exception identity. A pass, skip, competing xfail, changed
exception, teardown failure, or stale node ID makes verification red. Delete a line after a fix, update its node ID
after a rename/reparametrization, and prefer fixing a new failure. The tracker prints an exact ready-to-paste line
when a deliberate addition is justified.

Normal pull requests do not regenerate manifests. For an initial profile, supported-environment update, or deliberate
broad re-baseline, record only to an untracked candidate outside the repository, then replay it in a fresh process:

```bash
KORNIA_TEST_OPTIMIZER= pixi run test-module tests/ \
  --record-known-failures=/tmp/kornia-cpu-float16-candidate.txt \
  --known-failure-profile=cpu-float16

KORNIA_TEST_OPTIMIZER= pixi run test-module tests/ \
  --verify-known-failures --known-failure-profile=cpu-float16 \
  --known-failure-manifest=/tmp/kornia-cpu-float16-candidate.txt
```

The recording run exits nonzero because failures remain un-xfailed, but writes a candidate only after complete
collection and lifecycle accounting. It refuses checked-in/tracked destinations (including symlink aliases), partial
runs, unrepresentable candidate failures, and audited nodes that consume eager fixture RNG. Review every candidate
delta and the replay result before manually replacing the checked-in manifest. The same commands with
`cpu-bfloat16` and a distinct `/tmp` path handle bfloat16.

`pixi run test-half` remains the ordinary unseeded CPU sweep; it is useful for finding tests that accidentally depend
on shared RNG state. A failure seen only there is a test-determinism bug to fix, never a reason to edit the seeded
profile manifest.

**See also.** `docs/source/get-started/precision.rst` for the per-module half-precision support table.

### 8. MPS (Apple Silicon) Limitations

**What MPS is.** The `mps` device uses Apple's Metal Performance Shaders backend. Run tests against it with `--device=mps`.

**Known unsupported operations.** Several operations are not implemented in the MPS backend and raise at runtime:

| Operation | Error | Torch versions |
|---|---|---|
| `float64` (double precision) | `TypeError: Cannot convert a MPS Tensor to float64 dtype` | all |
| `complex128` (cdouble) | `TypeError: Cannot convert a MPS Tensor to float64 dtype` (`NotImplementedError: … not implemented for 'ComplexDouble'` before 2.14) | all |
| `torch.autocast("mps")` | Converts output to float16 instead of preserving original dtype | all |
| `torch.linalg.eigvals` (non-symmetric) | `NotImplementedError: The operator 'aten::_linalg_eigvals' is not currently implemented for the MPS device` | all, 2.14.0 included |
| `torch.linalg.svd` / `svdvals` / `lstsq` whose input holds 8192 elements or more | `RuntimeError: Failed to created pipeline state object … XPC_ERROR_CONNECTION_INTERRUPTED` | 2.14.0 |
| `F.grid_sample` with `padding_mode="border"` on 2D | `RuntimeError: MPS: Unsupported Border padding mode` | <= 2.9.1 |
| `F.grid_sample` with `mode="nearest"` on 5-D (3D volumes) | `RuntimeError: grid_sampler_3d: Unsupported Nearest interpolation` | <= 2.9.1 |
| `grid_sample` backward (2D and 3D) | `NotImplementedError: The operator 'aten::grid_sampler_2d_backward' …` | <= 2.9.1 |
| `torch.linalg` `eigh`, `svd`, `qr`, `lu_solve`, `lstsq`, `cholesky_solve`, `matrix_exp` | `NotImplementedError: … is not currently implemented for the MPS device` | <= 2.9.1 |

The last four rows moved between the two releases kornia's CI pins: torch 2.14.0 added MPS
kernels for the whole eigen/QR/LU/SVD family except `_linalg_eigvals`, for both `grid_sample`
backward passes, and for the two `grid_sample` forward modes above. That took the recorded MPS
baseline from 200 entries to 14. The intermediate 2.10-2.13 releases were not checked, and
kornia still supports torch 2.5.1, so the workarounds for those rows stay.

The new native SVD path carries its own ceiling: `svd`, `svdvals` or `lstsq` fails to build a
Metal pipeline once its input holds 8192 elements or more (batch x rows x cols). The threshold
is inclusive and it is the first failing size, not the last working one -- measured
`(511, 4, 4)` = 8176 passes, `(512, 4, 4)` = 8192 raises. The boundary is a byte count, not
an element count: 8192 float32 elements are exactly 32768 bytes, the MPS threadgroup limit,
and torch's own "matrix too large to stage in MPS threadgroup memory ... falling back to CPU"
check triggers on *greater than* that limit, so an input sitting exactly on it neither falls
back nor compiles. Shape is irrelevant -- `(128, 64)`, `(64, 128)`, `(8192, 1)` and `(1, 8192)`
all raise, and 8191 and 8193 elements both pass -- which is why the guard counts elements and
does not inspect batch rank. `_torch_svd_cast` and `_torch_linalg_svdvals` route such inputs
through the CPU, so `RANSAC`'s batched fundamental and homography solvers do not hit the ceiling.
The RANSAC tests are skipped on MPS for a different reason -- see the next section: they abort the
process before they can raise.

**How Kornia handles these automatically.** The test infrastructure in `conftest.py` and `testing/base.py` skips known-unsupported test classes at collection time so you don't need per-test guards for the common cases:

- `test_gradcheck[mps*]` — skipped automatically (`gradcheck` requires float64)
- `*[mps*cdtype1*]` — skipped automatically (parametrized `torch.cdouble` tests)
- `test_autocast[mps*]` — skipped automatically (MPS autocast changes dtype)

The `padding_mode="border"` issue in `F.grid_sample` (2D) is worked around in the implementation (`kornia/feature/laf.py`) by clamping the sampling grid to the outermost pixel *centers*, at `±(1 - 1/size)`, and using `padding_mode="zeros"`, which is exactly equivalent. Clamping to `[-1, 1]` is not: with `align_corners=False` that is the outer edge of the border pixel, where bilinear sampling blends it with the zero padding.

**CI coverage.** Pull-request CI runs one blocking MPS leg (`tests-mps` in `.github/workflows/pr_test_cpu.yml`): the newest Python and PyTorch 2.14.0, `float32`, on the GitHub-hosted **`macos-15`** Apple-silicon image. PyTorch stays one release behind the CPU matrix: torch 2.15 uses only a GPU in `MTLGPUFamilyApple7`, and the paravirtual GPU of every GitHub-hosted arm64 image does not report that family, so on 2.15 the leg finds no MPS device ([#6390](https://github.com/kornia/kornia/issues/6390)). The image is pinned deliberately. `macos-latest` is `macos-26-arm64`, whose GPU is paravirtualized, and on macOS 26 torch 2.14 emits Metal 4 `mpp::tensor_ops::matmul2d` cooperative-tensor shaders that this virtual GPU cannot compile — one specialization failed 664 tests across every subpackage, since `linalg.inv`, `linalg.solve`, `inv_ex`, `lu_factor_ex` and `F.conv3d` all route through it. The defect is the *virtual* GPU, not macOS 26: a physical M1 on macOS 26.5 compiled those shaders in the #4202 comparison. macOS 15 is Metal 3 and takes the older kernel path, which is the path that happens to work on the runner fleet. Two consequences worth knowing: no job in this repository now exercises the Metal 4 path that users on current macOS actually run, and when GitHub retires the `macos-15` image this leg stops scheduling and `collector` goes red with no fallback. Re-test `macos-latest` once upstream guards those kernels by device capability; `macos-14` segfaults inside MPS kernels and must never be used. The comparison is recorded in [#4159](https://github.com/kornia/kornia/issues/4159#issuecomment-5532655834). The exact known-failure baseline lives in `testing/known_failure_xfails/mps_float32.txt`. `--xfail-known-failures` applies the manifest's entries as strict xfails with their exact exception types. A new failure, a different exception, a skipped recorded test, or a fixed test therefore makes CI red.

Run the identical contract locally with `pixi run test-mps`. `PYTORCH_ENABLE_MPS_FALLBACK=1` silently routes missing MPS kernels to the CPU and invalidates the baseline, so unset it first. The manifest is a full-suite contract and is not intended for `-k` or partial-directory runs; use `pixi run test-module ... --device=mps --dtype=float32` for focused diagnosis without the manifest. "Full suite" means the suite as an `mps`-only run collects it: tests marked `device_agnostic` are deselected there and covered by the CPU jobs, so they never appear in the manifest.

When an MPS failure is fixed, remove its line from `mps_float32.txt`. When a test is renamed or reparametrized, update its node ID. Treat an unlisted failure as a regression to fix; add it only when the new limitation is understood and has its own tracking issue, named in a comment beside the entry. After a deliberate Python, PyTorch, or runner-image update, regenerate the full manifest from that CI job's final `short test summary info`, recording one exact exception type and node ID per `FAILED` line.

**Writing new tests that work on MPS.** Follow these rules:

1. **Never create `float64` tensors on the device in non-gradcheck tests.** Use the `dtype` fixture, or if the algorithm requires double precision, skip explicitly:
   ```python
   if device.type == "mps":
       pytest.skip("MPS does not support float64")
   ```
2. **`self.gradcheck()` is safe.** `BaseTester.gradcheck()` automatically skips on MPS devices.
3. **Avoid `padding_mode="border"` in 2D `grid_sample` on MPS.** Use the clamped-zeros workaround or check `device.type == "mps"`.
4. **3D `grid_sample` with `mode="nearest"` is unsupported on MPS.** Do not silently fall back to bilinear (it would corrupt segmentation masks); instead raise or skip the test.
5. **Device comparison.** `tensor.device` on MPS returns `device(type='mps', index=0)`, not `device(type='mps')`. The test fixture provides `torch.device("mps:0")` so `tensor.device == device` comparisons work correctly.

---

## Writing Robust Tests

- **Seed the RNG** when the test compares against reference values: `torch.manual_seed(seed)`.
- **Generate random inputs on CPU** then move to device, to avoid device-specific RNG sequences.
- **Use realistic inputs**: positive depths, in-bounds pixel coordinates, valid rotation matrices, and sensible intrinsic matrices.
- **Use independent expected values** from the mathematical definition or an appropriate reference, often at higher precision; another run of the implementation is not an independent oracle.
- **Use `self.gradcheck()`**; if CUDA nondeterminism exceeds `BaseTester`'s default, measure it and justify any larger `nondet_tol`.
- **Justify tolerance overrides** using dtype, scale, conditioning, and the algorithm's accuracy requirements. Measure the discrepancy and keep the test sensitive to regressions.

## Diagnosing reported problems

Start with a minimal reproduction and the narrowest relevant test. Do not claim a failure is pre-existing until you have compared against the relevant base revision safely.

Ask only for environment details that help the diagnosis: kornia, PyTorch, and Python versions; OS family; device backend; CUDA availability/version; and GPU model when relevant. Prefer the same privacy-conscious command used by the bug template:

```bash
python -W ignore -c "import kornia, platform, sys, torch; gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'; backend=torch.version.cuda or getattr(torch.version, 'hip', None); print(f'kornia {kornia.__version__} | torch {torch.__version__} | python {sys.version.split()[0]} | OS {platform.platform()} | CUDA/ROCm {backend} (available={torch.cuda.is_available()}, GPU={gpu}) | MPS={torch.backends.mps.is_available()}')"
```

Before asking a user to share broader diagnostic output, tell them to preview and redact it. Do not request usernames, home or project paths, environment variables, tokens, internal hostnames, or names of private data and models.
