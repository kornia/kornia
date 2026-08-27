# Agent guide for kornia

This file gives coding agents practical context for working in this repository.

## Project shape

kornia is a differentiable computer vision library built on PyTorch. Most code lives in subpackages under `kornia/`; reusable test support lives in `testing/`, not under `tests/`.

Important areas include:

- `kornia/geometry/`: transforms, cameras, epipolar geometry, stereo, and 3D
- `kornia/filters/`: filtering and edge operators
- `kornia/augmentation/`: differentiable augmentation pipelines
- `kornia/feature/` and `kornia/models/`: feature and model implementations
- `kornia/onnx/`: ONNX loading, execution, composition, and export
- `kornia/core/`: shared tensor, module, and ONNX mixins

Import order in `kornia/__init__.py` matters: `filters` and `geometry` are loaded before modules that depend on them. Check for circular imports when changing package exports.

## Environment and commands

Use Pixi for project tasks and environment selection. The tasks call `uv`, and Python tests run in the repository `.venv` or a versioned `.venv-py*`; do not assume packages in `.pixi` are the versions under test.

```bash
pixi install
pixi run install
pixi install -e py312
pixi run -e py312 install
pixi install -e py313
pixi run -e py313 install
pixi run -e cuda install
pixi run test-module tests/path/to/test_file.py
KORNIA_TEST_OPTIMIZER=inductor pixi run test-module tests/path/to/test_file.py
pixi run test-quick
pixi run test-slow
pixi run lint
pixi run pre-commit-all
pixi run typecheck
pixi run doctest
pixi run build-docs
```

The `py312` and `py313` features select isolated `.venv-py312` and `.venv-py313` project environments automatically for install and test tasks.

Select devices and dtypes with pytest options or environment variables:

```bash
pixi run test-module tests/geometry --device=cpu --dtype=float32,float64
KORNIA_TEST_DEVICE=cuda KORNIA_TEST_DTYPE=float32 pixi run test-module tests/geometry
```

Supported fixtures include CPU, CUDA, MPS, and TPU when available, and `float16`, `bfloat16`, `float32`, and `float64`. Run focused checks first and expand them in proportion to the change.

Use `KORNIA_TEST_RUNSLOW=true` to include slow tests. `KORNIA_TEST_OPTIMIZER=inductor` enables the dynamo and compile tests, which are deselected when the variable is unset. Before presenting a code change as finished, run the full pre-commit command above together with focused tests and other relevant checks; `pixi run lint` runs only the Ruff hooks.

### Precision and device details

- For focused CPU half-precision coverage, add `--dtype=float16,bfloat16` to a `test-module` run. `pixi run test-half` runs the whole CPU test suite.
- CUDA `float16`/`bfloat16` tests need per-test subprocess isolation; use `pixi run -e cuda test-cuda-half` or pytest's `--isolate-half-precision` option.
- MPS does not support float64 gradcheck. MPS autocast can also change the effective dtype; inspect nearby tests before changing tolerances or skips.
- TF32 matmul is disabled by default; `--tf32` enables it. cuDNN convolutions still use PyTorch's TF32 default.
- Preserve device and dtype rather than creating implicit CPU or default-dtype tensors. Use the injected `device` and `dtype` fixtures in tests.

## Library preferences

- Search for an existing `kornia` operation before building the same operation from raw PyTorch.
- Follow nearby batching, broadcasting, shape-validation, dtype, and device conventions.
- Use `BaseTester` from `testing.base`, injected `device`/`dtype` fixtures, and `self.assert_close()` for tensor comparisons. Its dtype-specific tolerances are intentional.
- Follow [TESTING.md](TESTING.md) for test mechanisms. Use `self.gradcheck()` for gradient checks and the injected `torch_optimizer` fixture for `torch.compile` coverage. Include smoke, exception, shape/cardinality, numerical, gradient, and compile tests when they apply; not every change needs every kind.
- Keep numerical correctness tests self-contained. For expected values produced by an optional reference library, prefer a hardcoded literal plus the small generation snippet and source. Optional-dependency integration tests, including ONNX tests, may still use `pytest.importorskip`.
- For a new algorithm, name the source that defines it, such as a paper or a reference implementation in PyTorch, OpenCV, or scikit-image.
- Public APIs need type hints, docstrings, and exports.
- The codebase keeps a 120-character line length and Apache 2.0 source headers. Ruff and `ty` enforce the current style and types.
- JIT-compatible modules have stricter typing constraints. Follow nearby annotations and use `torch.Tensor` directly where TorchScript expects it.
- For non-JIT modules, use `from __future__ import annotations`.
- Avoid new runtime dependencies. If a dependency is genuinely useful, explain why the existing stack is insufficient and consider its install, build, and platform cost.

## Skills

Repo-local skills live in `.claude/skills/<name>/SKILL.md`; agents that do not read that directory should open those files directly, at the paths below. `kornia-review-loop` is a rigid workflow with a red-flags table; the other two are rule lists.

- `.claude/skills/kornia-developer/SKILL.md` — making an op `torch.compile(fullgraph=True)`-compatible and faster.
- `.claude/skills/kornia-precision-testing/SKILL.md` — any test touching float16/bfloat16, device, `torch.jit.trace`, `torch.compile`, or degenerate sizes. Uses `testing.precision`.
- `.claude/skills/kornia-review-loop/SKILL.md` — responding to a review finding that states a cause, a numeric condition or threshold, or a list of files/sites/consumers (refactor and style findings take the short path it names): triage against the merge-base with the ref the PR actually targets (`gh pr view --json baseRefName`, not an assumed `origin/main`), fix defects only, delta self-review, `pixi run verify-delta`, then push once.

`pixi run verify-delta` diffs failing-test *sets* between the branch and `--base` (default `origin/main`; pass the PR's own base branch when it targets something else) on cpu float32, cpu half/float64, MPS, and inductor. Half precision and MPS have no CI job; this is their only signal, so run it before every push of a review-response commit. Pass `--tests tests/<module>` and `--only "<surface>"` after `--` to bound the run. It refuses a dirty checkout by default, since its automatic scope is `base...HEAD`; intentional working-tree runs require `--allow-dirty` together with explicit `--tests` covering every affected target. Exit 1 means new failures; exit 2 means a selected, available surface was never measured — read the table rather than the exit code alone, since an `unverified` row is a hole in the gate while `not selected` and `unavailable` rows are surfaces nobody asked for or the machine cannot run.

## Documentation and generated examples

- Add every new public class or function to the corresponding `docs/source/*.rst` page so it appears in the rendered API reference.
- Some modules document a known defect inline and link its tracking issue; `grep -rn "github.com/kornia/kornia/issues/" kornia/` finds them. That prose is part of the fix's blast radius: **before merging a change that closes one of those issues, run `grep -rnE "#NNNN|issues/NNNN" kornia/ tests/`.** Match the issue number rather than a name or a phrase — the wording varies across `Tracked in #NNNN`, a lowercase `tracked in` clause, a `Note:` block, a runtime error message, an `xfail` `reason=`, a plain comment and a test-name suffix (`test_wart_*_<issue>` and `test_convention_*_<issue>`, where that convention was followed), and a rendered link label can disagree with the URL beneath it. The `#`/`issues/` anchors keep the pattern off float literals. A surviving hit in `kornia/` means the change is incomplete. Hits in `tests/` are the pins that record the documented behavior; each has to be re-checked in the same change — a now-XPASSing strict `xfail` dropped, a wart assertion inverted — but a pin goes on naming its issue afterwards, so a hit there is not by itself a defect. Not every documented issue has a pin, so an empty `tests/` result is an answer rather than a failed search.
- When adding a feature detector or descriptor, update the `responses` list in `docs/generate_examples.py` and add the matching branch that renders its heatmap or score visualization. Follow the existing `DISK`, `ALIKED`, and `XFeat` examples and preserve the expected `(B, 3, H, W)` image shapes.

## Benchmarks

Follow the methodology contract and new-benchmark checklist in [benchmarks/README.md](benchmarks/README.md), using the helpers in `benchmarks/common.py`. Benchmark public kornia APIs rather than embedding a replacement implementation. For performance changes, run the same benchmark on the relevant base revision and changed branch.

## ONNX work

ONNX support is a real subsystem, not a generic export afterthought. Inspect `kornia/onnx/`, its mixins in `kornia/core/`, and existing ONNX tests before changing export behavior. Development uses the optional `onnx`, `onnxruntime`, and `onnxscript` packages. Test both graph construction/export and runtime behavior when the change affects both.

## Helping users diagnose problems

Start with a minimal reproduction and the narrowest relevant test. Do not claim a failure is pre-existing until you have compared against the relevant base revision safely.

Ask only for environment details that help the diagnosis: kornia, PyTorch, and Python versions; OS family; device backend; CUDA availability/version; and GPU model when relevant. Prefer the same privacy-conscious command used by the bug template:

```bash
python -W ignore -c "import kornia, platform, sys, torch; gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'; backend=torch.version.cuda or getattr(torch.version, 'hip', None); print(f'kornia {kornia.__version__} | torch {torch.__version__} | python {sys.version.split()[0]} | OS {platform.platform()} | CUDA/ROCm {backend} (available={torch.cuda.is_available()}, GPU={gpu}) | MPS={torch.backends.mps.is_available()}')"
```

Before asking a user to share broader diagnostic output, tell them to preview and redact it. Do not request usernames, home or project paths, environment variables, tokens, internal hostnames, or names of private data and models.
