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

Use Pixi for project tasks and environment selection. The tasks call `uv`, and Python tests run in the repository `.venv`; do not assume packages in `.pixi` are the versions under test.

```bash
pixi install
pixi install -e py312
pixi install -e py313
pixi run -e cuda install
pixi run test-module tests/path/to/test_file.py
pixi run test-quick
pixi run test-slow
pixi run lint
pixi run typecheck
pixi run doctest
pixi run build-docs
```

Select devices and dtypes with pytest options or environment variables:

```bash
pixi run test-module tests/geometry --device=cpu --dtype=float32,float64
KORNIA_TEST_DEVICE=cuda KORNIA_TEST_DTYPE=float32 pixi run test-module tests/geometry
```

Supported fixtures include CPU, CUDA, MPS, and TPU when available, and `float16`, `bfloat16`, `float32`, and `float64`. Run focused checks first and expand them in proportion to the change.

Use `KORNIA_TEST_RUNSLOW=true` to include slow tests and `KORNIA_TEST_OPTIMIZER=<backend>` to select the `torch.compile` optimizer. Run `pixi run lint` before presenting a code change as finished, together with the focused tests and other checks that match the files you changed.

### Precision and device details

- For CPU half-precision coverage, use
  `KORNIA_TEST_DTYPE=float16,bfloat16 pixi run test-module tests/<path>`.
- CUDA `float16`/`bfloat16` tests need per-test subprocess isolation; use `pixi run -e cuda test-cuda-half` or pytest's `--isolate-half-precision` option.
- MPS does not support float64 gradcheck. MPS autocast can also change the effective dtype; inspect nearby tests before changing tolerances or skips.
- TF32 is disabled by default for reproducibility. Enable it only when intentionally testing that mode.
- Preserve device and dtype rather than creating implicit CPU or default-dtype tensors. Use the injected `device` and `dtype` fixtures in tests.

## Library preferences

- Search for an existing `kornia` operation before building the same operation from raw PyTorch.
- Follow nearby batching, broadcasting, shape-validation, dtype, and device conventions.
- Use `BaseTester` from `testing.base`, injected `device`/`dtype` fixtures, and `self.assert_close()` for tensor comparisons. Its dtype-specific tolerances are intentional.
- Include smoke, exception, shape/cardinality, numerical, gradient, and `torch.compile` tests when they apply; not every change needs every kind.
- Keep numerical correctness tests self-contained. For expected values produced by an optional reference library, prefer a hardcoded literal plus the small generation snippet and source. Optional-dependency integration tests, including ONNX tests, may still use `pytest.importorskip`.
- For a new algorithm, name the source that defines it, such as a paper or a reference implementation in PyTorch, OpenCV, or scikit-image.
- Public APIs need type hints, docstrings, and exports.
- The codebase keeps a 120-character line length and Apache 2.0 source headers. Ruff and `ty` enforce the current style and types.
- JIT-compatible modules have stricter typing constraints. Follow nearby annotations and use `torch.Tensor` directly where TorchScript expects it.
- Avoid new runtime dependencies. If a dependency is genuinely useful, explain why the existing stack is insufficient and consider its install, build, and platform cost.

## Documentation and generated examples

- Add every new public class or function to the corresponding `docs/source/*.rst` page so it appears in the rendered API reference.
- When adding a feature detector or descriptor, update the `responses` list in `docs/generate_examples.py` and add the matching branch that renders its heatmap or score visualization. Follow the existing `DISK`, `ALIKED`, and `XFeat` examples and preserve the expected `(B, 3, H, W)` image shapes.

## Benchmarks

Use `benchmarks/README.md` and the helpers in `benchmarks/common.py` as the current source of truth. Benchmark public kornia APIs rather than embedding a replacement implementation in the benchmark itself.

When adding or changing a benchmark:

- cover CPU and CUDA where the operation supports both, and label unsupported or unavailable regimes rather than hiding them;
- include quality metrics when performance alone does not describe whether the result is useful;
- record the date, hardware, git commit, Python, PyTorch, kornia, and device details needed to interpret the result;
- compare the same benchmark on the relevant base revision and on the changed branch for performance work;
- use the benchmark suite's `--contribute` flow when producing committed result JSON.

## ONNX work

ONNX support is a real subsystem, not a generic export afterthought. Inspect `kornia/onnx/`, its mixins in `kornia/core/`, and existing ONNX tests before changing export behavior. Development uses the optional `onnx`, `onnxruntime`, and `onnxscript` packages. Test both graph construction/export and runtime behavior when the change affects both.

## Helping users diagnose problems

Start with a minimal reproduction and the narrowest relevant test. Do not claim a failure is pre-existing until you have compared against the relevant base revision safely.

Ask only for environment details that help the diagnosis: kornia, PyTorch, and Python versions; OS family; device backend; CUDA availability/version; and GPU model when relevant. Prefer the same privacy-conscious command used by the bug template:

```bash
python -c "import kornia, platform, sys, torch; gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'; print(f'kornia {kornia.__version__} | torch {torch.__version__} | python {sys.version.split()[0]} | OS {platform.system()} {platform.release()} | CUDA {torch.version.cuda} (available={torch.cuda.is_available()}, GPU={gpu}) | MPS={torch.backends.mps.is_available()}')"
```

Before asking a user to share broader diagnostic output, tell them to preview and redact it. Do not request usernames, home or project paths, environment variables, tokens, internal hostnames, or names of private data and models.
