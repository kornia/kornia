# Contributing to Kornia

Welcome! This guide will help you contribute to Kornia.

## Policies and Guidelines

- **15-Day Rule**: PRs with no activity for 15+ days will be automatically closed.
- **Quality Control**: AI-generated PRs without human oversight will be flagged. If a PR doesn't improve after review iterations, it will be closed.
- **Transparency**: All discussions must be public.

We're all volunteers. These policies help us focus on high-impact work.

## Ways to Contribute

1. **Ask/Answer questions:**
   - [GitHub Discussions](https://github.com/kornia/kornia/discussions)
   - `#kornia` tag in [PyTorch Discuss](https://discuss.pytorch.org)
   - [Discord](https://discord.gg/HfnywwpBnD)
   - Don't use GitHub issues for Q&A.

2. **Report bugs** via [GitHub issues](https://github.com/kornia/kornia/issues):
   - Search for existing issues first.
   - Use the bug report template.
   - Include: clear description, reproduction steps, package versions, and code sample.

3. **Fix bugs or add features:**
   - Check [help wanted issues](https://github.com/kornia/kornia/issues?q=is%3Aissue%20state%3Aopen%20label%3A%22help%20wanted%22) for starting points.
   - **PRs must be linked to an issue** (use "Closes #123" or "Fixes #123").
   - Follow the [development setup](#developing-kornia) below.
   - Run local tests before submitting.

4. **Donate resources:**
   - [Open Collective](https://opencollective.com/kornia)
   - [GitHub Sponsors](https://github.com/sponsors/kornia)
   - We're looking for CUDA server donations for testing.

# Developing Kornia

## Setup

1. **Fork** the [repository](https://github.com/kornia/kornia/fork)

2. **Clone your fork** and add upstream:
    ```bash
    $ git clone git@github.com:<your Github username>/kornia.git
    $ cd kornia
    $ git remote add upstream https://github.com/kornia/kornia.git
    ```

3. **Create a branch** (don't work on `main`):
   ```bash
   git checkout upstream/main -b feat/foo_feature
   # or
   git checkout upstream/main -b fix/bar_bug
   ```

4. **Development environment**

    We use [pixi](https://pixi.sh) for package and environment management.

    **Install Pixi:**

    ```bash
    # On Linux/macOS
    curl -fsSL https://pixi.sh/install.sh | bash

    # On Windows (PowerShell)
    irm https://pixi.sh/install.ps1 | iex

    # Or using conda/mamba
    conda install -c conda-forge pixi
    ```

    **Set up the development environment:**

    ```bash
    # Install all dependencies (defaults to Python 3.11)
    pixi install

    # For specific Python versions
    pixi install -e py312  # Python 3.12
    pixi install -e py313  # Python 3.13

    # For CUDA development (requires reinstall of PyTorch)
    pixi run -e cuda install
    ```

    **Available tasks:**

    Kornia provides several tasks via pixi for common development workflows:

    ```bash
    # Installation
    pixi run install          # Install dev dependencies
    pixi run install-docs     # Install dev + docs dependencies

    # Testing
    pixi run test             # Run tests (configure via KORNIA_TEST_* env vars)
    pixi run test-f32         # Run tests with float32
    pixi run test-f64         # Run tests with float64
    pixi run test-slow        # Run slow tests
    pixi run test-quick       # Run quick tests (excludes jit, grad, nn)

    # CUDA testing (requires cuda environment)
    pixi run -e cuda test-cuda      # Run tests on CUDA
    pixi run -e cuda test-cuda-f32  # Run CUDA tests with float32
    pixi run -e cuda test-cuda-f64  # Run CUDA tests with float64

    # Code quality
    pixi run lint             # Run ruff linting
    pixi run typecheck        # Run type checking with ty
    pixi run doctest          # Run doctests

    # Documentation
    pixi run build-docs       # Build documentation

    # Utilities
    pixi run clean            # Clean Python cache files
    ```

    **Environment variables for tests:**

    Tests can be configured using environment variables:

    ```bash
    # Set device (cpu, cuda, mps, tpu)
    export KORNIA_TEST_DEVICE=cuda

    # Set dtype (float32, float64, float16, bfloat16)
    export KORNIA_TEST_DTYPE=float32

    # Run slow tests
    export KORNIA_TEST_RUNSLOW=true

    # Then run tests
    pixi run test
    ```

    **Dependencies:** Defined in `pyproject.toml`. Update it and run `pixi install`.

    **CUDA:** The CUDA environment uses PyTorch with CUDA 12.1. Run `pixi run -e cuda install` to set it up.

5. **Develop and test:**

    **Requirements:**
    - No AI-generated code without human oversight
    - **All local tests must pass before submitting PRs**

    Create test cases for your code. Run tests with:
    ```bash
    # Run all tests
    pixi run test

    # Run specific test file
    pixi run test tests/<TEST_TO_RUN>.py

    # For specific test with pytest options
    pixi run test tests/<TEST_TO_RUN>.py --dtype=float32,float64 --device=all
    ```

    **dtype options:** `bfloat16`, `float16`, `float32`, `float64`, `all`
    **device options:** `cpu`, `cuda`, `tpu`, `mps`, `all`

    We use [pre-commit](https://pre-commit.com) for code quality. Install it with `pre-commit install`. See [coding standards](#coding-standards) below.

# Contributing to Documentation

1. Set up your development environment (see [above](#developing-kornia))
2. Edit files in `docs/`
3. Build docs: `make build-docs`
4. Preview: `open docs/build/html/index.html`
5. Submit a PR following the [Pull Request](#pull-request) guidelines

# Coding Standards

- Use meaningful names for variables, functions, and classes.

- **Write small incremental changes:**
  - Commit small, logical changes
  - Write clear commit messages
  - Avoid large files

- **Add tests:**
  - Write unit tests for each functionality
  - Use helpers from [testing/](./testing/)
  - Put test utilities (not tests or fixtures) in `testing/`

    ```python
    from testing.base import BaseTester

    class TestMyFunction(BaseTester):
        # To compare the actual and expected tensors use `self.assert_close(...)`


        def test_smoke(self, device, dtype):
            # test the function with different parameters arguments, to check if the function at least runs with all the
            # arguments allowed.
            pass

        def test_exception(self, device, dtype):
            # tests the exceptions which can occur on your function

            # example of how to properly test your exceptions
            # with pytest.raises(<raised Error>) as errinfo:
            #     your_function(<set of parameters that raise the error>)
            # assert '<msg of error>' in str(errinfo)

            pass

        def test_cardinality(self, device, dtype):
            # test if with different parameters the shape of the output is the expected
            pass

        def test_feature_foo(self, device, dtype):
            # test basic functionality
            pass

        def test_feature_bar(self, device, dtype):
            # test another functionality
            pass

        def test_gradcheck(self, device):
            # test the functionality gradients
            # Uses `self.gradcheck(...)`
            pass

        def test_dynamo(self, device, dtype, torch_optimizer):
            #  test the functionality using dynamo optimizer

            # Example of how to properly test your function for dynamo
            # inputs = (...)
            # op = your_function
            # op_optimized = torch_optimizer(op)
            # self.assert_close(op(inputs), op_optimized(inputs))

            pass
    ```

- **Test coverage:** Cover different devices, dtypes, and batch sizes. Use `--dtype` and `--device` pytest arguments to generate test combinations:

    ```python
    import pytest

    @pytest.mark.parametrize("batch_size", [1, 2, 5])
    def test_smoke(batch_size, device, dtype):
        x = torch.rand(batch_size, 2, 3, device=device, dtype=dtype)
        assert x.shape == (batch_size, 2, 3)
    ```

- **Type hints** (Python >= 3.11):
  - Use typing when it improves readability
  - Import from `kornia.core`, e.g. `from kornia.core import Tensor`
  - For non-JIT modules, use `from __future__ import annotations`
  - **Always** type function inputs and outputs:
  - Run type checking with `pixi run typecheck` (uses `ty`)
    ```python
    from __future__ import annotations
    from kornia.core import Tensor

    def homography_warp(
      patch_src: Tensor,
      dst_homo_src: Tensor,
      dsize: tuple[int, int],
      mode: str = 'bilinear',
      padding_mode: str = 'zeros'
    ) -> Tensor:
    ```

- **Code style:**
  - Follow [PEP8](https://www.python.org/dev/peps/pep-0008/)
  - Use f-strings: [PEP 498](https://peps.python.org/pep-0498/)
  - Line length: 120 characters
  - W504 (line break after binary operator) is sometimes acceptable. Example:

    ```python
    determinant = A[:, :, 0:1, 0:1] * A[:, :, 1:2, 1:2] -
                  A[:, :, 0:1, 1:2] * A[:, :, 1:2, 0:1])
    ```

- **Third-party libraries:** Not allowed. Only PyTorch.

# Pull Request

**Requirements:**
- Link PR to an issue (use "Closes #123" or "Fixes #123")
- Pass all local tests before submission
- 15-Day Rule: Inactive PRs (>15 days) will be closed
- Quality: AI-generated PRs without oversight will be flagged/closed
- Transparency: Keep discussions public

**Note:** Tickets may be closed during cleanup. Feel free to reopen if you plan to finish the work.

**CI checks:**
- All tests pass
- Test coverage maintained
- Type checking (ty)
- Documentation builds successfully
- Code formatting (ruff, docformatter via pre-commit)

Fix any failing checks before your PR will be considered.

# License

By contributing, you agree to license your contributions under the Apache License. See [LICENSE](./LICENSE).
