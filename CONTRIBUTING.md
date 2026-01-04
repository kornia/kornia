# Contributing to Kornia

Welcome! This guide will help you contribute to Kornia.

## Policies and Guidelines

- **AI Policy & Authorship**: See [AI_POLICY.md](AI_POLICY.md) for the complete policy. Summary:
    - Kornia accepts AI-assisted code but strictly rejects AI-generated contributions where the submitter acts as a proxy.
    - **Proof of Verification**: PRs must include local test logs proving execution.
    - **Hallucination & Redundancy Ban**: Use existing `kornia` utilities and never reinvent the wheel, except for when the utility is not available.
    - **The "Explain It" Standard**: You must be able to explain any code you submit.
    - Violations result in immediate closure or rejection.

- **15-Day Rule**: PRs with no activity for 15+ days will be automatically closed.

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
   - Follow the [development setup](#developing-kornia) below.
   - See [Pull Request](#pull-request) section for PR requirements.

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
  - **Use `torch.Tensor` directly** for type hints (preferred) or import from `kornia.core` for backward compatibility
  - Use `torch.nn.Module` directly for module classes (preferred) or import from `kornia.core` for backward compatibility
  - For non-JIT modules, use `from __future__ import annotations`
  - **Always** type function inputs and outputs:
  - Run type checking with `pixi run typecheck` (uses `ty`)
    ```python
    from __future__ import annotations
    import torch

    def homography_warp(
      patch_src: torch.Tensor,
      dst_homo_src: torch.Tensor,
      dsize: tuple[int, int],
      mode: str = 'bilinear',
      padding_mode: str = 'zeros'
    ) -> torch.Tensor:
    ```

    For module classes:
    ```python
    from __future__ import annotations
    import torch.nn as nn

    class MyModule(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x
    ```

- **Code style:**
  - Follow [PEP8](https://www.python.org/dev/peps/pep-0008/)
  - Use f-strings: [PEP 498](https://peps.python.org/pep-0498/)
  - Line length: 120 characters
  - Comments must be written in English and verified by a human with a good understanding of the code
  - Obvious or redundant comments are not allowed (see [Best Practices](#best-practices) for comment guidelines)
  - W504 (line break after binary operator) is sometimes acceptable. Example:

    ```python
    determinant = A[:, :, 0:1, 0:1] * A[:, :, 1:2, 1:2] -
                  A[:, :, 0:1, 1:2] * A[:, :, 1:2, 0:1])
    ```

- **Third-party libraries:** Not allowed. Only PyTorch.

# Best Practices

This section provides guidance for contributing to Kornia, with a focus on Python and PyTorch best practices, performance, and maintainability.

## Before You Start

1. **Discuss First**: Always discuss your proposed changes in Discord or via a GitHub issue before starting implementation. This ensures your work aligns with project goals and avoids duplicate effort.

2. **Start Small**: If you're new to the project, start with small bug fixes or documentation improvements to familiarize yourself with the codebase and contribution process.

3. **Understand the Codebase**: Take time to explore existing code patterns, architecture, and conventions before implementing new features.

4. **Review Existing Utilities**: Before implementing new functionality, search the codebase for existing utilities in `kornia`. This aligns with the AI Policy's Hallucination & Redundancy Ban (see [Policies and Guidelines](#policies-and-guidelines)).

## Development Workflow

1. **Keep PRs Focused**: Each PR should address a single concern. If you're working on multiple features, create separate PRs for each.

2. **Test Locally First**: Always run all relevant tests locally before submitting (see [Pull Request](#pull-request) for requirements):
   ```bash
   pixi run lint        # Check formatting and linting
   pixi run test         # Run all tests
   pixi run typecheck    # Verify type checking
   ```

3. **Update Documentation**: When adding new features or changing behavior, update docstrings for public APIs. For documentation contributions, see [Contributing to Documentation](#contributing-to-documentation).

## Code Quality

1. **Performance Considerations**:
   - Prefer in-place operations when possible (e.g., `tensor.add_(other)` vs `tensor = tensor.add(other)`)
   - Use tensor views and slicing instead of copying when possible
   - Leverage PyTorch's vectorized operations over Python loops
   - Profile before optimizing (use `torch.profiler` or `cProfile`)
   - Consider memory efficiency for large tensors (use appropriate dtypes, avoid unnecessary copies)
   - Use `torch.jit.script` or `torch.compile` for performance-critical paths when appropriate

2. **Code Clarity**:
   - Use descriptive variable and function names that convey intent
   - Keep functions focused and single-purpose
   - Prefer clear code over comments; when comments are needed, explain "why" not "what"
   - Avoid over-engineering; start simple and refactor when needed

3. **Tensor Operations**:
   - Use `kornia` utilities instead of reimplementing common operations (see [AI Policy](#policies-and-guidelines))
   - Ensure operations are device-agnostic (work on CPU, CUDA, MPS, etc.)
   - Support multiple dtypes (float32, float64, float16, bfloat16) when applicable
   - Handle batched and non-batched inputs consistently

## Testing Best Practices

- Write tests for happy paths, error cases, edge conditions, boundary conditions, and integration scenarios
- Use `BaseTester` from `testing.base` for consistent test structure (see [Coding Standards](#coding-standards) for examples)
- Test across different devices and dtypes using pytest parametrization (see [Coding Standards](#coding-standards) for examples)
- Make tests deterministic, fast, and independent
- Use descriptive test names; test both forward pass and gradients when applicable

## Review Process

- Review your own PR first: check for typos/formatting, verify tests pass, ensure documentation is updated, and confirm AI policy compliance
- Respond promptly to review feedback
- Be open to feedback and explain your decisions when questioned
- See [Pull Request](#pull-request) section for review requirements

## AI-Assisted Development

- Understand every line of code you submit; you must be able to explain it during review (see [AI Policy](#policies-and-guidelines))
- Review AI output thoroughly: check for unnecessary complexity, verify it follows project conventions, ensure it uses existing utilities, and test it
- Be transparent in PR descriptions about what was AI-assisted and what you manually reviewed (see [Pull Request](#pull-request) for AI Usage Disclosure requirements)

## Communication

- Write clear, concise PR descriptions (see [Pull Request](#pull-request) for requirements)
- Always link to related issues or discussions in your PR description
- Ask questions in Discord or PR comments if unsure; it's better to clarify early than to rework later

# Pull Request

## Issue Approval and Assignment Workflow

**Before submitting a PR, you must:**

1. **Open an issue first**: All PRs must be linked to an existing issue. If no issue exists for your work, create one using the appropriate template (bug report or feature request).

2. **Wait for maintainer approval**: A maintainer must review and approve the issue before you start working on it. New issues are automatically labeled with `triage` and will receive a welcome message explaining this process.

3. **Wait for assignment**: You must be assigned to the issue by a maintainer before submitting a PR. This ensures:
   - The issue aligns with project goals
   - No duplicate work is being done
   - Proper coordination of contributions

4. **Do not start work until assigned**: PRs submitted without prior issue approval and assignment may be closed or receive warnings during automated validation.

This workflow helps maintain quality, avoid conflicts, and ensure contributions align with the project's direction. The automated PR validation workflow will check these requirements and post warnings if they're not met.

**Requirements:**
- **Issue approval and assignment**: The linked issue must be approved by a maintainer and you must be assigned to it (see workflow above)
- Link PR to an issue (use "Closes #123" or "Fixes #123")
- Pass all local tests before submission
- Provide proof of local test execution in the PR description (this is especially important for first-time contributors)
- Fill the [pull request template](.github/pull_request_template.md)
- **AI Policy Compliance**: Must comply with [AI_POLICY.md](AI_POLICY.md). This includes:
  - Using existing `kornia` utilities instead of reinventing
  - Being able to explain all submitted code
  - Completing the AI Usage Disclosure in the PR template
- 15-Day Rule: Inactive PRs (>15 days) will be closed
- Transparency: Keep discussions public

**Code review:**
- By default, GitHub Copilot will check the PR against the AI Policy and the coding standards.
- Code must be reviewed by the repository owner or a senior contributor, who have the final say on the quality and acceptance of the PR.

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
