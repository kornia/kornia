# Contributing to kornia

## Our social contract

We (maintainers) develop kornia for ourselves and the computer vision community.
If kornia is useful to you in any way — to import, to prototype, to use as a reference for your agent — that makes us happy.
You are welcome to help us improve kornia: report a bug, implement a feature, propose an idea, answer a question, or simply tell others about the project.
We try to do our best to accept your help — by reviewing PRs, fixing bugs, and so on — but we do not promise to review every PR, let alone quickly.
We have jobs, families, health problems, and holidays, just like everyone else.

You may use AI to do the work — so do we. There are no approved or disapproved amounts of AI use. Tell us how AI helped; we do the same.
Above all, we ask you to care. Good work done with care makes the world better; sloppy work makes the world worse.

The rest of this guide contains practical and technical guidance so nobody has to guess. If you have experience with open source, you probably already know most of it.

## Practical guidelines

- Search existing issues and pull requests before starting. Focused fixes and documentation improvements are welcome as pull requests. For a new public API, a behavior change, or a large piece of work, an early issue or discussion can save everyone time. A conversation helps us align; it is not a promise that somebody will review or merge the result.
- You do not need to ask before working on an issue that is not reserved. Say on the issue that you are taking it, then open a pull request, as a draft if you still have questions. If a claim has no pull request after two weeks, anyone may take the issue; say so on the issue when you do.
- Some issues say they are reserved for a maintainer implementation, and some already have an open pull request. Please comment on the issue or on that pull request instead of opening another one.
- Open an issue with a reproducer before fixing a numerical edge case that realistic inputs do not reach, such as half-precision overflow at magnitudes far beyond image or coordinate ranges, or a matrix scaled by `1e15`. Each such fix costs a full review across dtypes and devices and adds a promotion path to maintain, so we decide first whether it is worth that cost. One issue can list several cases. Wrong results on realistic inputs are welcome directly as pull requests.
- Check the part you changed and tell us what you ran. A bug fix is easier to trust when it includes a test that fails without the fix. Numerical work is easier to review when expected values come from a reference implementation, a paper, or a mathematical invariant.
- Look for an existing `kornia` utility before adding another implementation. Keep changes focused, follow nearby code, and update the documentation when public behavior changes.
- Read your own contribution and stay part of the conversation about it. You do not need perfect English or an immediate answer; we care that questions and technical concerns are treated seriously.
- If you are contributing as part of an application, remember that we read the work and review conversation, not pull request counts.
- Keep project decisions and review discussions in public GitHub issues, discussions, or pull requests so others can find and learn from them.
- Pull requests are marked stale after 15 quiet days to keep the queue usable. **We do not close pull requests for inactivity** — the label is a triage signal and nothing else, and it clears itself as soon as you comment or push. Maintainers can add `pinned` when a pull request should be exempt from the sweep entirely. A quiet pull request is often waiting on us, so take the label as a nudge to ping us, not as a deadline.

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
   - See the [Pull Request](#pull-request) section for practical guidance.

4. **Donate resources:**
   - [Open Collective](https://opencollective.com/kornia)
   - [GitHub Sponsors](https://github.com/sponsors/kornia)
   - We're looking for CUDA server donations for testing.

## Developing kornia

### Setup

1. **Fork** the [repository](https://github.com/kornia/kornia/fork)

2. **Clone your fork** and add upstream:
    ```bash
    $ git clone git@github.com:<your GitHub username>/kornia.git
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
    # Create the default Pixi environment, then install development dependencies
    pixi install
    pixi run -e default install

    # For specific Python versions
    pixi install -e py312  # Python 3.12
    pixi run -e py312 install
    pixi install -e py313  # Python 3.13
    pixi run -e py313 install

    # For CUDA development (requires reinstall of PyTorch)
    pixi run -e cuda install
    ```

    The `py312` and `py313` environments use isolated `.venv-py312` and `.venv-py313` project environments automatically.

    **Available tasks:**

    kornia provides several tasks via pixi for common development workflows:

    ```bash
    # Installation
    pixi run -e default install # Install dev dependencies
    pixi run -e default install-docs # Install dev + docs dependencies

    # Testing
    pixi run test             # Run tests (configure via KORNIA_TEST_* env vars)
    pixi run test-f32         # Run tests with float32
    pixi run test-f64         # Run tests with float64
    pixi run test-slow        # Run slow tests
    pixi run test-quick       # Run quick tests (same as `pixi run test`)
    pixi run test-module tests/<path>  # Run a focused test path
    pixi run test-half        # Run CPU float16/bfloat16 tests

    # CUDA testing (requires cuda environment)
    pixi run -e cuda test-cuda      # Run tests on CUDA
    pixi run -e cuda test-cuda-f32  # Run CUDA tests with float32
    pixi run -e cuda test-cuda-f64  # Run CUDA tests with float64
    pixi run -e cuda test-cuda-half # Run isolated CUDA float16/bfloat16 tests

    # Code quality
    pixi run lint             # Run ruff linting
    pixi run pre-commit-all   # Run all repository pre-commit checks
    pixi run pre-commit-install  # Install the pre-commit Git hook
    pixi run typecheck        # Run type checking with ty
    pixi run doctest          # Run doctests

    # Documentation
    pixi run -e default build-docs # Build documentation

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

    **Dependencies:** Defined in `pyproject.toml`. Update it and run `pixi run -e default install`.

    **CUDA:** Run `pixi run -e cuda install` to install the CUDA 12.1 PyTorch build into the project `.venv`,
    which is also used by `pixi run -e cuda test-cuda`. That index stops at torch 2.5.1, the oldest version
    Kornia supports, so CUDA runs test the floor rather than the locked PyTorch. Pixi's `.pixi/envs/cuda`
    supplies the tools, not the Python dependencies under test. Commands run with `-e cuda` disable automatic uv
    synchronization so the CUDA build is preserved. Re-run the CUDA install task after changing dependencies or
    using a default-environment command that synchronizes `.venv`. For focused CUDA tests, use
    `pixi run -e cuda test-module tests/<path> --device=cuda`.

5. **Develop and test:**

    Create test cases for your code. Run tests with:
    ```bash
    # Run all tests
    pixi run test

    # Run specific test file
    pixi run test-module tests/<TEST_TO_RUN>.py

    # For specific test with pytest options
    pixi run test-module tests/<TEST_TO_RUN>.py --dtype=float32,float64 --device=all
    ```

    **dtype options:** `bfloat16`, `float16`, `float32`, `float64`, `all`
    **device options:** `cpu`, `cuda`, `tpu`, `mps`, `all`

    We use [pre-commit](https://pre-commit.com) for code quality. Install its Git hook with `pixi run pre-commit-install`. See [coding standards](#coding-standards) below.

## Contributing to Documentation

1. Set up your development environment (see [above](#developing-kornia))
2. Edit files in `docs/`
3. Build docs: `pixi run -e default build-docs`
4. Preview: `open docs/build/html/index.html`
5. Submit a PR following the [Pull Request](#pull-request) guidelines

### Docstring style

A docstring is how a reader, human or agent, decides which function to call, so it states the choice before the details. `kornia/geometry/depth.py` is the worked example. Docstrings are reST: write literals with double backticks, as in ``` ``(B, 1, H, W)`` ```, and keep single backticks for roles such as `` :func:`name` ``.

- **Title line**: one physical line, in task words, imperative, naming what sets the function apart from its siblings (2D or 3D, channels first or last, the criterion, one call or module form). Aim for about 80 characters; 120 is the hard limit. Avoid markup that breaks reST, such as a bare `*`. No two unrelated public symbols may share one (`tests/test_docstring_first_lines.py`); a class and the function it wraps may.
- **Status words** come first when they apply: `Deprecated: use X.` or `Alias of X.`. Do not label a function "Legacy"; for a nearby better choice, write a neutral "for X, use Y" in the contract paragraph.
- **Contract paragraph** (2-5 sentences) starts from the call: ``` ``points = depth_to_3d(depth, camera_matrix)`` ``` takes ... and returns .... A callable module shows construction and invocation; a data class (`Boxes`, `PinholeCamera`) shows construction and its main method. Name inputs and output with shapes, frames and units, and end with "use X instead" when a nearby task is better served elsewhere.
- **Existing blocks** stay: `.. note::`, `Convention:`, `Args:`, `Return:`/`Returns:`, `Raises:`, trimmed where the paragraph already says the same. In a long class docstring, put whatever decides correct use in the first paragraph.
- **`See Also:`** goes after `Returns`/`Raises` and before `Example`.
  - Write each entry as `` - :func:`name`: how it differs. ``; the explanation is required.
  - Spell targets as Sphinx registers them: a short name for an object on the same docs page, the documented path (`~kornia.geometry.camera.perspective.unproject_points`) for the rest. A shorter import alias renders as plain text, with no warning.
  - No self-links or private names. Link true siblings both ways (v1 and v2, 2D and 3D); reviewers check that, no test does.
- **Module docstrings** open with a title line, then the conventions the module's functions share, then "Which function to use", grouped by task, one line each, naming every entry of the `__all__` of the namespace users import from. A namespace without `__all__` (`kornia.geometry`, `kornia.geometry.transform`) catalogs its subpackages and main entry points instead; a defining file without `__all__` gets no catalog.
- **Shapes** are literals such as ``` ``(B, 1, H, W)`` ```. Keep `:math:` for formulas.

What `tests/test_docstring_style.py` checks:

- every package of `kornia` is assigned to in scope or excluded (global);
- each public function and class of a converted module has a `See Also:` whose entries carry an explanation, with no self-link or private target; an exception needs a reason;
- every role in a converted object's docstring resolves to a name Sphinx registers, built from `docs/source/*.rst`; roles in a catalog module docstring must import;
- no source file of a shape-checked module writes a shape tuple as `:math:`, comments included;
- a catalog namespace's module docstring names every export of its `__all__`.

Read the Docs builds with `fail_on_warning: true` (`readthedocs.yml`), so a Sphinx warning fails the docs check. A module joins these checks when it is added to the lists in `testing/docstring_rollout.py`: `DOCSTRING_MODULES` (defining-module prefixes: `See Also:` and references), `CATALOG_NAMESPACES` (namespaces whose `__all__` the module docstring must name) and `SHAPE_CHECK_MODULES` (source-file prefixes, or `None` for all of `kornia/`). An object that needs no `See Also:` goes into `SEE_ALSO_EXCEPTIONS` with a reason. The package inventory is `IN_SCOPE_PACKAGES` and `EXCLUDED_PACKAGES`. The helpers are in `testing/docstring_style.py`.

## Coding Standards

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
  - Write comments in English so they are useful to the whole community
  - Prefer comments that explain why; remove comments that merely repeat the code or no longer match it
  - W504 (line break after binary operator) is sometimes acceptable. Example:

    ```python
    determinant = A[:, :, 0:1, 0:1] * A[:, :, 1:2, 1:2] -
                  A[:, :, 0:1, 1:2] * A[:, :, 1:2, 0:1])
    ```

- **Dependencies:** Avoid adding runtime dependencies. If one is useful, explain why existing dependencies are not enough and discuss the trade-off first.

## Best Practices

This section provides guidance for contributing to kornia, with a focus on Python and PyTorch best practices, performance, and maintainability.

### Before You Start

1. **Start Small**: If you're new to the project, a small bug fix or documentation improvement is a good way to learn the codebase and contribution process.

2. **Understand the Codebase**: Take time to explore existing code patterns, architecture, and conventions before implementing new features.

3. **Review Existing Utilities**: Before implementing new functionality, search the codebase for existing utilities in `kornia`.

### Development Workflow

1. **Keep PRs Focused**: Each PR should address a single concern. If you're working on multiple features, create separate PRs for each.

2. **Test Locally**: Run the checks that are relevant to your change:
   ```bash
   pixi run lint          # Run the Ruff hooks
   pixi run pre-commit-all  # Run the full repository checks
   pixi run test          # Run all tests
   pixi run typecheck     # Verify type checking
   ```

3. **Update Documentation**: When adding new features or changing behavior, update docstrings for public APIs. For documentation contributions, see [Contributing to Documentation](#contributing-to-documentation).

### Changelog entries

For a user-visible change, add a short Markdown file such as `changelog.d/1234.fixed.md`,
where `1234` is the PR number (not an issue number). Use `added`, `fixed`, or `breaking` for the type.
Write what changes for the user; breaking changes must describe the old and new behavior.
Only release PRs edit `CHANGELOG.md`: separate files prevent unrelated PRs
from conflicting on the same lines.

If the PR number is not known yet, use a unique name such as `+fix-camera-shape.fixed.md`
and rename it to the PR number before merging. Internal refactors, tests, and CI-only changes can omit
an entry; explain why in the PR description and ask a maintainer to apply `no-changelog`.
CI still validates fragment names and rendering when this label is present.

Run `pixi run changelog-preview` to preview the notes without changing any files.
See [the fragment guide](changelog.d/README.md) for examples and release instructions.

### Code Quality

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
   - Use `kornia` utilities instead of reimplementing common operations
   - Ensure operations are device-agnostic (work on CPU, CUDA, MPS, etc.)
   - Support multiple dtypes (float32, float64, float16, bfloat16) when applicable
   - Handle batched and non-batched inputs consistently

### Testing Best Practices

- Write tests for happy paths, error cases, edge conditions, boundary conditions, and integration scenarios
- Use `BaseTester` from `testing.base` for consistent test structure (see [Coding Standards](#coding-standards) for examples)
- Test across different devices and dtypes using pytest parametrization (see [Coding Standards](#coding-standards) for examples)
- Make tests deterministic, fast, and independent
- Use descriptive test names; test both forward pass and gradients when applicable

### Review Process

- Review your own PR first: check for typos and formatting, run relevant tests, and update affected documentation
- Respond to review feedback when you can
- Be open to feedback and explain your decisions when questioned
- Remember that review is done by people with limited time and may not happen quickly

### Working with AI

- AI tools are welcome. Read their output, check it in the same way you would check your own work, and state in the pull request which part was done by a human, if any, and which by AI.
- Useful details include whether you diagnosed the problem, designed the approach, edited the diff, or checked the result yourself, or whether an agent did the whole task from a prompt. We do not need a percentage.

### Communication

- Write a clear, concise pull request description
- Link related issues or discussions when they exist
- Ask questions in Discord, GitHub Discussions, or pull request comments when useful

## Pull Request

Fill in the [pull request template](.github/pull_request_template.md) with enough context for another person to understand the change. Link related issues or discussions when they exist, say what you checked, and state which part was done by a human, if any, and which by AI. The description goes in the pull request itself; do not commit it to the repository as a file such as `PR.md`.

For bug fixes, a regression test that fails before the fix is especially helpful. For numerical or geometry changes, say where the expected behavior comes from. For new public APIs or behavior changes, an issue or discussion before a large implementation can uncover compatibility and design concerns early.

Reviewers may ask for changes, tests, documentation, or a smaller scope. They may also decide that a contribution does not fit the project. We will try to explain why, but review is not guaranteed. Pull requests that have been quiet for a while may be closed and can be reopened later.

**PR checks and merge requirements:**
- Code changes run the CPU test matrix, dynamo/compile tests, and type checking with `ty`
- Pre-commit checks formatting, linting, license headers, spelling, and repository file hygiene
- Documentation changes run the documentation build
- Every non-draft pull request gets an automated Copilot review when opened or marked ready and needs one maintainer approval
- Pull requests are squash-merged

If a check fails because of your change, please fix it or explain what you found.

## License

By contributing, you agree to license your contributions under the Apache License. See [LICENSE](./LICENSE).
