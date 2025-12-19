# Contributing to Kornia
**Welcome !!** This is the Kornia library contributor's corner. If you are reading this, it means that you have an interest
in **Differentiable Computer Vision**, and are willing to contribute to the project.

Everyone is welcome to get involved with the project. There are different ways to contribute with your two cents:

1. Ask/Answer questions:
    - Where can you ask questions?
      1. using the GitHub discussion at Kornia repo: [GH Discussions](https://github.com/kornia/kornia/discussions)
      2. using the `#kornia` tag in [PyTorch Discuss](https://discuss.pytorch.org)
      3. using Discord Link [Join Discord](https://discord.gg/HfnywwpBnD)
    - Please, don't use GitHub issues for Q&A.
    - In case you are a developer and want to learn more about the PyTorch ecosystem, we suggest you join the PyTorch
      slack. You can apply using this form: [https://bit.ly/ptslack](https://bit.ly/ptslack)

2. Report bugs through [GitHub issues](https://github.com/kornia/kornia/issues):
   - Do a quick search first to see whether others reported a similar issue.
   - In case you find an unreported bug, please open a new ticket.
   - Try to provide as much information as possible. Report using one of the available templates. Some tips:
     - Clear title and description of the issue.
     - Explain how to reproduce the error.
     - Report your package versions to facilitate the task.
     - Try to include a code sample/test that raises the error.

3. Fix a bug or develop a feature from the roadmap:
   - We will always have an open ticket showing the current roadmap.
   - Pick an unassigned feature (or potentially propose a new one) or an open bug ticket.
   - Follow the instructions from [Developing Kornia](#developing-kornia) to setup your development
     environment and start coding.
   - Check our coding conventions. See more details below.
   - Run the test framework locally and make sure all works as expected before sending a pull request.
   - Open a Pull Request, get the green light from the CI, and get your code merged.

4. Donate resources to the project:
   - In case you are an organization/institution that wants to give support, sponsor, or just use the project, please
     contact us.
     - [opencollective.com/kornia](https://opencollective.com/kornia)
     - [github.com/sponsors/kornia](https://github.com/sponsors/kornia)
   - We are open to starting any kind of collaboration and hearing feedback from you.
   - We pretend to provide features on demand. Reach us!
   - Currently looking for some kind of server donation to test *CUDA* code. (Please contact us).

# Developing Kornia

To start to develop, please follow the steps below:

1. Fork the [kornia repository](https://github.com/kornia/kornia) by clicking on the
[fork](https://github.com/kornia/kornia/fork) button on the repository page. This will create a copy of the Kornia
repository under your GitHub account.


2. Clone your fork of Kornia, and add the Kornia repository as a remote:
    ```bash
    $ git clone git@github.com:<your Github username>/kornia.git
    $ cd kornia
    $ git remote add upstream https://github.com/kornia/kornia.git
    ```

3. Create a new branch with a meaningful name reflecting your contribution. See an example:
    ```bash
    $ git checkout upstream/main -b feat/foo_feature
    # or
    $ git checkout upstream/main -b fix/bar_bug
    ```
    🚨 **Do not** work on the `main` branch!

4. Creating a development environment

    **Using kornia script (Recommended)**

    Kornia now uses [uv](https://github.com/astral-sh/uv) for fast Python package management and virtual environment creation.
    The `setup_dev_env.sh` script will automatically install uv (if not already installed), create a virtual environment,
    and install all development dependencies including PyTorch with the appropriate CUDA version.

    ```bash
    $ ./setup_dev_env.sh
    ```

    This script will:
    - Install uv if it's not already available
    - Create a virtual environment in the `./venv` directory
    - Install PyTorch with the appropriate CUDA version (default CUDA 12.1)
    - Install all development dependencies using uv dependency groups
    - Install documentation dependencies using uv dependency groups

    You can customize the Python version, PyTorch version, and CUDA version using environment variables:
    ```bash
    $ PYTHON_VERSION=3.10 PYTORCH_VERSION=2.4.0 CUDA_VERSION=11.8 ./setup_dev_env.sh
    ```

    To use CPU-only PyTorch:
    ```bash
    $ PYTORCH_MODE=cpuonly ./setup_dev_env.sh
    ```

    **Using justfile commands (Recommended)**

    Kornia provides a `justfile` with convenient commands for development tasks. The justfile automatically
    ensures the virtual environment is set up before running any commands.

    To see all available commands:
    ```bash
    $ just
    ```

    To run tests:
    ```bash
    $ just test-cpu        # Run CPU tests
    $ just test-cuda       # Run CUDA tests
    $ just test-all        # Run all tests
    ```

    To run linting and type checking:
    ```bash
    $ just lint            # Run code formatting and linting
    $ just mypy            # Run type checking
    ```

    **Manually setup with uv**

    If you prefer to set up the environment manually:

    1. Install uv:
    ```bash
    # On Linux/macOS
    $ curl -LsSf https://astral.sh/uv/install.sh | sh

    # Or using pip
    $ pip install uv
    ```

    2. Create and activate a virtual environment:
    ```bash
    $ uv venv
    $ source .venv/bin/activate  # On Windows: .venv\Scripts\activate
    ```

    3. Install PyTorch with appropriate CUDA version:
    ```bash
    # For CUDA 12.1 (default)
    $ uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

    # For CPU-only
    $ uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
    ```

    4. Install Kornia development dependencies:
    ```bash
    $ uv sync --group dev --group x
    $ uv sync --group docs  # For documentation development
    ```

    **Attention**: If *Kornia* was already installed in your virtual environment, remove it with
    `uv pip uninstall kornia` before reinstalling it in editable mode with the `-e` flag.

    **Dependency Management with uv.lock**

    Kornia uses a `uv.lock` file for reproducible dependency management. This ensures all developers get exactly
    the same versions of dependencies:

    - **Adding dependencies**: Add new dependencies to the appropriate section in `pyproject.toml`:
      - Main dependencies: `dependencies = [...]`
      - Dev dependencies: `dependency-groups.dev = [...]`
      - Docs dependencies: `dependency-groups.docs = [...]`
      - Extra dependencies: `dependency-groups.x = [...]`

    - **Updating the lock file**: After adding dependencies, regenerate the lock file:
      ```bash
      $ just lock-update
      # or: uv lock --upgrade
      ```

    - **Installing from lock file**: The setup script automatically uses the lock file when available:
      ```bash
      $ just sync  # Install dependencies from lock file
      # or: uv sync --frozen
      ```

    - **Updating specific packages**:
      ```bash
      $ uv lock --upgrade-package <package-name>
      ```

    Always commit both your `pyproject.toml` changes AND the updated `uv.lock` file together.

5. Develop the code on your branch, and before creating the pull request, make sure to ensure the code passes the checks.

    As you develop your code, you should also create test cases for your code. As well as, In addition to ensuring that
    the other tests continue to pass. You can run the tests with:
    ```bash
    $ pytest tests/<TEST_TO_RUN>.py --dtype=float32,float64 --device=all
    ```
    With the `dtype` argument, run the tests using tensors with all `dtypes` desired. Options: `bfloat16`, `float16`,
    `float32`, `float64`, and `all`.

    In the same way, the `device`, will run the tests using tensors on the `device` desired. Options: `cpu`, `cuda`,
    `tpu`, `mps`, and `all`.


    Kornia relies on [pre-commit](https://pre-commit.com) to run code quality tools. Make sure to have `pre-commit`
    under your dev environment, otherwise, you can install the tools manually and run them with the help of the available
    commands of the [Makefile](./Makefile). Read more about the code standards adopted [here](#coding-standards).

# Contributing to Documentation

We welcome contributions to the Kornia documentation! If you'd like to improve our docs, please follow these steps:

1. Set up your development environment as described in the [Developing Kornia](#developing-kornia) section above.

2. Make your changes to the documentation files located in the `docs/` directory.

3. Build the documentation using the provided Makefile:

   ```bash
   $ make build-docs
   ```

   This command will delete any previously built files and generate the newest version of the documentation.

4. The built documentation will be available in the `docs/build/html/` directory. You can open the main page in your browser by running:

   ```bash
   $ open docs/build/html/index.html
   ```

5. Review your changes in the browser to ensure they appear as expected.

6. Once you're satisfied with your changes, commit them and submit a pull request following the guidelines in the [Pull Request](#pull-request) section below.

## Benchmarking

We have a benchmark suite configured in [benchmarks/](./benchmarks/). We used the
 [pytest-benchmark](https://pypi.org/project/pytest-benchmark/) library to benchmark our function units.

Our [Makefile](./Makefile) has an `benchmark` command as an alias on how to run our benchmarks.

```console
# To run all suite
$ make benchmark

# To run a specific file you can pass `BENCHMARK_SOURCE`
$ make benchmark BENCHMARK_SOURCE=benchmarks/augmentation/2d_geometric_test.py

# To run a specific benchmark you use `BENCHMARK_SOURCE` as the pytest standard behaviour
$ make benchmark BENCHMARK_SOURCE=benchmarks/augmentation/2d_geometric_test.py::test_aug_2d_elastic_transform

# To update the optimizer backends desired to execute you can pass `BENCHMARK_BACKENDS=`
$ make benchmark BENCHMARK_BACKENDS='inductor,eager'

# To pass other options to the runner, you can use `BENCHMARK_OPTS`
# Example, setup to run the benchmark on cuda on verbose mode
$ make benchmark BENCHMARK_OPTS='--device=cuda -vv'
```

We use the same tests generator suite, so you can set up the device within `--device`, the dtype within
`--dtype`, and the optimizer backend within `--optimizer`.

The optimizer backend supported on the suite, is the torch compile backend on non-experimental mode,
 and the `''` or `None` which will do the same as `eager` mode and do anything, and `'jit'` which will
 try to `torch.jit.script` the operation.

You can use the `BENCHMARK_OPTS` on `make benchmark` to overload the default options we use on pytest-benchmark.

We are using as default:
- the warmup, because the optimizer/jit may had an overhead.
- the group: to display the benchmark per each test
- the precision: to have a better precision on the results
- the default for `BENCHMARK_BACKENDS` are `'inductor,eager'`.
- the default for `BENCHMARK_SOURCE` is `benchmarks/`.

You can also run the benchmark within docker:
```console
$ make benchmark-docker
```

which will build and run the image [docker/Dockerfile.benchmark](docker/Dockerfile.benchmark).
 The benchmark command can be used within `BENCHMARK_BACKENDS` and `BENCHMARK_SOURCE`.

# Coding Standards

This section provides general guidance for developing code for the project. The following rules will serve as a guide in
writing high-quality code that will allow us to scale the project and ensure that the code base remains readable and
maintainable.

- Use meaningful names for variables, functions, and classes.

- Write small incremental changes:

  - To have a linear and clean commits history, we recommend committing each small change that you do to the
    source code.
  - Clear commit messages will help to understand the progress of your work.
  - Please, avoid pushing large files.

- Add tests:
  - Tests are crucial and we expect you to write unit tests for each of the functionalities that you implement.
    It is also a good idea to group the tests for functionalities
  - At [testing/](./testing/) directory we have a bunch of functions to help you to produce meaningful tests. Feel free,
    to add any functionality that you think is essential and can be used with the test suite. Under this
    testing, directory should go all code which are needed under the tests and aren't tests or pytest configs (fixtures,
    etc).

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

- Tests should cover different devices (`CPU`, `CUDA`, etc), dtypes, and different input batch sizes. The `device`, and
  `dtype`, are generated from the arguments (`--dtype` and `--device`) as explained before. These arguments when invoking the
  tests suits with pytest, will generate all possibilities, providing fixtures for all functions. See an example:

    ```python
    import pytest

    @pytest.mark.parametrize("batch_size", [1, 2, 5])
    def test_smoke(batch_size, device, dtype):
        x = torch.rand(batch_size, 2, 3, device=device, dtype=dtype)
        assert x.shape == (batch_size, 2, 3)
    ```

- We give support to static type checker for Python >= 3.8

  - Please, read
    [MyPy cheatsheet](https://mypy.readthedocs.io/en/stable/cheat_sheet_py3.html#type-hints-cheat-sheet-python-3) for
    Python 3.
  - It is recommended to use typing inside the function, **when** it would increase readability.
  - Try to use all things available under `kornia.core`, e.g. `from kornia.core import Tensor`
  - For modules which not support anymore `JIT` consider, adding `from __future__ import annotations`, to enable the
    new features of typing.
  - **Always** type function input and output, e.g.:
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

- We suggest using new Python 3's f-Strings improved string formatting syntax:

  Guidelines: [PEP 498 - Literal String Interpolation](https://peps.python.org/pep-0498/)

- Format your code:

  - We follow [PEP8 style guide](https://www.python.org/dev/peps/pep-0008).
  - Use `pre-commit` to autoformat each commit before push: [pre-commit.com](https://pre-commit.com)
    To do so, just install it for this repository by running the command: `pre-commit install` on your terminal

- Changes to PEP8:
  - Line length is 120 characters.
  - W504 (line break after binary operator) is sometimes acceptable. E.g.

    ```python
    determinant = A[:, :, 0:1, 0:1] * A[:, :, 1:2, 1:2] -
                  A[:, :, 0:1, 1:2] * A[:, :, 1:2, 0:1])
    ```

-  Using 3rd party libraries:
  - Everything from the standard library (https://docs.python.org/3/library/) and PyTorch (https://pytorch.org/) is OK.
    It doesn’t mean, that one should import `urllib` just because, but doing it when needed is fine.

# Pull Request

Once you finish implementing a feature or bug fix, please send a Pull Request to https://github.com/kornia/kornia
through the website.

If you are not familiar with creating a Pull Request, here are some guides:

- http://stackoverflow.com/questions/14680711/how-to-do-a-github-pull-request
- https://help.github.com/articles/creating-a-pull-request

Once your pull request is created, our continuous build system will check your pull request. Continuous build will
test that:
- [pytest](https://docs.pytest.org/en/latest) all tests pass.
- Test coverage remains high. Please add unit tests so we maintain our code coverage.
- Typing with [mypy](http://mypy-lang.org) type checks the Python code.
- If the docs can be generated successfully
- [pre-commit ci](https://pre-commit.ci)
  - [ruff](https://pypi.org/project/ruff/) accepts the code style (our guidelines are based on PEP8) and checks if the code
    is well formatted
  - [docformatter](https://pypi.org/project/docformatter/) checks if the code docstrings are well formatted
  - and some other checks. Check our [pre-commit config](./.pre-commit-config.yaml)

If your code fails one of these checks, you will be expected to fix your pull request before it is considered.

# Licence

By contributing to the project, you agree that your contributions will be licensed under the Apache LICENSE. Check the
complete license [here](./LICENSE)
