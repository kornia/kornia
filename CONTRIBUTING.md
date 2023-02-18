# Contributing to Kornia
**Welcome !!** This is the Kornia library contributors corner. If you are reading this, it means that you have interest
in **Differentiable Computer Vision**, and willing to contribute to the project.

Everyone is welcomed to get involved with the project. There are different ways in how you can put your two cents:

1. Ask/Answer questions:
    - Where can you ask questions?
      1. using the github discussion at Kornia repo: [GH Discussions](https://github.com/kornia/kornia/discussions)
      2. our slack workspace to keep in touch with our core contributors and the community:
         [Join Here](https://join.slack.com/t/kornia/shared_invite/zt-csobk21g-2AQRi~X9Uu6PLMuUZdvfjA)
      3. using the `#kornia` tag in [PyTorch Discuss](https://discuss.pytorch.org>)
    - Please, don't use GitHub issues for Q&A.
    - In case you are a developer and want to learn more about the PyTorch ecosystem, we suggest you to join the PyTorch
      slack. You can apply using this form: [https://bit.ly/ptslack](https://bit.ly/ptslack>)

2. Report bugs through [GitHub issues](https://github.com/kornia/kornia/issues>):
   - Do a quick search first to see whether others reported a similar issue.
   - In case you find an unreported bug, please open a new ticket.
   - Try to provide as much information as possible. Report using one of the available templates. Some tips:
     - Clear title and description of the issue.
     - Explain how to reproduce the error.
     - Report your packages versions to facilitate the task.
     - Try to include a code sample/test that raises the error.

3. Fix a bug or develop a feature from the roadmap:
   - We will always have an open ticket showing the current roadmap.
   - Pick an unassigned feature (or potentially propose new one) or an open bug ticket.
   - Follow the instructions from [Developing Kornia](#developing-kornia) in order to setup your development
     environment and start coding.
   - Checkout our coding conventions. See more details below.
   - Run the test framework locally and make sure all works as expected before sending a pull request.
   - Open a Pull Request, get the green light from the CI and get your code merged.

4. Donate resources to the project:
   - In case you are an organization/institution that want to give support, sponsor or just use the project, please
     contact us.
     - [opencollective.com/kornia](https://opencollective.com/kornia)
     - [github.com/sponsors/kornia](https://github.com/sponsors/kornia)
   - We are open to start any kind of collaboration and hear feedback from you.
   - We pretend to provide features on demand. Reach us !
   - Currently looking for some kind of server donation in order to test *CUDA* code. (Please contact us).

# Developing Kornia

In order to start to develop, please follow the steps below:

1. Fork the [kornia repository](https://github.com/kornia/kornia) by clicking on the
[fork](https://github.com/kornia/kornia/fork) button on the repository page. This will create a copy of the kornia
repository under your GitHub account.


2. Clone your fork of Kornia, and add the kornia repository as a remote:
    ```bash
    $ git clone git@github.com:<your Github username>/kornia.git
    $ cd kornia
    $ git remote add upstream https://github.com/kornia/kornia.git
    ```

3. Create a new branch with a meaningful name reflecting your contribution. See an example:
    ```bash
    $ git checkout upstream/master -b feat/foo_feature
    # or
    $ git checkout upstream/master -b fix/bar_bug
    ```
    🚨 **Do not** work on the `master` branch!

4. Creating development environment

    **Using kornia script**

    Assuming that you are on ubuntu, with nvidia-drivers installed. In bash, source the ``path.bash.inc`` script.
    This will install a local conda environment under ``./.dev_env``, which includes pytorch and some dependencies
    (no root required).

    ```bash
    $ source ./path.bash.inc
    $ python setup.py develop
    $ python -c "import kornia; print(kornia.__version__)"
    ```

    To install, or update the conda environment run ``setup_dev_env.sh``

    ```bash
    $ ./setup_dev_env.sh
    ```


    **Manually setup**

    Otherwise, using a virtual environment of you preference. We recommend to use
    [conda](https://docs.conda.io/en/latest/)) because it facilitates the Pytorch setup, mainly for those who have a
    **GPU** available.

    Example of creating and activate a virtualenv under `venv` name:
    ```bash
    # Using virtualenv
    $ virtualenv venv -p <your python version / alias> # e.g python3.10
    $ ./venv/bin/activate

    # Using conda
    $ conda create -p venv python=<language version> # e.g 3.10
    $ conda activate venv/
    ```

    Setup Pytorch and Kornia:
    ```bash
    # Installing pytorch: https://pytorch.org/get-started/locally/
    # With pip
    $ pip install torch
    # With conda
    $ conda install pytorch cudatoolkit -c pytorch -c nvidia # For GPU env
    # or
    $ conda install pytorch cpuonly -c pytorch # For CPU env

    # Installing Kornia for development
    $ pip install -e .[dev]

    # If you want to contribute for the documentation
    $ pip install -e .[docs]
    ```

    **Attention**: If *Kornia* was already installed in you virtual environment, remove it with
    `pip uninstall kornia` before reinstalling it in editable mode with the `-e` flag.

5. Develop the code on your branch, and before create the pull request, make sure to ensure the code pass the checks.

    As you develop your code, you should also create test cases for your code. As well as, In addition to ensuring that
    the other tests continue to pass. You can run the tests with:
    ```bash
    $ pytest test/<TEST_TO_RUN>.py --dtype=float32,float64 --device=all
    ```
    With the `dtype` argument, run the tests using tensors with all `dtypes` desired. Options: `bfloat16`, `float16`,
    `float32`, `float64` and `all`.

    In the same way, the `device`, will run the tests using tensors on the `device` desired. Options: `cpu`, `cuda`,
    `tpu`, `mps` or `all`.


    Kornia relies on [pre-commit](https://pre-commit.com) to run code quality tools. Make sure to have `pre-commit`
    under you dev environment, otherwise you can install the tools manually and run with the help of the available
    commands of the [Makefile](./Makefile). Read more about the code standards adopt [here](#coding-standards).


# Coding Standards

This section provides general guidance for developing code for the project. The following rules will serve as guide in
writing high-quality code that will allow us to scale the project and ensure that the code base remains readable and
maintainable.

- Use meaningful names for variables, functions and classes.

- Write small incremental changes:

  - In order to have a linear and clean commits history, we recommend to commit each small change that you do to the
    source code.
  - Clear commit messages will help to understand the progress of your work.
  - Please, avoid pushing large files.

- Add tests:
  - Tests are crucial and we expect you to write unit test for each of the functionalities that you implement.
    It is also a good idea to group the tests for functionalities

    ```python
    import pytorch
    from kornia.testing import BaseTester

    class TestMyFunction(BaseTester):
        # To compare the actual and expected tensors use `self.assert_close(...)`


        def test_smoke(self, device, dtype):
            # test the function with different parameters arguments, to check if the function at least runs with all the
            # arguments allowed.
            pass

        def test_exception(self, device, dtype):
            # tests the exceptions which can occurs on your function

            # example of how to properly test your exceptions
            # with pytest.raises(<raised Error>) as errinfo:
            #     your_function(<set of parameters that raise the error>)
            # assert '<msg of error>' in str(errinfo)

            pass

        def test_cardinality(self, device, dtype):
            # test if with different parameters the outputs shape is the expected
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

- Tests should cover different devices (`CPU`, `CUDA`, etc), dtypes and different input batch size. The device, and
  dtype, are generated from the arguments (`--dtype` and `--device`) as explained before. This arguments when invoking the
  tests suits with pytest, will generate all possibilities, providing as fixtures for all functions. See an example:

    ```python
    import pytest

    @pytest.mark.parametrize("batch_size", [1, 2, 5])
    def test_smoke(batch_size, device, dtype):
        x = torch.rand(batch_size, 2, 3, device=device, dtype=dtype)
        assert x.shape == (batch_size, 2, 3)
    ```

- We give support to static type checker for Python >= 3.7

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

- We suggest to use new Python 3's f-Strings improved string formatting syntax:

  Guidelines: [PEP 498 - Literal String Interpolation](https://peps.python.org/pep-0498/)

- Format your code:

  - We follow [PEP8 style guide](https://www.python.org/dev/peps/pep-0008).
  - Use `pre-commit` to autoformat each commit before push: [pre-commit.com](https://pre-commit.com)
    For doing so, just install it for this repository `pre-commit install`

- Changes to PEP8:
  - Line length is 120 char.
  - W504 (line break after binary operator) is sometimes acceptable. E.g.

    ```python
    determinant = A[:, :, 0:1, 0:1] * A[:, :, 1:2, 1:2] -
                  A[:, :, 0:1, 1:2] * A[:, :, 1:2, 0:1])
    ```

-  Using 3rd party libraries:

  - Everything from standard library (https://docs.python.org/3/library/) and PyTorch (https://pytorch.org/) is OK.
    It does’t mean, that one should import urllib  just because, but doing it when needed is fine.

# Pull Request

Once you finish implementing a feature or bug-fix, please send a Pull Request to https://github.com/kornia/kornia
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
  - [flake8](https://pypi.org/project/flake8/) accepts the code style (our guidelines are based on PEP8).
  - [black](https://black.readthedocs.io/en/stable/) checks if the code are well formatted
  - [docformatter](https://pypi.org/project/docformatter/) checks if the code docstrings are well formatted
  - and some others checks. Check our [pre-commit config](./.pre-commit-config.yaml)

If your code fails one of these checks, you will be expected to fix your pull request before it is considered.

# Licence

By contributing to the project, you agree that your contributions will be licensed under the apache LICENSE. Check the
complete license [here](./LICENSE)
