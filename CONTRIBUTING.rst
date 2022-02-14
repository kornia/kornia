Contributing to Kornia
======================

**Welcome !!** This is the Kornia library contributors corner. If you are reading this, it means that you have interest in **Differentiable Computer Vision**, and willing to contribute to the project.

Everyone is welcomed to get involved with the project. There are different ways in how you can put your two cents:

1. Ask/Answer questions using the ``#kornia`` tag in `PyTorch Discuss <https://discuss.pytorch.org>`_

   - Please, don't use GitHub issues for Q&A.
   - In case you are a developer and want to learn more about the PyTorch ecosystem, we suggest you to join the PyTorch slack. You can apply using this form: `https://bit.ly/ptslack <https://bit.ly/ptslack>`_


2. Report bugs through `GitHub issues <https://github.com/kornia/kornia/issues>`_:

   - Do a quick search first to see whether others reported a similar issue.
   - In case you find an unreported bug, please open a new ticket.
   - Try to provide as much information as possible. Some tips:

     - Clear title and description of the issue.
     - Explain how to reproduce the error.
     - Report your packages versions to facilitate the task.
     - Try to include a code sample/test that raises the error.

3. Fix a bug or develop a feature from the roadmap:

   - We will always have an open ticket showing the current roadmap.
   - Pick an unassigned feature (or potentially propose new one) or an open bug ticket.
   - Follow the instructions from Developing Kornia in order to setup your development environment and start coding.
   - Checkout our coding conventions. See more details below.
   - Run the test framework locally and make sure all works as expected before sending a pull request.
   - Open a Pull Request, get the green light from the CI and get your code merged.

4. Donate resources to the project:

   - In case you are an organization/institution that want to give support, sponsor or just use the project, please contact us.
   - We are open to start any kind of collaboration and hear feedback from you.
   - We pretend to provide features on demand. Reach us !
   - Currently looking for some kind of server donation in order to test *CUDA* code. (Please contact us).


Developing Kornia
=================

In order to start to develop, please follow the steps below:

1. Uninstall all existing installs:

.. code:: bash

    pip uninstall kornia
    pip uninstall kornia  # run this command twice

2. Clone a copy of Kornia from source:

.. code:: bash

    git clone https://github.com/kornia/kornia.git
    cd kornia

3. Create a new branch with a meaningful name reflecting your contribution. See an example:

.. code:: bash

    git checkout -b feat/foo_feature
    # or
    git checkout -b fix/bar_bug

4. Assuming that you are on ubuntu 16.04, with nvidia-drivers installed. In bash, source the ``path.bash.inc`` script.  This will install a local conda environment under ``./.dev_env``, which includes pytorch and some dependencies (no root required).

.. code:: bash

   source ./path.bash.inc
   python setup.py develop
   python -c "import kornia; print(kornia.__version__)"

To install, or update the conda environment run ``setup_dev_env.sh``

.. code:: bash

    ./setup_dev_env.sh

Coding Standards
================

This section provides general guidance for developing code for the project. The following rules will serve as guide in writing high-quality code that will allow us to scale the project and ensure that the code base remains readable and maintainable.

- Use meaningful names for variables, functions and classes.

- Write small incremental changes:

  - In order to have a linear and clean commits history, we recommend to commit each small change that you do to the source code.
  - Clear commit messages will help to understand the progress of your work.
  - Please, avoid pushing large files.

- Add tests:

  - Tests are crucial and we expect you to write unit test for each of the functionalities that you implement.
    It is also a good idea to group the tests for functionalities

  .. code:: python

        class TestMyFunction:
            def test_smoke(self):
                # check defaults parameters, i/o shapes
                pass

            def test_feature_foo(self):
                # test basic functionality
                pass

             def test_feature_bar(self):
                 # test another functionality
                 pass

             def test_gradcheck(self):
                 # test the functionality gradients
                 pass

             def test_jit(self):
                 #  test the functionality using jit modules
                 pass

  - Tests should cover different devices (CPU and CUDA) and different input batch size. See an example:

  .. code:: python

   @pytest.mark.parametrize("device_type", ("cpu", "cuda"))
   @pytest.mark.parametrize("batch_size", [1, 2, 5])
   def test_smoke(batch_size, device_type):
       x = torch.rand(batch_size, 2, 3)
       x = x.to(torch.device(device_type))
       assert x.shape == (batch_size, 2, 3), x.shape

- We give support to static type checker for Python >= 3.6

  - Please, read `MyPy cheatsheet <https://mypy.readthedocs.io/en/stable/cheat_sheet_py3.html#type-hints-cheat-sheet-python-3>`_ for Python 3.
  - It is recommended to use typing inside the function, when it would increase readability.
  - Always type function input and output, e.g.:

.. code:: python

    def homography_warp(patch_src: torch.Tensor,
                        dst_homo_src: torch.Tensor,
                        dsize: Tuple[int, int],
                        mode: str = 'bilinear',
                        padding_mode: str = 'zeros') -> torch.Tensor:

- We suggest to use new Python 3's f-Strings improved string formatting syntax:

  Guidelines: https://realpython.com/python-f-strings/

.. code:: python

    python_version: int = 3
    print(f"This is an example to use Python {python_version}'s f-Strings")

- Format your code:

  - We follow `PEP8 style guide <https://www.python.org/dev/peps/pep-0008>`_.
  - Use ``pre-commit`` to autoformat each commit before push: https://pre-commit.com/.
    For doing so, just install it for this repository:

  .. code:: bash

    pre-commit install

- Changes to PEP8:

  - Line length is 120 char.
  - W504 (line break after binary operator) is sometimes acceptable. E.g.

.. code:: python

   determinant = A[:, :, 0:1, 0:1] * A[:, :, 1:2, 1:2] -
                 A[:, :, 0:1, 1:2] * A[:, :, 1:2, 0:1])

-  Using 3rd party libraries:

  - Everything from standard library (https://docs.python.org/3/library/) and PyTorch (https://pytorch.org/) is OK.
    It does`t mean, that one should import urllib  just because, but doing it when needed is fine.



Pull Request
============

Once you finish implementing a feature or bug-fix, please send a Pull Request to
https://github.com/kornia/kornia through the website.

If you are not familiar with creating a Pull Request, here are some guides:

- http://stackoverflow.com/questions/14680711/how-to-do-a-github-pull-request
- https://help.github.com/articles/creating-a-pull-request

Once your pull request is created, our continuous build system will check your pull request. Continuous build will test that:

- `pytest <https://docs.pytest.org/en/latest>`_ all tests pass.
- `flake8 <https://pypi.org/project/flake8/>`_ accepts the code style (our guidelines are based on PEP8).
- `mypy <http://mypy-lang.org>`_ type checks the Python code.
- The docs can be generated successfully
- Test coverage remains high. Please add unit tests so we maintain our code coverage.

If your code fails one of these checks, you will be expected to fix your pull request before it is considered.



Unit testing
============

To run the test suite locally, make sure that you have activated the conda environment, then:

.. code:: bash

    make test

Licence
=======

By contributing to the project, you agree that your contributions will be licensed under the LICENSE file in the root directory of this source tree.
