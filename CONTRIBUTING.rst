Contributing to PyTorch Geometry
================================

**Welcome !!** This is the PyTorch Geometry library contributors corner. If you are reading this, it means that you have interest in **Computer Vision and Geometry**, and willing to contribute to the project.

Everyone is welcomed to get involved with the project. There are different ways in how you can put your two cents:


1. Ask/Answer questions in the ``#geometry`` PyTorch Slack channel:

 - We suggest you to join the channel in order to get more involved to the PyTorch ecosystem. Send an email to `slack@pytorch.org <slack@pytorch.org>`_ in order to get access.
 - Please, don't use GitHub issues for Q&A.

2. Report bugs through `GitHub issues <https://github.com/arraiy/torchgeometry/issues>`_:

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
 - Follow the instructions from Developing PyTorch Geometry in order to setup your development environment and start coding.
 - Checkout our coding conventions. See more details below.
 - Run the test framework locally and make sure all works as expected before sending a pull request.
 - Open a Pull Request, get the green light from the CI and get your code merged.

4. Donate resources to the project:

 - In case you are an organization/institution that want to give support, sponsor or just use the project, please contact us.
 - We are open to start any kind of collaboration and hear feedback from you.
 - We pretend to provide features on demand. Reach us !
 - Currently looking for some kind of server donation in order to test *CUDA* code. (Please contact us).


Developing PyTorch Geometry
===========================

In order to start to develop, please follow the steps below:

1. Uninstall all existing installs:

.. code:: bash

    pip uninstall torchgeometry
    pip uninstall torchgeometry  # run this command twice

2. Clone a copy of PyTorch Geometry from source:

.. code:: bash

    git clone https://github.com/arraiy/torchgeometry.git
    cd torchgeometry

3. Create a new branch with a meaningful name reflecting your contribution. See an example:

.. code:: bash

    git checkout -b feat/foo_feature
    # or
    git checkout -b fix/bar_bug

4. Assuming that you are on ubuntu 16.04, with nvidia-drivers installed. In bash, source the ``path.bash.inc`` script.  This will install a local conda environment under ``./.dev_env``, which includes pytorch and some dependencies (no root required).

.. code:: bash

   source ./path.bash.inc
   python -c "import torchgeometry; print(torchgeometry.__version__)"

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
  - Tests should cover different devices (CPU and CUDA) and different input batch size. See an example:

.. code:: bash

   @pytest.mark.parametrize("device_type", ("cpu", "cuda"))
   @pytest.mark.parametrize("batch_size", [1, 2, 5])
   def test_smoke(batch_size, device_type):
       x = torch.rand(batch_size, 2, 3)
       x = x.to(torch.device(device_type))
       assert x.shape == (batch_size, 2, 3), x.shape

- We give support to static type checker for Python >= 3.6

  - Please, read `MyPy cheatsheet <https://mypy.readthedocs.io/en/stable/cheat_sheet_py3.html#type-hints-cheat-sheet-python-3>`_ for Python 3.

- Format your code:

  - We follow `PEP8 style guide <https://www.python.org/dev/peps/pep-0008>`_.
  - Use ``autopep`` to autoformat: https://pypi.org/project/autopep8/#id3

Pull Request
============

Once you finish implementing a feature or bug-fix, please send a Pull Request to
https://github.com/arraiy/torchgeometry through the website.

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

    python setup.py test

Licence
=======

By contributing to the project, you agree that your contributions will be licensed under the LICENSE file in the root directory of this source tree.
