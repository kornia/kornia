Installation
============

.. meta::
   :description: How to install Kornia with pip, conda or from source, and how to verify the installation. Kornia requires PyTorch 2.5.1 or newer and works on CPU, CUDA and Apple MPS devices.

*Kornia* is distributed as pure-Python wheels on `PyPI <https://pypi.org/project/kornia/>`_ and on
`conda-forge <https://anaconda.org/conda-forge/kornia>`_. It requires
`PyTorch <https://pytorch.org/get-started/locally/>`_ 2.5.1 or newer; the only other dependencies are
``numpy`` and `kornia-rs <https://github.com/kornia/kornia-rs>`_ (the Rust image I/O
backend used by :mod:`kornia.io`). Install PyTorch first if you need a specific CUDA build.

.. tab-set::

   .. tab-item:: pip

      .. code-block:: bash

         pip install kornia

   .. tab-item:: conda

      .. code-block:: bash

         conda install -c conda-forge kornia

   .. tab-item:: From source

      .. code-block:: bash

         pip install git+https://github.com/kornia/kornia

      or, from a local clone, an editable install for development:

      .. code-block:: bash

         git clone https://github.com/kornia/kornia.git
         cd kornia
         pip install -e .

Once the installation has finished, check that you can import the package:

.. code-block:: bash

    python -c "import kornia; print(kornia.__version__)"

Pretrained models (RT-DETR, LoFTR, DISK, SAM, ...) download their checkpoints on first use, so no
extra installation step is needed for them.

Optional extras
---------------

A few Kornia features wrap third-party packages that are not installed with the base wheel. They are
declared as `extras <https://packaging.python.org/en/latest/specifications/dependency-specifiers/#extras>`_,
so you only pay for the ones you use. If one of the extras below is missing, the corresponding Kornia
object raises an ``ImportError`` naming the extra to install. The installation mode changes this: set
``kornia.config.kornia_config.lazyloader.installation_mode``, or the ``KORNIA_INSTALLATION_MODE``
environment variable before Kornia is imported, to ``"ask"`` to be asked on an interactive terminal
whether to install the extra, or to ``"auto"`` to install the declared extra with
``pip install "kornia[<extra>]"`` without asking. Without an interactive terminal, ``"ask"`` raises the
same ``ImportError``: for example in a CI job, when output is redirected to a file, or in a Jupyter
notebook, whose kernel's stdin is not a terminal (use ``"auto"`` there). The default is ``"raise"``.

.. list-table::
   :header-rows: 1
   :widths: 15 55 30

   * - Extra
     - What it enables
     - Install command
   * - ``image``
     - Pillow-backed PIL input/output and display helpers, plus remote image decoding in :func:`kornia.io.get_sample_images`
     - ``pip install "kornia[image]"``
   * - ``onnx``
     - :doc:`kornia.onnx </onnx>`, ONNX export of Kornia modules, and :class:`~kornia.feature.OnnxLightGlue`
     - ``pip install "kornia[onnx]"``
   * - ``sd``
     - ``kornia.filters.StableDiffusionDissolving``
     - ``pip install "kornia[sd]"``
   * - ``dev``
     - Contributor environment: the test and lint toolchain (includes ``kornia[onnx]``)
     - ``pip install -e ".[dev]"``
   * - ``docs``
     - Documentation toolchain, on top of ``dev``
     - ``pip install -e ".[dev,docs]"``

Next steps
----------

- :doc:`introduction` -- what Kornia is and what each module contains.
- :doc:`conventions` -- the tensor layout, coordinate and angle conventions to know before writing code.
- :doc:`Applications </applications/intro>` -- end-to-end guides, or the :doc:`API reference </api>`.
