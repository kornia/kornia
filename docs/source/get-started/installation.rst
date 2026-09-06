Installation
============

.. meta::
   :description: How to install Kornia with pip, conda or from source, and how to verify the installation. Kornia requires PyTorch 2.5.1 or newer and works on CPU, CUDA and Apple MPS devices.

*Kornia* is distributed as pure-Python wheels on `PyPI <https://pypi.org/project/kornia/>`_ and on
`conda-forge <https://anaconda.org/conda-forge/kornia>`_. It requires
`PyTorch <https://pytorch.org/get-started/locally/>`_ 2.5.1 or newer; the only other dependencies are
``numpy``, ``packaging`` and `kornia-rs <https://github.com/kornia/kornia-rs>`_ (the Rust image I/O
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
so you only pay for the ones you use. If one is missing, the corresponding Kornia object raises an
``ImportError`` naming the extra to install.

.. list-table::
   :header-rows: 1
   :widths: 15 55 30

   * - Extra
     - What it enables
     - Install command
   * - ``onnx``
     - :doc:`kornia.onnx </onnx>`, ONNX export of Kornia modules, and :class:`~kornia.feature.OnnxLightGlue`
     - ``pip install "kornia[onnx]"``
   * - ``sd``
     - ``kornia.filters.StableDiffusionDissolving``
     - ``pip install "kornia[sd]"``
   * - ``tracking``
     - :class:`~kornia.contrib.boxmot_tracker.BoxMotTracker`
     - ``pip install "kornia[tracking]"``
   * - ``segmentation``
     - :class:`~kornia.models.segmentation.segmentation_models.SegmentationModelsBuilder`
     - ``pip install "kornia[segmentation]"``
   * - ``dev``, ``docs``
     - Contributor environments: test, lint and documentation toolchains
     - ``pip install -e ".[dev]"``

Independently of the extras, the ``visualize`` and ``save`` helpers of the model wrappers need
`pillow <https://pypi.org/project/pillow/>`_ for ``output_type="pil"``.

Next steps
----------

- :doc:`introduction` -- what Kornia is and what each module contains.
- :doc:`conventions` -- the tensor layout, coordinate and angle conventions to know before writing code.
- :doc:`Applications </applications/intro>` -- end-to-end guides, or the :doc:`API reference </api>`.
