Installation
============

To install *Kornia*, you can do it in two different ways: using the provided `PyPi
<https://pypi.org/project/kornia/>`_ wheels or directly from source.

.. note::
    *Kornia only has as a dependency Pytorch.*

1. From pip:

.. code:: bash

    pip install kornia
    pip install kornia[x]  # to get the training API !

2. From source:

.. code:: bash

    python setup.py install

3. From source using pip:

.. code:: bash

    pip install git+https://github.com/kornia/kornia

Once you succeeded installing *Kornia* check you can import:

.. code:: bash

    python -c "import kornia; print(kornia.__version__)"
