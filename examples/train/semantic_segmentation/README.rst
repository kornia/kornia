Semantic Segmentation Example
=============================

This is a toy example implementing a semantic segmentation application.

1. Install dependencies

.. code-block:: python

  pip install -r requirements.txt

2. Execute the script: The entry point to this example is the file

.. code-block:: python

  python main.py

3. Modify the hyper-parameters in `config.yml` and execute

.. code-block:: python

  python main.py num_epochs=50

4. Sweep hyper-parameters

.. code-block:: python

  python main.py --multirun num_epochs=1 lr=1e-3,1e-4

Explore hydra to make cool stuff with the config files: https://hydra.cc/
