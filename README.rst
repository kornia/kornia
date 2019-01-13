.. raw:: html

  <p align="center">
    <img width="50%" src="https://github.com/arraiy/torchgeometry/blob/master/docs/source/_static/img/torchgeometry_logo.svg" />
  </p>

--------------------------------------------------------------------------------

.. image:: https://travis-ci.com/arraiy/torchgeometry.svg?token=M8pF2LfWb2ZxBDWRRvcP&branch=master
    :target: https://travis-ci.com/arraiy/torchgeometry
    
.. image:: https://badge.fury.io/py/torchgeometry.svg
    :target: https://badge.fury.io/py/torchgeometry

`Documentation <https://arraiy.github.io/torchgeometry/>`_

The *PyTorch Geometry* package is a geometric computer vision library for `PyTorch <https://pytorch.org/>`_.

It consists of a set of routines and differentiable modules to solve generic geometry computer vision problems. At its core, the package uses *PyTorch* as its main backend both for efficiency and to take advantage of the reverse-mode auto-differentiation to define and compute the gradient of complex functions.

Installation
============

**From pip:**

.. code:: bash

    pip install torchgeometry

**From source:**

.. code:: bash

    python setup.py install

**From source using pip:**

.. code:: bash

    pip install git+https://github.com/arraiy/torchgeometry

Quick Usage
===========

.. code:: python

 import torch
 import torchgeometry as tgm

 x_rad = tgm.pi * torch.rand(1, 3, 3)
 x_deg = tgm.rad2deg(x_rad)

 torch.allclose(x_rad, tgm.deg2rad(x_deg))  # True

Examples
========

Run our Jupyter notebooks `examples <https://github.com/arraiy/torchgeometry/tree/master/examples/>`_ to learn to use the library.


Cite
============

If you are using torchgeometry in your research-related documents, it is recommended that you cite the poster.

.. code:: bash

 @misc{Arraiy2018,
  author    = {E. Riba, M. Fathollahi, W. Chaney, E. Rublee and G. Bradski}
  title     = {torchgeometry: when PyTorch meets geometry},
  booktitle = {PyTorch Developer Conference},
  year      = {2018},
  url       = {https://drive.google.com/file/d/1xiao1Xj9WzjJ08YY_nYwsthE-wxfyfhG/view?usp=sharing}
 }


Contributing
============
We appreciate all contributions. If you are planning to contribute back bug-fixes, please do so without any further discussion. If you plan to contribute new features, utility functions or extensions, please first open an issue and discuss the feature with us. Please, consider to read the `CONTRIBUTING <https://github.com/arraiy/torchgeometry/CONTRIBUTING.rst>`_ notes.
