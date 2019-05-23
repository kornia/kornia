:github_url: https://github.com/arraiy/kornia
             
kornia
======

Kornia is a differentiable computer vision library for PyTorch.

It consists of a set of routines and differentiable modules to solve generic computer vision problems. At its core, the package uses PyTorch as its main backend both for efficiency and to take advantage of the reverse-mode auto-differentiation to define and compute the gradient of complex functions.
Overview

Inspired by OpenCV, this library is composed by a subset of packages containing operators that can be inserted within neural networks to train models to perform image transformations, epipolar geometry, depth estimation, and low level image processing such as filtering and edge detection that operate directly on tensors.

At a granular level, Kornia is a library that consists of the following components:

.. toctree::
   :maxdepth: 2
   :caption: Package Reference

   color
   contrib
   feature
   filters
   geometry
   losses
   utils

.. toctree::
   :maxdepth: 2
   :caption: Tutorials

   warp_affine
   warp_perspective
   gaussian_blur

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
