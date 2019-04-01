:github_url: https://github.com/arraiy/torchgeometry
             
torchgeometry
=============

The *PyTorch Geometry* (TGM) package is a geometric computer vision library for `PyTorch <https://pytorch.org/>`_.

It consists of a set of routines and differentiable modules to solve generic geometry computer vision problems. At its core, the package uses *PyTorch* as its main backend both for efficiency and to take advantage of the reverse-mode auto-differentiation to define and compute the gradient of complex functions.

In this first version, we provide different functions designed mainly for computer vision applications, such as image and tensors warping modules which rely on the epipolar geometry theory. The roadmap will include adding more and more functionality so that developers in the short term can use the package for the purpose of optimizing their loss functions to solve geometry problems.

TGM focuses on Image and tensor warping functions such as:

* Calibration
* Epipolar geometry
* Homography
* Depth

.. toctree::
   :maxdepth: 2
   :caption: Package Reference

   core
   image
   losses
   metrics
   contrib
   augmentation
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
