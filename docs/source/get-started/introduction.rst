What is Kornia ?
================

Kornia is a differentiable library that allows classical computer vision to be integrated into deep learning models.

It consists of a set of routines and differentiable modules to solve generic computer vision problems.
At its core, the package uses PyTorch as its main backend both for efficiency and to take advantage of
the reverse-mode auto-differentiation to define and compute the gradient of complex functions.

.. image:: https://raw.githubusercontent.com/kornia/kornia/master/docs/source/_static/img/hakuna_matata.gif
   :align: center

The library is composed by a subset of packages containing operators that can be inserted
within neural networks to train models to perform image transformations, epipolar geometry, depth estimation,
and low level image processing such as filtering and edge detection that operate directly on tensors.

Why Kornia ?
------------

With *Kornia* we fill the gap between classical and deep computer vision that implements
standard and advanced vision algorithms for AI:

1. **Computer Vision:** Kornia fills the gap between Classical and Deep computer Vision.
2. **Differentiable:** We leverage the Computer Vision 2.0 paradigm.
3. **Open Source:** Our libraries and initiatives are always according to the community needs.
4. **PyTorch:** At our core we use PyTorch and its Autograd engine for its efficiency.
