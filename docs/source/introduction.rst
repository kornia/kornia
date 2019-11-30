Introduction
============

Kornia is a differentiable computer vision library for PyTorch.

It consists of a set of routines and differentiable modules to solve generic computer vision problems.
At its core, the package uses PyTorch as its main backend both for efficiency and to take advantage of 
the reverse-mode auto-differentiation to define and compute the gradient of complex functions.

Inspired by OpenCV, this library is composed by a subset of packages containing operators that can be inserted
within neural networks to train models to perform image transformations, epipolar geometry, depth estimation,
and low level image processing such as filtering and edge detection that operate directly on tensors.

**Why Kornia ?**
    With *Kornia* we fill the gap within the PyTorch ecosystem introducing a computer vision library that implements
    standard vision algorithms taking advantage of the different properties that modern frameworks for deep learning
    like PyTorch can provide:

    1. **Differentiability** for commodity avoiding to write derivative functions for complex loss  functions.

    2. **Transparency** to perform parallel or serial computing eitherin CPU or GPU devices using batches in a common API.

    3. **Distributed** for computing large-scale applications.

    4. **Production** ready using the *JIT* compiler.

Hightlighted Features
---------------------

At a granular level, Kornia is a library that consists of the following components:

+----------------------------------------------------------------------------+---------------------------------------------------------------------------------------------------------------------------------------+
| **Component**                                                              | **Description**                                                                                                                       |
+----------------------------------------------------------------------------+---------------------------------------------------------------------------------------------------------------------------------------+
| `kornia <https://kornia.readthedocs.io/en/latest/index.html>`_             | a Differentiable Computer Vision library like OpenCV, with strong GPU support                                                         |
+----------------------------------------------------------------------------+---------------------------------------------------------------------------------------------------------------------------------------+
| `kornia.color <https://kornia.readthedocs.io/en/latest/color.html>`_       | a set of routines to perform color space conversions                                                                                  |
+----------------------------------------------------------------------------+---------------------------------------------------------------------------------------------------------------------------------------+
| `kornia.contrib <https://kornia.readthedocs.io/en/latest/contrib.html>`_   | a compilation of user contrib and experimental operators                                                                              |
+----------------------------------------------------------------------------+---------------------------------------------------------------------------------------------------------------------------------------+
| `kornia.feature <https://kornia.readthedocs.io/en/latest/feature.html>`_   | a module to perform feature detection                                                                                                 |
+----------------------------------------------------------------------------+---------------------------------------------------------------------------------------------------------------------------------------+
| `kornia.filters <https://kornia.readthedocs.io/en/latest/filters.html>`_   | a module to perform image filtering and edge detection                                                                                |
+----------------------------------------------------------------------------+---------------------------------------------------------------------------------------------------------------------------------------+
| `kornia.geometry <https://kornia.readthedocs.io/en/latest/geometry.html>`_ | a geometric computer vision library to perform image transformations, 3D linear algebra and conversions using different camera models |
+----------------------------------------------------------------------------+---------------------------------------------------------------------------------------------------------------------------------------+
| `kornia.losses <https://kornia.readthedocs.io/en/latest/losses.html>`_     | a stack of loss functions to solve different vision tasks                                                                             |
+----------------------------------------------------------------------------+---------------------------------------------------------------------------------------------------------------------------------------+
| `kornia.utils <https://kornia.readthedocs.io/en/latest/utils.html>`_       | image to tensor utilities and metrics for vision problems                                                                             |
+----------------------------------------------------------------------------+---------------------------------------------------------------------------------------------------------------------------------------+
