:github_url: https://github.com/arraiy/torchgeometry
             
torchgeometry
=============

The `torchgeometry <https://github.com/arraiy/torchgeometry/>`_ is a differentiable computer vision library for `PyTorch <https://pytorch.org/>`_.

It consists of a set of routines and differentiable modules to solve generic geometry problems. At its core, the package uses PyTorch as its main backend both for efficiency and to take advantage of autograd to define and compute the gradient of complex functions. In this first version, we provide different functions designed mainly for computer vision applications, such as image and tensors warping modules which rely on the epipolar geometry theory. The roadmap will include adding more and more functionality so that developers in the short term can use the package for the purpose of optimizing their loss functions to solve geometry problems.

Version v0.1.0 focuses on Image and tensor warping functions such as:

* Calibration
* Epipolar geometry
* Homography
* Depth

.. automodule:: torchgeometry

Pinhole
--------

.. autofunction:: inverse
.. autofunction:: inverse_pose
.. autofunction:: pinhole_matrix
.. autofunction:: inv_pinhole_matrix
.. autofunction:: scale_pinhole
.. autofunction:: homography_i_H_ref

Conversions
-----------

.. autofunction:: rad2deg
.. autofunction:: deg2rad
.. autofunction:: convert_points_from_homogeneous
.. autofunction:: convert_points_to_homogeneous
.. autofunction:: transform_points

Utilities
---------

.. autofunction:: tensor_to_image
.. autofunction:: image_to_tensor

Containers
----------

.. autoclass:: HomographyWarper
    :members:

.. autoclass:: DepthWarper
    :members:
