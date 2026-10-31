kornia.geometry.grid
====================

.. meta::
   :description: The kornia.geometry.grid module provides functions for generating coordinate grids for images. It includes functions for creating 2D and 3D mesh grids with optional coordinate normalization, making it easy to work with grid-based transformations and sampling operations in computer vision tasks.

.. currentmodule:: kornia.geometry.grid

Grids of pixel and voxel coordinates. :func:`create_meshgrid` returns the ``(1, H, W, 2)`` grid of ``(x, y)``
coordinates of an image and :func:`create_meshgrid3d` the ``(1, D, H, W, 3)`` grid of ``(d, x, y)`` coordinates of a
volume, normalized to :math:`[-1, 1]` for :func:`torch.nn.functional.grid_sample` by default or in pixels. See
:ref:`Coordinates and sizes <coordinate-conventions>` for the pixel-centre and ``align_corners`` conventions.

.. autofunction:: create_meshgrid
.. autofunction:: create_meshgrid3d
