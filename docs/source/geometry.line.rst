kornia.geometry.line
====================

.. meta::
   :description: The kornia.geometry.line module provides functionality for working with lines and line segments in geometric space. It includes classes such as ParametrizedLine for line representation, Hyperplane for planes in 3D space, and functions like fit_line and fit_plane for fitting lines and planes to data points. This module is essential for tasks like line segment matching and line fitting in computer vision and geometric analysis.

.. currentmodule:: kornia.geometry.line

.. autoclass:: ParametrizedLine
   :members:
   :special-members: __init__

``kornia.geometry.ray.Ray`` is an alias of :class:`ParametrizedLine`.

.. autofunction:: fit_line

.. currentmodule:: kornia.geometry.plane

.. autoclass:: Hyperplane
   :members:

.. autofunction:: fit_plane
