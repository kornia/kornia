kornia.geometry.line
====================

.. meta::
   :description: The kornia.geometry.line module provides functionality for working with lines and line segments in geometric space. It includes classes such as ParametrizedLine for line representation, Hyperplane for planes in 3D space, and functions like fit_line and fit_plane for fitting lines and planes to data points. This module is essential for tasks like line segment matching and line fitting in computer vision and geometric analysis.

.. currentmodule:: kornia.geometry.line

Lines in any dimension and planes in 3D, as objects. :class:`ParametrizedLine` is the line :math:`o + t d` from an
origin and a direction, or through two points, with the points along it, the distances and projections of points and
the intersection with a plane; :func:`fit_line` fits one to each of ``B`` point sets by total least squares, optionally
weighted. :class:`~kornia.geometry.plane.Hyperplane` is the plane :math:`n \cdot x + d = 0` from a normal and an
offset, a normal and a point, or three points, with signed distances and projections;
:func:`~kornia.geometry.plane.fit_plane` fits one to 3D points. The distances and projections are Euclidean for a unit
direction or normal, which the two-point and three-point constructors and the fits return. For 2D lines given as
coefficients :math:`ax + by + c = 0`, use :func:`~kornia.geometry.linalg.point_line_distance`.

.. autoclass:: ParametrizedLine
   :members:
   :special-members: __init__

``kornia.geometry.ray.Ray`` is an alias of :class:`ParametrizedLine`.

.. autofunction:: fit_line

.. currentmodule:: kornia.geometry.plane

.. autoclass:: Hyperplane
   :members:

.. autofunction:: fit_plane
