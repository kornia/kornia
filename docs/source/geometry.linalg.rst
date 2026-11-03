kornia.geometry.linalg
======================

.. meta::
   :description: The kornia.geometry.linalg module offers essential linear algebra utilities for geometric computations in computer vision and robotics. It includes functions for transforming points, calculating transformations, computing distances, and performing vector operations. Key functions include relative transformation, inverse transformation, point-line distance, squared norm, dot product, and Euclidean distance. These operations are fundamental for tasks such as geometric transformations, distance computations, and vector manipulations.

.. currentmodule:: kornia.geometry.linalg

Homogeneous transforms and the operations around them. :func:`compose_transformations` chains two ``4 x 4``
transforms, :func:`inverse_transformation` inverts a rigid one and :func:`relative_transformation` gives the transform
between two poses expressed in one frame; ``trans_ab`` maps points in frame ``b`` to frame ``a``.
:func:`transform_points` applies a ``(D + 1, D + 1)`` matrix to ``(..., N, D)`` points. :func:`euclidean_distance` and
:func:`point_line_distance` measure distances, and :func:`batched_dot_product` and :func:`batched_squared_norm` (also
exported as :func:`squared_norm`) reduce the last dimension.

.. autofunction:: relative_transformation
.. autofunction:: compose_transformations
.. autofunction:: inverse_transformation
.. autofunction:: transform_points
.. autofunction:: point_line_distance
.. autofunction:: batched_squared_norm
.. autofunction:: squared_norm
.. autofunction:: batched_dot_product
.. autofunction:: euclidean_distance
