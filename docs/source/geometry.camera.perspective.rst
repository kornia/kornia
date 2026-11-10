Perspective Camera
------------------

.. meta::
   :description: The kornia.geometry.camera.perspective module provides functions for 3D point projection and unprojection in perspective camera models. These functions are essential for transforming 3D camera-frame coordinates into 2D image plane coordinates and vice versa, enabling advanced computer vision applications like depth estimation and 3D scene reconstruction.

.. currentmodule:: kornia.geometry.camera.perspective

:func:`project_points` maps camera-frame 3D points to pixels with a ``(*, 3, 3)`` ``K`` and no extrinsics, and
:func:`unproject_points` maps pixels at given depths back to camera-frame points. For world points, use
:class:`~kornia.geometry.camera.pinhole.PinholeCamera`; for every pixel of a depth map,
:func:`~kornia.geometry.depth.depth_to_3d_v2`; to compose the projection with a lens model, the normalized-plane
functions of :doc:`geometry.camera`.

.. autofunction:: project_points
.. autofunction:: unproject_points
