kornia.geometry.camera
======================

.. meta::
   :description: The kornia.geometry.camera module provides a variety of functions for handling camera projections and distortions. It includes support for projecting 3D points to a 2D image plane, both with perspective and orthographic projections, as well as distortion models like affine and Kannala-Brandt. This module enables robust camera calibration and 3D scene transformations in computer vision applications.

.. currentmodule:: kornia.geometry.camera

Camera models that map 3D points to pixels and back. Points are in the camera frame, +X right, +Y down and +Z forward,
and pixels are ``(u, v)`` = (column, row) with integer pixel centres; :doc:`get-started/camera-conventions` compares
these conventions with other libraries. This page holds the building blocks of a camera model, which take image-plane
coordinates before the intrinsics (normalized ``z = 1`` coordinates for a perspective camera) and flat parameter
vectors: a projection to the image plane, then the intrinsics, with or without a fisheye distortion. For pixels from a
``3x3`` ``K`` in one call, see :doc:`perspective <geometry.camera.perspective>`; for cameras with extrinsics,
:doc:`pinhole <geometry.camera.pinhole>`; for a rectified stereo rig, :doc:`stereo camera <geometry.camera.stereo>`; for
OpenCV's Brown-Conrady distortion of pixel coordinates, :doc:`geometry.calibration`.

The Double Sphere model is a complete 3D projection rather than a distortion of normalized ``z = 1`` coordinates.
Its projection accepts camera-frame points and its inverse returns unit-length rays, including valid rays with negative
``z``. The equations and valid domains follow Usenko et al.,
`The Double Sphere Camera Model <https://arxiv.org/abs/1807.08957>`_.

Projections
-----------

Camera-frame points to the image plane and back, without intrinsics: :func:`project_points_z1` divides by ``z``, onto
the normalized ``z = 1`` plane (perspective), and :func:`project_points_orthographic` drops it.
:func:`dx_project_points_z1` returns the ``(..., 2, 3)`` Jacobian of the perspective projection, and
:func:`dx_project_points_orthographic` the derivative ``du/dx`` alone.

.. autofunction:: project_points_z1
.. autofunction:: unproject_points_z1
.. autofunction:: dx_project_points_z1

.. autofunction:: project_points_orthographic
.. autofunction:: unproject_points_orthographic
.. autofunction:: dx_project_points_orthographic

.. autofunction:: project_points_double_sphere
.. autofunction:: unproject_points_double_sphere

Distortion
----------

Normalized points to pixels and back with a flat parameter vector: the affine model ``[fx, fy, cx, cy]``, which
applies the intrinsics and no lens distortion, and the Kannala-Brandt fisheye model
``[fx, fy, cx, cy, k0, k1, k2, k3]``, each with its inverse and its Jacobian with respect to the point.

.. autofunction:: distort_points_affine
.. autofunction:: undistort_points_affine
.. autofunction:: dx_distort_points_affine

.. autofunction:: distort_points_kannala_brandt
.. autofunction:: undistort_points_kannala_brandt
.. autofunction:: dx_distort_points_kannala_brandt

.. toctree::
   :maxdepth: 1

   pinhole <geometry.camera.pinhole>
   perspective <geometry.camera.perspective>
   stereo camera <geometry.camera.stereo>
