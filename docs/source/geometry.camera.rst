kornia.geometry.camera
======================

.. meta::
   :description: The kornia.geometry.camera module provides a variety of functions for handling camera projections and distortions. It includes support for projecting 3D points to a 2D image plane, both with perspective and orthographic projections, as well as distortion models like affine and Kannala-Brandt. This module enables robust camera calibration and 3D scene transformations in computer vision applications.

.. currentmodule:: kornia.geometry.camera

.. note::
   :mod:`kornia.geometry.camera` is part of Kornia's :doc:`Stable core </get-started/stability>` and is not
   deprecated. For new code that needs the current stability guarantees, continue to use this API.
   :mod:`kornia.sensors.camera` is an experimental, ``Vector``-typed future direction for camera models, but it
   remains a separate API and is not a drop-in replacement. If this module is deprecated in the future, the Stable
   core policy requires at least one minor release of ``DeprecationWarning`` before an incompatible removal or
   replacement; no removal release is currently specified.

Projections
-----------

.. autofunction:: project_points_z1
.. autofunction:: unproject_points_z1
.. autofunction:: dx_project_points_z1

.. autofunction:: project_points_orthographic
.. autofunction:: unproject_points_orthographic
.. autofunction:: dx_project_points_orthographic

Distortion
----------

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
