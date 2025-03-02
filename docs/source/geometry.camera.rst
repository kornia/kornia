kornia.geometry.camera
======================

.. meta::
   :name: description
   :content: "The kornia.geometry.camera module provides a variety of functions for handling camera projections and distortions. It includes support for projecting 3D points to a 2D image plane, both with perspective and orthographic projections, as well as distortion models like affine and Kannala-Brandt. This module enables robust camera calibration and 3D scene transformations in computer vision applications."

.. currentmodule:: kornia.geometry.camera

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
   :maxdepth: 2

   geometry.camera.pinhole
   geometry.camera.perspective
   geometry.camera.stereo
