kornia.geometry.camera
======================

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

.. toctree::
   :maxdepth: 2

   geometry.camera.pinhole
   geometry.camera.perspective
   geometry.camera.stereo
