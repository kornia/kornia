kornia.geometry.conversions
==================================

.. currentmodule:: kornia.geometry.conversions

Angles
------

.. autofunction:: rad2deg
.. autofunction:: deg2rad
.. autofunction:: pol2cart
.. autofunction:: cart2pol

Coordinates
-----------

.. autofunction:: convert_points_from_homogeneous
.. autofunction:: convert_points_to_homogeneous
.. autofunction:: convert_affinematrix_to_homography
.. autofunction:: denormalize_pixel_coordinates
.. autofunction:: normalize_pixel_coordinates
.. autofunction:: denormalize_pixel_coordinates3d
.. autofunction:: normalize_pixel_coordinates3d

Quaternion
----------

.. autofunction:: quaternion_to_angle_axis
.. autofunction:: quaternion_to_rotation_matrix
.. autofunction:: quaternion_log_to_exp
.. autofunction:: quaternion_exp_to_log
.. autofunction:: normalize_quaternion

Rotation Matrix
---------------

.. autofunction:: rotation_matrix_to_angle_axis
.. autofunction:: rotation_matrix_to_quaternion

Angle Axis
----------

.. autofunction:: angle_axis_to_quaternion
.. autofunction:: angle_axis_to_rotation_matrix
