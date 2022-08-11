kornia.geometry.conversions
==================================

.. currentmodule:: kornia.geometry.conversions

Angles
------

.. autofunction:: rad2deg
.. autofunction:: deg2rad
.. autofunction:: pol2cart
.. autofunction:: cart2pol
.. autofunction:: angle_to_rotation_matrix

Coordinates
-----------

.. autofunction:: convert_points_from_homogeneous
.. autofunction:: convert_points_to_homogeneous
.. autofunction:: convert_affinematrix_to_homography
.. autofunction:: denormalize_pixel_coordinates
.. autofunction:: normalize_pixel_coordinates
.. autofunction:: denormalize_pixel_coordinates3d
.. autofunction:: normalize_pixel_coordinates3d

Homography
----------

.. autofunction:: normalize_homography
.. autofunction:: denormalize_homography
.. autofunction:: normalize_homography3d

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

Pose (extrinsics)
----------

.. autofunction:: extrinsics_from_Rt
.. autofunction:: Rt_from_extrinsics
.. autofunction:: camtoworldRt_to_poseRt
.. autofunction:: poseRt_to_camtoworldRt
.. autofunction:: camtoworld_graphics_to_vision
.. autofunction:: camtoworld_vision_to_graphics
