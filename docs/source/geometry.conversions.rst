kornia.geometry.conversions
==================================

.. meta::
   :description: The kornia.geometry.conversions module provides a collection of functions for converting between various geometric representations, such as angles, coordinates, homographies, quaternions, rotation matrices, and Euler angles. This module includes utilities for transforming points between homogeneous and non-homogeneous coordinates, normalizing and denormalizing pixel coordinates, as well as handling pose transformations and quaternion operations for 3D geometry tasks.

.. currentmodule:: kornia.geometry.conversions

The functions on this page convert between geometric representations: Euclidean and homogeneous points, pixel
and normalized :math:`[-1, 1]` coordinates, pixel and normalized homographies, the forms of a 3D rotation, and pose
matrices in different camera conventions. Quaternions are ``(w, x, y, z)``, real part first, in the Hamilton
convention; axis-angle vectors and Euler angles are in radians, and :func:`angle_to_rotation_matrix` takes degrees.
The :math:`[-1, 1]` normalizations are corner-aligned unless an ``align_corners=False`` argument selects the
half-pixel convention, which every matrix and homography normalization offers except
:func:`denormalize_homography3d`. See :doc:`Conventions & Pitfalls </get-started/conventions>`
and :doc:`camera and world conventions </get-started/camera-conventions>`.

Angles
------

Convert angles between radians and degrees (:func:`rad2deg`, :func:`deg2rad`) and points between polar and cartesian
coordinates (:func:`pol2cart`, :func:`cart2pol`), and build a 2D rotation matrix from an angle in degrees
(:func:`angle_to_rotation_matrix`).

.. autofunction:: rad2deg
.. autofunction:: deg2rad
.. autofunction:: pol2cart
.. autofunction:: cart2pol
.. autofunction:: angle_to_rotation_matrix

Coordinates
-----------

Append or divide out the homogeneous coordinate of ``(*, N, D)`` points (:func:`convert_points_to_homogeneous`,
:func:`convert_points_from_homogeneous`), extend an affine matrix to a homography
(:func:`convert_affinematrix_to_homography`, :func:`convert_affinematrix_to_homography3d`), map pixel coordinates to
:math:`[-1, 1]` and back (:func:`normalize_pixel_coordinates`, :func:`denormalize_pixel_coordinates`, and in 3D
:func:`normalize_pixel_coordinates3d`, :func:`denormalize_pixel_coordinates3d`, which read ``(d, x, y)``), and map
pixels to normalized camera coordinates with the intrinsics and back (:func:`normalize_points_with_intrinsics`,
:func:`denormalize_points_with_intrinsics`).

.. autofunction:: convert_points_from_homogeneous
.. autofunction:: convert_points_to_homogeneous
.. autofunction:: convert_affinematrix_to_homography
.. autofunction:: convert_affinematrix_to_homography3d
.. autofunction:: denormalize_pixel_coordinates
.. autofunction:: normalize_pixel_coordinates
.. autofunction:: denormalize_pixel_coordinates3d
.. autofunction:: normalize_pixel_coordinates3d
.. autofunction:: normalize_points_with_intrinsics
.. autofunction:: denormalize_points_with_intrinsics


Homography
----------

Express a source→destination pixel homography in normalized :math:`[-1, 1]` coordinates and back
(:func:`normalize_homography`, :func:`denormalize_homography`, and in 3D :func:`normalize_homography3d`,
:func:`denormalize_homography3d`), or get the normalization matrix itself (:func:`normal_transform_pixel`,
:func:`normal_transform_pixel3d`).

.. autofunction:: normalize_homography
.. autofunction:: denormalize_homography
.. autofunction:: normalize_homography3d
.. autofunction:: denormalize_homography3d
.. autofunction:: normal_transform_pixel
.. autofunction:: normal_transform_pixel3d

Quaternion
----------

Convert ``(w, x, y, z)`` quaternions to axis-angle vectors and rotation matrices (:func:`quaternion_to_axis_angle`,
:func:`quaternion_to_rotation_matrix`), apply the exponential and logarithm maps, whose 3-vector is half the
axis-angle vector (:func:`quaternion_log_to_exp`, :func:`quaternion_exp_to_log`), and normalize to unit length
(:func:`normalize_quaternion`). For a quaternion object that composes and inverts, see
:class:`~kornia.geometry.quaternion.Quaternion`.

.. autofunction:: quaternion_to_axis_angle
.. autofunction:: quaternion_to_rotation_matrix
.. autofunction:: quaternion_log_to_exp
.. autofunction:: quaternion_exp_to_log
.. autofunction:: normalize_quaternion

Vector
------

Build the skew-symmetric cross-product matrix of a 3-vector (:func:`vector_to_skew_symmetric_matrix`).

.. autofunction:: vector_to_skew_symmetric_matrix

Rotation Matrix
---------------

Convert rotation matrices to axis-angle vectors (:func:`rotation_matrix_to_axis_angle`) and to quaternions
(:func:`rotation_matrix_to_quaternion`).

.. autofunction:: rotation_matrix_to_axis_angle
.. autofunction:: rotation_matrix_to_quaternion

Axis Angle
----------

Convert axis-angle vectors, the unit axis scaled by the angle in radians, to quaternions
(:func:`axis_angle_to_quaternion`) and to rotation matrices (:func:`axis_angle_to_rotation_matrix`).

.. autofunction:: axis_angle_to_quaternion
.. autofunction:: axis_angle_to_rotation_matrix

Euler Angles
------------

Convert roll, pitch and yaw in radians, for the rotation ``Rz(yaw) @ Ry(pitch) @ Rx(roll)``, to a quaternion and
back, with the quaternion given as four separate tensors (:func:`quaternion_from_euler`,
:func:`euler_from_quaternion`).

.. autofunction:: quaternion_from_euler
.. autofunction:: euler_from_quaternion

Pose
----------

Pack and split ``(B, 4, 4)`` pose matrices (:func:`Rt_to_matrix4x4`, :func:`matrix4x4_to_Rt`), invert a rigid
``(R, t)`` pose between camera-to-world and world-to-camera (:func:`camtoworld_to_worldtocam_Rt`,
:func:`worldtocam_to_camtoworld_Rt`), switch a camera-to-world pose between the OpenGL and OpenCV camera axes
(:func:`camtoworld_graphics_to_vision_4x4`, :func:`camtoworld_graphics_to_vision_Rt`,
:func:`camtoworld_vision_to_graphics_4x4`, :func:`camtoworld_vision_to_graphics_Rt`), and convert quaternion poses
between ARKit and Colmap (:func:`ARKitQTVecs_to_ColmapQTVecs`, :func:`ColmapQTVecs_to_ARKitQTVecs`).

.. autofunction:: Rt_to_matrix4x4
.. autofunction:: matrix4x4_to_Rt
.. autofunction:: worldtocam_to_camtoworld_Rt
.. autofunction:: camtoworld_to_worldtocam_Rt
.. autofunction:: camtoworld_graphics_to_vision_4x4
.. autofunction:: camtoworld_vision_to_graphics_4x4
.. autofunction:: camtoworld_graphics_to_vision_Rt
.. autofunction:: camtoworld_vision_to_graphics_Rt
.. autofunction:: ARKitQTVecs_to_ColmapQTVecs
.. autofunction:: ColmapQTVecs_to_ARKitQTVecs
