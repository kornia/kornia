:github_url: https://github.com/arraiy/torchgeometry
             
torchgeometry
=============

The *PyTorch Geometry* (TGM) package is a geometric computer vision library for `PyTorch <https://pytorch.org/>`_.

It consists of a set of routines and differentiable modules to solve generic geometry computer vision problems. At its core, the package uses *PyTorch* as its main backend both for efficiency and to take advantage of the reverse-mode auto-differentiation to define and compute the gradient of complex functions.

In this first version, we provide different functions designed mainly for computer vision applications, such as image and tensors warping modules which rely on the epipolar geometry theory. The roadmap will include adding more and more functionality so that developers in the short term can use the package for the purpose of optimizing their loss functions to solve geometry problems.

TGM v0.1.0 focuses on Image and tensor warping functions such as:

* Calibration
* Epipolar geometry
* Homography
* Depth


.. automodule:: torchgeometry


Geometric Image Transformations
-------------------------------

The functions in this section perform various geometrical transformations of 2D images.

.. autofunction:: warp_perspective
.. autofunction:: get_perspective_transform


Pinhole
--------

.. note::
    The pinhole model is represented in a single vector as follows:

    .. math::
        pinhole = (f_x, f_y, c_x, c_y, height, width, r_x, r_y, r_z, t_x, t_y, t_z)
 
    where:
        :math:`(r_x, r_y, r_z)` is the rotation vector in angle-axis convention.

        :math:`(t_x, t_y, t_z)` is the translation vector.

.. autofunction:: inverse_pose
.. autofunction:: pinhole_matrix
.. autofunction:: inverse_pinhole_matrix
.. autofunction:: scale_pinhole
.. autofunction:: homography_i_H_ref

.. autoclass:: InversePose
.. autoclass:: PinholeMatrix
.. autoclass:: InversePinholeMatrix
.. autoclass:: ScalePinhole
.. autoclass:: Homography_i_H_ref


Conversions
-----------

.. autofunction:: rad2deg
.. autofunction:: deg2rad
.. autofunction:: convert_points_from_homogeneous
.. autofunction:: convert_points_to_homogeneous
.. autofunction:: transform_points
.. autofunction:: angle_axis_to_rotation_matrix
.. autofunction:: rotation_matrix_to_angle_axis
.. autofunction:: rotation_matrix_to_quaternion
.. autofunction:: quaternion_to_angle_axis
.. autofunction:: rtvec_to_pose

.. autoclass:: RadToDeg
.. autoclass:: DegToRad
.. autoclass:: ConvertPointsFromHomogeneous
.. autoclass:: ConvertPointsToHomogeneous
.. autoclass:: TransformPoints
.. autoclass:: AngleAxisToRotationMatrix
.. autoclass:: RotationMatrixToAngleAxis
.. autoclass:: RotationMatrixToQuaternion
.. autoclass:: QuaternionToAngleAxis
.. autoclass:: RtvecToPose


Utilities
---------

.. autofunction:: inverse
.. autofunction:: tensor_to_image
.. autofunction:: image_to_tensor

.. autoclass:: Inverse


Warping
-------

.. autoclass:: HomographyWarper
    :members:

.. autoclass:: DepthWarper
    :members:

.. autofunction:: homography_warp
.. autofunction:: depth_warp
