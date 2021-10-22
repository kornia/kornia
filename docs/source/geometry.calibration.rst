kornia.geometry.calibration
===========================

.. currentmodule:: kornia.geometry.calibration

Module with useful functionalities for camera calibration.

The pinhole model is an ideal projection model that not considers lens distortion for the projection of a 3D point :math:`(X, Y, Z)` onto the image plane. To model the distortion of a projected 2D pixel point :math:`(u,v)` with the linear pinhole model, we need first to estimate the normalized 2D points coordinates :math:`(\bar{u}, \bar{v})`. For that, we can use the calibration matrix :math:`\mathbf{K}` with the following expression

.. math::
    \begin{align}
    \begin{bmatrix}
    \bar{u}\\
    \bar{v}\\
    1
    \end{bmatrix} = \mathbf{K}^{-1} \begin{bmatrix}
    u \\
    v \\
    1
    \end{bmatrix} \enspace,
    \end{align}

which is equivalent to directly using the internal parameters: focals :math:`f_u, f_v` and principal point :math:`(u_0, v_0)` to estimated the normalized coordinates

.. math::
    \begin{equation}
    \bar{u} = (u - u_0)/f_u \enspace, \\
    \bar{v} = (v - v_0)/f_v \enspace.
    \end{equation}

The normalized distorted point :math:`(\bar{u}_d, \bar{v}_d)` is given by

.. math::
    \begin{align}
    \begin{bmatrix}
    \bar{u}_d\\
    \bar{v}_d
    \end{bmatrix} = \dfrac{1+k_1r^2+k_2r^4+k_3r^6}{1+k_4r^2+k_5r^4+k_6r^6} \begin{bmatrix}
    \bar{u}\\
    \bar{v}
    \end{bmatrix} +
    \begin{bmatrix}
    2p_1\bar{u}\bar{v} + p_2(r^2 + 2\bar{u}^2) + s_1r^2 + s_2r^4\\
    2p_2\bar{u}\bar{v} + p_1(r^2 + 2\bar{v}^2) + s_3r^2 + s_4r^4
    \end{bmatrix} \enspace,
    \end{align}

where :math:`r = \bar{u}^2 + \bar{v}^2`. With this model we consider radial :math:`(k_1, k_2, k_3, k_4, k_4, k_6)`, tangential :math:`(p_1, p_2)`, and thin prism :math:`(s_1, s_2, s_3, s_4)` distortion. If we want to consider tilt distortion :math:`(\tau_x, \tau_y)`, we need an additional step where we estimate a point :math:`(\bar{u}'_d, \bar{v}'_d)`

.. math::
    \begin{align}
    \begin{bmatrix}
    \bar{u}'_d\\
    \bar{v}'_d\\
    1
    \end{bmatrix} = \begin{bmatrix}
    \mathbf{R}_{33}(\tau_x, \tau_y) & 0 & -\mathbf{R}_{13}(\tau_x, \tau_y)\\
    0 & \mathbf{R}_{33}(\tau_x, \tau_y) & -\mathbf{R}_{23}(\tau_x, \tau_y)\\
    0 & 0 & 1
    \end{bmatrix}
    \mathbf{R}(\tau_x, \tau_y) \begin{bmatrix}
    \bar{u}_d \\
    \bar{v}_d \\
    1
    \end{bmatrix} \enspace,
    \end{align}

where :math:`\mathbf{R}(\tau_x, \tau_y)` is a 3D rotation matrix defined by an :math:`X` and :math:`Y` rotation given by the angles :math:`\tau_x` and :math:`\tau_y`. Furthermore, :math:`\mathbf{R}_{ij}(\tau_x, \tau_y)` represent the :math:`i`-th row and :math:`j`-th column from :math:`\mathbf{R}(\tau_x, \tau_y)` matrix.

.. math::
    \begin{align}
    \mathbf{R}(\tau_x, \tau_y) =
    \begin{bmatrix}
    \cos \tau_y & 0 & -\sin \tau_y \\
    0 & 1 & 0 \\
    \sin \tau_y & 0 & \cos \tau_y
    \end{bmatrix}
    \begin{bmatrix}
    1 & 0 & 0 \\
    0 & \cos \tau_x & \sin \tau_x \\
    0 & -\sin \tau_x & \cos \tau_x
    \end{bmatrix}  \enspace.
    \end{align}


Finally, we just need to come back to the original (unnormalized) pixel space. For that we can use the intrinsic matrix

.. math::
    \begin{align}
    \begin{bmatrix}
    u_d\\
    v_d\\
    1
    \end{bmatrix} = \mathbf{K} \begin{bmatrix}
    \bar{u}'_d\\
    \bar{v}'_d\\
    1
    \end{bmatrix} \enspace,
    \end{align}


which is equivalent to

.. math::
    \begin{equation}
    u_d = f_u \bar{u}'_d + u_0 \enspace, \\
    v_d = f_v \bar{v}'_d + v_0 \enspace.
    \end{equation}

Undistortion
------------

To compensate for lens distortion a set of 2D points, i.e., to estimate the undistorted coordinates for a given set of distorted points, we need to inverse the previously explained distortion model. For the case of undistorting an image, instead of estimating the undistorted location for each pixel, we distort each pixel in the destination image (final undistorted image) to match them with the input image. We finally interpolate the intensity values at each pixel.

.. autofunction:: undistort_image

.. autofunction:: undistort_points

.. autofunction:: distort_points

.. autofunction:: tilt_projection

Perspective-n-Point (PnP)
-------------------------

.. autofunction:: solve_pnp_dlt
