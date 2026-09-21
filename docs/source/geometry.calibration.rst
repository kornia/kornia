kornia.geometry.calibration
===========================

.. meta::
   :description: The kornia.geometry.calibration module provides essential functions for camera calibration, including lens distortion modeling, undistortion, and Perspective-n-Point (PnP) solutions. This module supports advanced camera calibration techniques for both pinhole and distorted models, aiding in accurate 3D point projection, distortion correction, and camera pose estimation.

.. currentmodule:: kornia.geometry.calibration

Module with useful functionalities for camera calibration.

The pinhole model is an ideal projection model that does not consider lens distortion when projecting a 3D point :math:`(X, Y, Z)` onto the image plane. To model the distortion of a 2D pixel point :math:`(u,v)` projected with the linear pinhole model, we first need to estimate the normalized 2D point coordinates :math:`(\bar{u}, \bar{v})`. For that, we can use the calibration matrix :math:`\mathbf{K}` with the following expression

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

which is equivalent to directly using the internal parameters, the focal lengths :math:`f_u, f_v` and the principal point :math:`(u_0, v_0)`, to estimate the normalized coordinates

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

where :math:`r^2 = \bar{u}^2 + \bar{v}^2`. With this model we consider radial :math:`(k_1, k_2, k_3, k_4, k_5, k_6)`, tangential :math:`(p_1, p_2)`, and thin prism :math:`(s_1, s_2, s_3, s_4)` distortion. If we also want to consider tilt distortion :math:`(\tau_x, \tau_y)`, we need an additional step where we estimate a point :math:`(\bar{u}'_d, \bar{v}'_d)`

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

where :math:`\mathbf{R}(\tau_x, \tau_y)` is a 3D rotation matrix defined by rotations about the :math:`X` and :math:`Y` axes by the angles :math:`\tau_x` and :math:`\tau_y`. Furthermore, :math:`\mathbf{R}_{ij}(\tau_x, \tau_y)` denotes the element in the :math:`i`-th row and :math:`j`-th column of the :math:`\mathbf{R}(\tau_x, \tau_y)` matrix.

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

Finally, we just need to go back to the original (unnormalized) pixel space. For that we can use the intrinsic matrix

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

To compensate a set of 2D points for lens distortion, i.e., to estimate the undistorted coordinates of a given set of distorted points, we need to invert the distortion model explained above. When undistorting an image, instead of estimating the undistorted location of each pixel, we distort each pixel of the destination image (the final undistorted image) to find its match in the input image, and then interpolate the intensity values at each pixel.

.. autofunction:: undistort_image

.. autofunction:: undistort_points

.. autofunction:: distort_points

.. autofunction:: tilt_projection

Perspective-n-Point (PnP)
-------------------------

.. autofunction:: solve_pnp_dlt

Planar intrinsic initialization
--------------------------------

Use :func:`init_camera_intrinsics_zhang` when corresponding points on a known metric plane
are available and a PyTorch pipeline needs an initial pinhole intrinsic matrix. It implements
the linear stage of Zhang's method :cite:`zhang2000calibration` with zero skew. It estimates
the two focal lengths and the principal point; target detection, distortion estimation,
poses, and nonlinear reprojection refinement are outside its scope.

The `author's expanded report
<https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/tr98-71.pdf>`_,
Sections 2.3 and 3.1, derives the planar constraints and linear solution. For homography
columns :math:`h_1,h_2` and :math:`B=K^{-\mathsf{T}}K^{-1}`, the constraints are

.. math::

    h_1^{\mathsf{T}} B h_2 = 0, \qquad
    h_1^{\mathsf{T}} B h_1 = h_2^{\mathsf{T}} B h_2.

This implementation imposes :math:`B_{12}=0` for zero skew, giving a five-column
linear system. It solves that system by SVD, checks degeneracy and positive definiteness,
and recovers :math:`K` through Cholesky factorization. The factorization and image-coordinate
preconditioning are implementation choices; the API does not implement every stage of the paper.

From planar correspondences to intrinsics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For well-conditioned correspondences, first estimate each plane-to-pixel homography with
:func:`~kornia.geometry.homography.find_homography_dlt`, explicitly selecting ``solver="svd"``, then
group the homographies by camera as ``(B,V,3,3)``. Both board axes must use the same length
unit, image coordinates are pixel ``(x,y)``, and each camera must retain the same intrinsics
across its views.

This self-contained synthetic example projects an 8-by-6 metric grid into six tilted views.
Only the resulting image observations and board coordinates enter the estimation step.
The known matrix is used to check the result, not supplied to the initializer.

.. code-block:: python

    import torch
    from kornia.geometry.calibration import init_camera_intrinsics_zhang
    from kornia.geometry.conversions import axis_angle_to_rotation_matrix
    from kornia.geometry.homography import find_homography_dlt

    dtype = torch.float64
    known_k = torch.tensor(
        [[800.0, 0.0, 312.0], [0.0, 820.0, 245.0], [0.0, 0.0, 1.0]], dtype=dtype
    )
    rotations = axis_angle_to_rotation_matrix(torch.tensor(
        [[0.3, -0.2, 0.1], [-0.3, 0.2, -0.1], [0.1, 0.4, 0.2],
         [-0.4, -0.3, 0.1], [0.4, 0.2, -0.2], [0.1, -0.4, 0.3]], dtype=dtype
    ))
    translation = torch.tensor([0.02, -0.03, 0.9], dtype=dtype)[None, :, None]
    h_true = known_k @ torch.cat((rotations[:, :, :2], translation.expand(6, -1, -1)), dim=-1)
    y, x = torch.meshgrid(torch.arange(6, dtype=dtype), torch.arange(8, dtype=dtype), indexing="ij")
    board_xy = torch.stack(((x - 3.5) * 0.03, (y - 2.5) * 0.03), dim=-1).reshape(-1, 2)
    board_h = torch.cat((board_xy, torch.ones_like(board_xy[:, :1])), dim=-1)
    pixels_h = board_h @ h_true.transpose(-1, -2)
    image_xy = (pixels_h[..., :2] / pixels_h[..., 2:])[None]  # (B=1, V=6, N=48, 2)

    # Estimation: replace image_xy with matching ideal-camera pixel observations.
    batch, views, count, _ = image_xy.shape
    homographies = find_homography_dlt(
        board_xy[None].expand(batch * views, -1, -1),
        image_xy.reshape(batch * views, count, 2),
        solver="svd",
    ).reshape(batch, views, 3, 3)
    initial_k = init_camera_intrinsics_zhang(homographies, (480, 640))
    torch.testing.assert_close(initial_k[0], known_k, atol=0.02, rtol=0)

This checks an ideal, distortion-free synthetic case, not real-camera accuracy. Noisy
observations and lens distortion can bias a linear initial estimate. Diverse tilted views
are needed; repeated views and fronto-parallel translations can be degenerate. One invalid
camera in a batch raises an error for the entire call. The current API requires at least
three views; the zero-skew problem can theoretically be solved from two suitable views.

Comparison with OpenCV initialization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``image_size`` is ``(height,width)`` and supplies a numerical preconditioner, not a
principal-point prior. Both focal lengths and the principal point remain free. In contrast,
`OpenCV initCameraMatrix2D
<https://docs.opencv.org/4.13.0/d9/d0c/group__calib3d.html>`_ takes planar correspondences
and ``(width,height)``, fixes the initial principal point at the image center, and defaults
to ``aspectRatio=1``. Its ``aspectRatio=0`` mode frees the focal-length ratio but still
does not implement this API's principal-point estimation. ``calibrateCamera`` additionally
estimates distortion and poses and performs nonlinear refinement. Neither API is an exact
output-equivalence reference for this initializer.

.. autofunction:: init_camera_intrinsics_zhang
