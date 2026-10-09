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

Use :func:`intrinsics_from_homographies` when corresponding points on a known metric plane
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
linear system. Each view's two residuals are multiplied by the square root of its weight,
so the squared algebraic error is weighted linearly. It solves the system by SVD, checks degeneracy and positive definiteness,
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
    from kornia.geometry.calibration import intrinsics_from_homographies
    from kornia.geometry.conversions import axis_angle_to_rotation_matrix
    from kornia.geometry.homography import find_homography_dlt
    from kornia.image import ImageSize

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
    # Zero weights remove unusable views, including NaN-filled homographies.
    weights = homographies.new_ones(batch, views)
    weights[:, -1] = 0
    homographies[:, -1] = float("nan")
    initial_k, valid = intrinsics_from_homographies(homographies, ImageSize(480, 640), weights)
    assert valid.all()
    torch.testing.assert_close(initial_k[0], known_k, atol=0.02, rtol=0)

This checks an ideal, distortion-free synthetic case, not real-camera accuracy. Noisy
observations and lens distortion can bias a linear initial estimate. Diverse tilted views
are needed; repeated views and fronto-parallel translations can be degenerate. Two suitable
positive-weight views suffice under the zero-skew model; more diverse views help with noise.
Weights have shape ``(B,V)``, allowing cameras to retain different numbers of usable views.
They must be finite, nonnegative, and match the homographies' dtype and device.

The result is ``(intrinsics, valid)``, with a boolean ``valid`` of shape ``(B,)``. A camera
with fewer than two retained views, invalid retained homographies, weak geometry, or a
non-positive-definite conic returns ``False`` and an identity matrix. The identity is a
placeholder and must not be used as a calibration. Other cameras in the batch keep their
own results. Shape, dtype, image-size and weight errors still raise. Invalid cameras and
excluded views have zero gradients; gradients of accepted cameras are local and require
distinct singular values. Eager weight-value checks may cause a compile graph break;
numerical calibration failures use tensor masks without Python data-dependent branches.

Choosing the degeneracy threshold
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The default ``degeneracy_rtol=0.01`` is independent of the working dtype. In the normalized,
weighted constraint system, both :math:`\sigma_4/\sigma_1` and
:math:`(\sigma_4-\sigma_5)/\sigma_1` must exceed it. The second check separates the solution
direction from the next singular direction. For exactly two views, adding a zero row to
the 4-by-5 system preserves its nullspace and makes its fifth singular vector available
to reduced SVD and autograd.

The following synthetic sweep uses the 8-by-6 grid and camera above, translation
``[0.02,-0.03,0.9]``, Gaussian pixel noise, and seeds 0 through 19. Six-view cases use
axis-angle vectors ``[r,0,0],[-r,0,0],[0,r,0],[0,-r,0],[r,r,0],[-r,r,0]``; random cases use
eight views with each axis-angle component uniform in ``[-0.45,0.45]`` radians. The error
is the upper median of the largest absolute error across ``fx,fy,cx,cy`` among accepted
cameras, in pixels. The relaxed threshold illustrates why checking near-singularity at
machine precision alone is insufficient.

.. list-table:: CPU float64 sweep (20 seeds per row)
   :header-rows: 1

   * - Views
     - Noise (px)
     - Accepted at 1e-12
     - Median error at 1e-12 (px)
     - Accepted at 0.01
   * - 6, 1 degree
     - 0.1
     - 18/20
     - 1284.0
     - 0/20
   * - 6, 3 degrees
     - 0.1
     - 20/20
     - 31.9
     - 0/20
   * - 6, 10 degrees
     - 0.1
     - 20/20
     - 5.4
     - 20/20
   * - 8, random
     - 0.1
     - 20/20
     - 1.9
     - 20/20
   * - 6, 1 degree
     - 0.5
     - 15/20
     - 3921.8
     - 0/20
   * - 6, 3 degrees
     - 0.5
     - 19/20
     - 410.6
     - 0/20
   * - 6, 10 degrees
     - 0.5
     - 20/20
     - 20.9
     - 20/20
   * - 8, random
     - 0.5
     - 20/20
     - 10.2
     - 20/20

To reproduce the sweep after the example above:

.. code-block:: python

    import math

    for sigma in (0.1, 0.5):
        for degrees in (1, 3, 10, None):
            errors, accepted = [], 0
            for seed in range(20):
                g = torch.Generator().manual_seed(seed)
                if degrees is None:
                    angles = (torch.rand(8, 3, generator=g, dtype=dtype) - 0.5) * 0.9
                else:
                    r = math.radians(degrees)
                    angles = torch.tensor(
                        [[r,0,0],[-r,0,0],[0,r,0],[0,-r,0],[r,r,0],[-r,r,0]], dtype=dtype
                    )
                rotations = axis_angle_to_rotation_matrix(angles)
                h_true = known_k @ torch.cat(
                    (rotations[:, :, :2], translation.expand(len(angles), -1, -1)), -1
                )
                ph = board_h @ h_true.transpose(-1, -2)
                pixels = ph[..., :2] / ph[..., 2:]
                pixels = pixels + sigma * torch.randn(pixels.shape, generator=g, dtype=dtype)
                h = find_homography_dlt(
                    board_xy[None].expand(len(angles), -1, -1), pixels, solver="svd"
                )[None]
                relaxed_k, relaxed_valid = intrinsics_from_homographies(h, (480, 640), degeneracy_rtol=1e-12)
                _, valid = intrinsics_from_homographies(h, (480, 640))
                accepted += int(valid[0])
                if relaxed_valid[0]:
                    errors.append((relaxed_k[0] - known_k).abs().max().item())
            errors.sort()
            print(degrees, sigma, len(errors), errors[len(errors) // 2], accepted)

This is a bounded geometry diagnostic, not a universal accuracy threshold. In an extended
sweep, 5- and 7-degree cases were also rejected at both noise levels; 15-degree cases were
retained. Even retained 10-degree views at 0.5 px noise had a 20.9 px median error. Different
intrinsics, image sizes, view weights, and noise can change the singular ratios. Pass the
true image size, examine reprojection residuals in the downstream calibration pipeline,
and change ``degeneracy_rtol`` explicitly if a different geometry tradeoff is appropriate.

Comparison with OpenCV initialization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``image_size`` is ``(height,width)`` or :class:`~kornia.image.ImageSize` with the same order.
Pass the true image size: centering at ``((width-1)/2,(height-1)/2)`` and scaling by
``2/max(height,width)`` sets the normalization of the algebraic error. It affects noisy
estimates and validity decisions, although exact well-conditioned constraints recover the
same intrinsics with different sizes. For seed 0 of the eight-view sweep, at 2 px noise,
changing only the supplied size from ``(480,640)`` to ``(4000,4000)`` changed estimated
``fx`` from 898.38 to 861.96 px. Both focal lengths and the principal point remain free;
the image center is not a fixed principal-point prior. In contrast,
`OpenCV initCameraMatrix2D
<https://docs.opencv.org/4.13.0/d9/d0c/group__calib3d.html>`_ takes planar correspondences
and ``(width,height)``, fixes the initial principal point at the image center, and defaults
to ``aspectRatio=1``. Its ``aspectRatio=0`` mode frees the focal-length ratio but still
does not implement this API's principal-point estimation. ``calibrateCamera`` additionally
estimates distortion and poses and performs nonlinear refinement. Neither API is an exact
output-equivalence reference for this initializer.

.. autofunction:: intrinsics_from_homographies
