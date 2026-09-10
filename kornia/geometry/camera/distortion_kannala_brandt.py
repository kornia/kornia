# LICENSE HEADER MANAGED BY add-license-header
#
# Copyright 2018 Kornia Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

# inspired by: shttps://github.com/farm-ng/sophus-rs/blob/main/src/sensor/kannala_brandt.rs

import torch

from kornia.core.check import KORNIA_CHECK_SHAPE
from kornia.geometry.camera.distortion_affine import distort_points_affine


def _distort_points_kannala_brandt_impl(
    projected_points_in_camera_z1_plane: torch.Tensor,
    params: torch.Tensor,
    radius_sq: torch.Tensor,
) -> torch.Tensor:
    # https://github.com/farm-ng/sophus-rs/blob/20f6cac68f17fe1ac41d0aa8a27489e2b886806f/
    # src/sensor/kannala_brandt.rs#L51-L67
    x = projected_points_in_camera_z1_plane[..., 0]
    y = projected_points_in_camera_z1_plane[..., 1]

    fx, fy = params[..., 0], params[..., 1]
    cx, cy = params[..., 2], params[..., 3]

    k0 = params[..., 4]
    k1 = params[..., 5]
    k2 = params[..., 6]
    k3 = params[..., 7]

    radius = radius_sq.sqrt()
    radius_inverse = 1.0 / radius
    theta = radius.atan2(torch.ones_like(radius))
    theta2 = theta**2
    theta4 = theta2**2
    theta6 = theta2 * theta4
    theta8 = theta4**2

    r_distorted = theta * (1.0 + k0 * theta2 + k1 * theta4 + k2 * theta6 + k3 * theta8)

    scaling = r_distorted * radius_inverse

    u = fx * scaling * x + cx
    v = fy * scaling * y + cy

    return torch.stack([u, v], dim=-1)


def distort_points_kannala_brandt(
    projected_points_in_camera_z1_plane: torch.Tensor, params: torch.Tensor
) -> torch.Tensor:
    r"""Distort points from the canonical z=1 plane into the camera frame using the Kannala-Brandt model.

    Convention:
        - ``projected_points_in_camera_z1_plane`` is a point on the **normalized** :math:`z = 1` plane, not a
          pixel, and the result is in pixels.
          Pixel centres lie at integer coordinates: the top-left centre is ``(0, 0)``.
        - ``params`` is the flat vector ``[fx, fy, cx, cy, k0, k1, k2, k3]``: the first four are the affine
          part that :func:`~kornia.geometry.camera.distort_points_affine` takes on its own, and ``k0`` to
          ``k3`` multiply :math:`\theta^2`, :math:`\theta^4`, :math:`\theta^6` and :math:`\theta^8` in the
          fish-eye polynomial.
        - :func:`undistort_points_kannala_brandt` is the inverse map.

    Args:
        projected_points_in_camera_z1_plane: torch.Tensor representing the points to distort with shape (..., 2).
        params: torch.Tensor representing the parameters of the Kannala-Brandt distortion model with shape (..., 8).

    Returns:
        torch.Tensor representing the distorted points with shape (..., 2).

    Example:
        >>> points = torch.tensor([319.5, 239.5])  # center of a 640x480 image
        >>> params = torch.tensor([1000.0, 1000.0, 320.0, 280.0, 0.1, 0.01, 0.001, 0.0001])
        >>> distort_points_kannala_brandt(points, params)
        tensor([1982.6832, 1526.3619])

    """
    KORNIA_CHECK_SHAPE(projected_points_in_camera_z1_plane, ["*", "2"])
    KORNIA_CHECK_SHAPE(params, ["*", "8"])

    x = projected_points_in_camera_z1_plane[..., 0]
    y = projected_points_in_camera_z1_plane[..., 1]

    radius_sq = x**2 + y**2

    distorted_points = torch.where(
        radius_sq[..., None] > 1e-8,
        _distort_points_kannala_brandt_impl(
            projected_points_in_camera_z1_plane,
            params,
            radius_sq,
        ),
        distort_points_affine(projected_points_in_camera_z1_plane, params[..., :4]),
    )

    return distorted_points


def undistort_points_kannala_brandt(distorted_points_in_camera: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
    r"""Undistort points from the camera frame into the canonical z=1 plane using the Kannala-Brandt model.

    Convention:
        - ``distorted_points_in_camera`` is a **pixel** coordinate and the result is a point on the normalized
          :math:`z = 1` plane; ``params`` is the same 8-element vector documented on
          :func:`distort_points_kannala_brandt`.
        - the inverse is a fixed number of Gauss-Newton steps rather than a closed form: the step count is not
          a parameter and there is no convergence test, so the round trip through
          :func:`distort_points_kannala_brandt` closes only to the accuracy that iteration has reached. The
          step count cannot be raised by a caller. In ``float32`` the residual reaches the rounding floor; in
          ``float64`` it stops at about ``1e-8`` on the normalized plane, because the final radial rescale
          divides by ``r + 1e-8`` rather than ``r`` and so scales every result by ``1 - 1e-8 / r``. That is the
          ``float64`` side of `#4308 <https://github.com/kornia/kornia/issues/4308>`_.
          :func:`~kornia.geometry.camera.undistort_points_affine` is the closed-form contrast.
        - three small constants guard the Newton start (``1e-16``), Newton denominator (``1e-12``), and final
          radial rescale (``1e-8``), so a point at the principal point comes back as the origin rather than
          ``nan`` -- as long as those constants are representable in the dtype of ``params``, which is the
          dtype the whole body runs in.

    .. warning::
        All three guard constants underflow to zero in ``float16``. At the principal point the unguarded Newton
        denominator is one, but the final ``1e-8`` rescale guard then underflows and returns ``nan`` instead of
        the origin. ``float32``, ``float64`` and ``bfloat16`` are unaffected. Tracked as
        `#4308 <https://github.com/kornia/kornia/issues/4308>`_.

    Args:
        distorted_points_in_camera: torch.Tensor representing the points to undistort with shape (..., 2).
        params: torch.Tensor representing the parameters of the Kannala-Brandt distortion model with shape (..., 8).

    Returns:
        torch.Tensor representing the undistorted points with shape (..., 2).

    Example:
        >>> points = torch.tensor([319.5, 239.5])  # center of a 640x480 image
        >>> params = torch.tensor([1000.0, 1000.0, 320.0, 280.0, 0.1, 0.01, 0.001, 0.0001])
        >>> undistort_points_kannala_brandt(points, params).shape
        torch.Size([2])

    """
    KORNIA_CHECK_SHAPE(distorted_points_in_camera, ["*", "2"])
    KORNIA_CHECK_SHAPE(params, ["*", "8"])

    iters = 10
    eps = 1e-8
    device = distorted_points_in_camera.device
    out_dtype = distorted_points_in_camera.dtype

    pts = distorted_points_in_camera.to(device=device, dtype=params.dtype)
    p = params.to(device=device, dtype=params.dtype)

    x = pts[..., 0]
    y = pts[..., 1]

    fx = p[..., 0]
    fy = p[..., 1]
    cx = p[..., 2]
    cy = p[..., 3]
    k0 = p[..., 4]
    k1 = p[..., 5]
    k2 = p[..., 6]
    k3 = p[..., 7]

    un = (x - cx) / fx
    vn = (y - cy) / fy

    rth2 = un * un + vn * vn
    rth = rth2.sqrt()

    th = rth.clamp(min=1e-16).sqrt()

    # gauss-newton
    for _ in range(iters):
        th2 = th * th
        inner = k0 + th2 * (k1 + th2 * (k2 + th2 * k3))
        thd = th * (1.0 + th2 * inner)
        d_thd = 1.0 + th2 * (3.0 * k0 + th2 * (5.0 * k1 + th2 * (7.0 * k2 + 9.0 * k3 * th2)))
        step = (thd - rth) / (d_thd + 1e-12)
        th = th - step

    radius_undistorted = th.tan()
    denom = rth + eps
    mag = radius_undistorted.abs() / denom
    undistorted = torch.stack([mag * un, mag * vn], dim=-1)

    return undistorted.to(device=device, dtype=out_dtype)


def dx_distort_points_kannala_brandt(
    projected_points_in_camera_z1_plane: torch.Tensor, params: torch.Tensor
) -> torch.Tensor:
    r"""Return the analytic Jacobian of the Kannala-Brandt distortion model.

    Convention:
        - the result has shape :math:`(..., 2, 2)` and is laid out like the Jacobian that
          :func:`~kornia.geometry.camera.dx_distort_points_affine` returns: rows are the output components
          ``(u, v)``, columns the input components ``(x, y)``.
        - the Jacobian matches :func:`distort_points_kannala_brandt` with respect to the input point.
          For squared radii less than or equal to ``1e-8``, the forward function uses its affine branch,
          so the Jacobian is ``diag(fx, fy)``.

    Args:
        projected_points_in_camera_z1_plane: torch.Tensor representing the points to distort with shape (..., 2).
        params: torch.Tensor representing the parameters of the Kannala-Brandt distortion model with shape (..., 8).

    Returns:
        torch.Tensor representing the derivative of the distortion with respect to the point
        with shape (..., 2, 2).

    Example:
        >>> points = torch.tensor([1., 2.])
        >>> params = torch.tensor([1000.0, 1000.0, 320.0, 280.0, 0.1, 0.01, 0.001, 0.0001])
        >>> dx_distort_points_kannala_brandt(points, params)
        tensor([[ 524.3779, -136.9029],
                [-136.9029,  319.0236]])

    """
    KORNIA_CHECK_SHAPE(projected_points_in_camera_z1_plane, ["*", "2"])
    KORNIA_CHECK_SHAPE(params, ["*", "8"])

    # Match normal PyTorch dtype promotion. Half-precision arithmetic is
    # evaluated in float32 and rounded once at the end for numerical stability.
    output_dtype = torch.promote_types(projected_points_in_camera_z1_plane.dtype, params.dtype)
    compute_dtype = output_dtype

    if compute_dtype in (torch.float16, torch.bfloat16):
        compute_dtype = torch.float32

    points = projected_points_in_camera_z1_plane.to(dtype=compute_dtype)
    params_work = params.to(dtype=compute_dtype)

    x = points[..., 0]
    y = points[..., 1]

    fx, fy = params_work[..., 0], params_work[..., 1]

    k0 = params_work[..., 4]
    k1 = params_work[..., 5]
    k2 = params_work[..., 6]
    k3 = params_work[..., 7]

    radius_sq = x * x + y * y

    # The forward distortion uses the affine model for very small radii.
    # Keep the nonlinear expression finite too because both branch tensors
    # are evaluated before torch.where selects the result.
    nonlinear_mask = radius_sq > 1e-8
    safe_radius_sq = torch.where(
        nonlinear_mask,
        radius_sq,
        torch.ones_like(radius_sq),
    )

    radius = safe_radius_sq.sqrt()
    theta = radius.atan2(torch.ones_like(radius))

    theta2 = theta * theta
    theta4 = theta2 * theta2
    theta6 = theta4 * theta2
    theta8 = theta4 * theta4

    polynomial = 1.0 + k0 * theta2 + k1 * theta4 + k2 * theta6 + k3 * theta8

    radius_distorted = theta * polynomial

    d_radius_distorted_d_theta = 1.0 + 3.0 * k0 * theta2 + 5.0 * k1 * theta4 + 7.0 * k2 * theta6 + 9.0 * k3 * theta8

    d_radius_distorted_d_radius = d_radius_distorted_d_theta / (1.0 + safe_radius_sq)

    scaling = radius_distorted / radius

    radial_term = (radius * d_radius_distorted_d_radius - radius_distorted) / (radius * radius * radius)

    nonlinear_jacobian = torch.stack(
        [
            torch.stack(
                [
                    fx * (scaling + x * x * radial_term),
                    fx * x * y * radial_term,
                ],
                dim=-1,
            ),
            torch.stack(
                [
                    fy * x * y * radial_term,
                    fy * (scaling + y * y * radial_term),
                ],
                dim=-1,
            ),
        ],
        dim=-2,
    )

    zero = torch.zeros_like(fx)

    affine_jacobian = torch.stack(
        [
            torch.stack([fx, zero], dim=-1),
            torch.stack([zero, fy], dim=-1),
        ],
        dim=-2,
    )

    jacobian = torch.where(
        nonlinear_mask[..., None, None],
        nonlinear_jacobian,
        affine_jacobian,
    )

    if output_dtype in (torch.float16, torch.bfloat16):
        jacobian = jacobian.to(dtype=output_dtype)

    return jacobian
