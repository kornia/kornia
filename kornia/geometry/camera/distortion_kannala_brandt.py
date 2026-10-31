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

"""The Kannala-Brandt fisheye camera model between normalized z = 1 points and pixels, its inverse and Jacobian."""

import torch

from kornia.core.check import KORNIA_CHECK_SHAPE
from kornia.geometry.camera.distortion_affine import distort_points_affine


def _distort_points_kannala_brandt_impl(
    projected_points_in_camera_z1_plane: torch.Tensor,
    params: torch.Tensor,
    radius_sq: torch.Tensor,
    nonlinear_mask: torch.Tensor,
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

    # Keep the lanes the caller discards finite: torch.where still differentiates both branch tensors.
    # Their radius is moved to about 1 and their angle zeroed. The active lanes add 0 and multiply by 1,
    # so they are unchanged, and this avoids two torch.where calls (slow on MPS).
    radius = (radius_sq + ~nonlinear_mask).sqrt()
    # The reciprocal's backward squares its value, which can overflow float16.
    radius_inverse = (1.0 / radius.float()).to(radius.dtype) if radius.dtype == torch.float16 else 1.0 / radius
    theta = radius.atan2(torch.ones_like(radius)) * nonlinear_mask
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
    r"""Map normalized z = 1 points to fisheye pixels with the Kannala-Brandt model.

    ``uv = distort_points_kannala_brandt(xy, params)`` takes points on the normalized ``z = 1`` plane ``(..., 2)``
    and the vector ``params = [fx, fy, cx, cy, k0, k1, k2, k3]``, ``(..., 8)``, and returns their pixels
    ``(..., 2)``: the angle :math:`\theta = \arctan r` of each point's ray from the optical axis, with ``r`` its
    distance from the axis, becomes the radius
    :math:`\theta (1 + k_0 \theta^2 + k_1 \theta^4 + k_2 \theta^6 + k_3 \theta^8)` before the intrinsics of
    :func:`distort_points_affine`, as OpenCV's ``cv2.fisheye.distortPoints`` with zero skew. For OpenCV's
    Brown-Conrady coefficients on pixel coordinates, use :func:`~kornia.geometry.calibration.distort_points`.

    Convention:
        - ``projected_points_in_camera_z1_plane`` is a point on the **normalized** :math:`z = 1` plane, not a
          pixel, and the result is in pixels (integer pixel centres; see
          :class:`~kornia.geometry.camera.pinhole.PinholeCamera`).
        - ``params`` is the flat vector ``[fx, fy, cx, cy, k0, k1, k2, k3]``: the first four are the affine
          part that :func:`~kornia.geometry.camera.distort_points_affine` takes on its own, and ``k0`` to
          ``k3`` multiply :math:`\theta^2`, :math:`\theta^4`, :math:`\theta^6` and :math:`\theta^8` in the
          fish-eye polynomial.
        - :func:`undistort_points_kannala_brandt` is the inverse map.
        - a point with squared radius ``<= 1e-8`` takes the affine branch, ``(fx x + cx, fy y + cy)``.
        - the leading dimensions of the points and of ``params`` broadcast as in :func:`distort_points_affine`:
          per-camera ``params`` for ``(B, N, 2)`` points are ``(B, 1, 8)``.

    Args:
        projected_points_in_camera_z1_plane: torch.Tensor representing the points to distort with shape
            ``(..., 2)``.
        params: torch.Tensor representing the parameters of the Kannala-Brandt distortion model with shape
            ``(..., 8)``.

    Returns:
        torch.Tensor representing the distorted points with shape ``(..., 2)``.

    See Also:
        - :func:`undistort_points_kannala_brandt`: the inverse, pixels to normalized points.
        - :func:`dx_distort_points_kannala_brandt`: its ``(..., 2, 2)`` Jacobian with respect to the point.
        - :func:`distort_points_affine`: the intrinsics alone, with no distortion.
        - :func:`~kornia.geometry.calibration.distort_points`: OpenCV's Brown-Conrady distortion with a ``3x3``
          ``K``, from pixels to pixels.

    Example:
        >>> points = torch.tensor([0.1, -0.2])  # on the normalized z = 1 plane
        >>> params = torch.tensor([1000.0, 1000.0, 320.0, 280.0, 0.1, 0.01, 0.001, 0.0001])
        >>> distort_points_kannala_brandt(points, params)
        tensor([418.8600,  82.2799])

    """
    KORNIA_CHECK_SHAPE(projected_points_in_camera_z1_plane, ["*", "2"])
    KORNIA_CHECK_SHAPE(params, ["*", "8"])

    x = projected_points_in_camera_z1_plane[..., 0]
    y = projected_points_in_camera_z1_plane[..., 1]

    radius_sq = x**2 + y**2
    nonlinear_mask = radius_sq > 1e-8

    return torch.where(
        nonlinear_mask[..., None],
        _distort_points_kannala_brandt_impl(
            projected_points_in_camera_z1_plane,
            params,
            radius_sq,
            nonlinear_mask,
        ),
        distort_points_affine(projected_points_in_camera_z1_plane, params[..., :4]),
    )


def undistort_points_kannala_brandt(distorted_points_in_camera: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
    r"""Map fisheye pixels back to normalized z = 1 points by inverting the Kannala-Brandt model.

    ``xy = undistort_points_kannala_brandt(uv, params)`` takes pixels ``(..., 2)`` and the ``(..., 8)`` vector of
    :func:`distort_points_kannala_brandt` and returns the normalized points ``(..., 2)`` that it maps to them,
    solving for the ray angle iteratively. The result matches OpenCV's ``cv2.fisheye.undistortPoints`` where the
    distorted angle, the distance of the normalized distorted point from the axis, is below :math:`\pi / 2`, the
    polynomial increases up to the solution, and the solved ray angle is below :math:`\pi / 2` too. Where the
    first two fail, OpenCV clamps the distorted angle to :math:`\pi / 2` or returns ``(-1e6, -1e6)`` for a point
    it cannot invert, while this function returns its last iterate, unflagged. Where only the solved angle is past
    :math:`\pi / 2`, the ray points behind the camera: OpenCV's signed tangent puts the point on the far side of
    the centre, and this function's absolute tangent keeps it on the side of the pixel. For
    OpenCV's Brown-Conrady coefficients on pixel coordinates, use
    :func:`~kornia.geometry.calibration.undistort_points`.

    Convention:
        - ``distorted_points_in_camera`` is a **pixel** coordinate and the result is a point on the normalized
          :math:`z = 1` plane; ``params`` is the same 8-element vector documented on
          :func:`distort_points_kannala_brandt`.
        - the inverse is a fixed number of Gauss-Newton steps (10) with no convergence test and no
          caller-visible step count, so the round trip through :func:`distort_points_kannala_brandt` closes only
          to the accuracy that iteration reaches.
        - a zero distorted radius maps to the origin.
        - the iteration runs in the dtype of ``params`` and the result takes the dtype of the points.

    Args:
        distorted_points_in_camera: torch.Tensor representing the points to undistort with shape ``(..., 2)``.
        params: torch.Tensor representing the parameters of the Kannala-Brandt distortion model with shape
            ``(..., 8)``.

    Returns:
        torch.Tensor representing the undistorted points with shape ``(..., 2)``.

    See Also:
        - :func:`distort_points_kannala_brandt`: the forward map, normalized points to pixels.
        - :func:`dx_distort_points_kannala_brandt`: the Jacobian of the forward map.
        - :func:`undistort_points_affine`: the closed-form inverse of the intrinsics alone.
        - :func:`~kornia.geometry.calibration.undistort_points`: removes OpenCV's Brown-Conrady distortion, from
          pixels to pixels.

    Example:
        >>> points = torch.tensor([319.5, 239.5])  # center of a 640x480 image
        >>> params = torch.tensor([1000.0, 1000.0, 320.0, 280.0, 0.1, 0.01, 0.001, 0.0001])
        >>> undistort_points_kannala_brandt(points, params).shape
        torch.Size([2])

    """
    KORNIA_CHECK_SHAPE(distorted_points_in_camera, ["*", "2"])
    KORNIA_CHECK_SHAPE(params, ["*", "8"])

    iters = 10
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

    nonzero_radius = (un != 0) | (vn != 0)
    safe_un = torch.where(nonzero_radius, un, torch.ones_like(un))
    safe_vn = torch.where(nonzero_radius, vn, torch.zeros_like(vn))
    radius_dtype = safe_un.dtype
    if radius_dtype == torch.float16:
        safe_un = safe_un.float()
        safe_vn = safe_vn.float()
    rth = (safe_un * safe_un + safe_vn * safe_vn).sqrt().to(radius_dtype)
    rth = torch.where(nonzero_radius, rth, torch.zeros_like(rth))

    safe_rth = torch.where(nonzero_radius, rth, torch.ones_like(rth))
    th = safe_rth.sqrt()
    th = torch.where(nonzero_radius, th, torch.zeros_like(th))

    # gauss-newton
    for _ in range(iters):
        th2 = th * th
        inner = k0 + th2 * (k1 + th2 * (k2 + th2 * k3))
        thd = th * (1.0 + th2 * inner)
        d_thd = 1.0 + th2 * (3.0 * k0 + th2 * (5.0 * k1 + th2 * (7.0 * k2 + 9.0 * k3 * th2)))
        step = (thd - rth) / (d_thd + 1e-12)
        th = th - step

    radius_undistorted = th.tan()
    denom = torch.where(nonzero_radius, rth, torch.ones_like(rth))
    mag = radius_undistorted.abs() / denom
    mag = torch.where(nonzero_radius, mag, torch.ones_like(mag))
    undistorted = torch.stack([mag * un, mag * vn], dim=-1)

    return undistorted.to(device=device, dtype=out_dtype)


def dx_distort_points_kannala_brandt(
    projected_points_in_camera_z1_plane: torch.Tensor, params: torch.Tensor
) -> torch.Tensor:
    r"""Compute the 2x2 Jacobian of the Kannala-Brandt model with respect to the point.

    ``J = dx_distort_points_kannala_brandt(xy, params)`` takes normalized points ``(..., 2)`` and the ``(..., 8)``
    vector of :func:`distort_points_kannala_brandt` and returns the Jacobian of that map at each point as
    ``(..., 2, 2)``. Chain it after :func:`dx_project_points_z1` for the Jacobian of a fisheye camera model.

    Convention:
        - the result has shape ``(..., 2, 2)`` and is laid out like the Jacobian that
          :func:`~kornia.geometry.camera.dx_distort_points_affine` returns: rows are the output components
          ``(u, v)``, columns the input components ``(x, y)``.
        - for squared radii ``<= 1e-8`` the forward function uses its affine branch, so the Jacobian there is
          ``diag(fx, fy)``.
        - float16 and bfloat16 inputs are computed in float32 and rounded once to the promoted input dtype.

    Args:
        projected_points_in_camera_z1_plane: torch.Tensor representing the points to distort with shape
            ``(..., 2)``.
        params: torch.Tensor representing the parameters of the Kannala-Brandt distortion model with shape
            ``(..., 8)``.

    Returns:
        torch.Tensor representing the derivative of the distortion with respect to the point
        with shape ``(..., 2, 2)``.

    See Also:
        - :func:`distort_points_kannala_brandt`: the map this differentiates.
        - :func:`undistort_points_kannala_brandt`: the inverse map.
        - :func:`dx_distort_points_affine`: the Jacobian of the intrinsics alone, ``diag(fx, fy)``.
        - :func:`dx_project_points_z1`: the Jacobian of the projection step before it.

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
