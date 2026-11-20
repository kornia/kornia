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

r"""The Double Sphere camera model between normalized z = 1 points and pixels, its inverse and Jacobian.

Adapted from:
    Vladyslav Usenko, Nikolaus Demmel, and Daniel Cremers.
    "The Double Sphere Camera Model", 3DV 2018.
    https://vision.in.tum.de/research/vslam/double-sphere
    https://arxiv.org/abs/1807.08957
"""

from __future__ import annotations

import torch

from kornia.core.check import KORNIA_CHECK_SHAPE


def distort_points_double_sphere(
    projected_points_in_camera_z1_plane: torch.Tensor,
    params: torch.Tensor,
) -> torch.Tensor:
    r"""Map normalized :math:`z = 1` points to pixels with the Double Sphere camera model.

    ``uv = distort_points_double_sphere(xy, params)`` takes points on the normalized :math:`z = 1` plane ``(..., 2)``
    and the parameter vector ``params = [fx, fy, cx, cy, xi, alpha]`` of shape ``(..., 6)``, and returns their distorted
    pixel coordinates ``(..., 2)``:

    .. math::
        d_1 = \sqrt{x^2 + y^2 + 1}

    .. math::
        d_2 = \sqrt{x^2 + y^2 + (\xi d_1 + 1)^2}

    .. math::
        \text{denom} = \alpha d_2 + (1 - \alpha) (\xi d_1 + 1)

    .. math::
        u = f_x \frac{x}{\text{denom}} + c_x, \quad v = f_y \frac{y}{\text{denom}} + c_y

    Convention:
        - ``projected_points_in_camera_z1_plane`` is a point on the **normalized** :math:`z = 1` plane, not a
          pixel, and the result is in pixels (integer pixel centers; see
          :class:`~kornia.geometry.camera.pinhole.PinholeCamera`).
        - ``params`` is the flat vector ``[fx, fy, cx, cy, xi, alpha]``: the first four are camera intrinsics
          and the last two are the Double Sphere parameters (:math:`\xi` and :math:`\alpha`).
        - :func:`undistort_points_double_sphere` is the closed-form algebraic inverse map.
        - The leading dimensions of the points and of ``params`` broadcast together: per-camera ``params`` for
          ``(B, N, 2)`` points are ``(B, 1, 6)``.
        - The result has the promoted dtype of the points and ``params``; ``float16`` and ``bfloat16`` are
          computed in ``float32`` and rounded once to the promoted dtype.

    Args:
        projected_points_in_camera_z1_plane: Tensor representing points on the normalized :math:`z = 1` plane
            with shape ``(..., 2)``.
        params: Tensor representing the parameters of the Double Sphere camera model with shape ``(..., 6)``.

    Returns:
        Tensor representing distorted pixel coordinates with shape ``(..., 2)``.

    See Also:
        - :func:`undistort_points_double_sphere`: the closed-form inverse, pixels to normalized points.
        - :func:`dx_distort_points_double_sphere`: its ``(..., 2, 2)`` Jacobian with respect to the point.
        - :func:`~kornia.geometry.camera.distort_points_kannala_brandt`: the Kannala-Brandt fisheye model.
        - :class:`~kornia.sensors.camera.distortion_model.DoubleSphereTransform`: the same map on ``Vector2`` points.

    Example:
        >>> points = torch.tensor([0.0, 0.0])
        >>> params = torch.tensor([500.0, 500.0, 320.0, 240.0, -0.18, 0.59])
        >>> distort_points_double_sphere(points, params)
        tensor([320., 240.])

    """
    KORNIA_CHECK_SHAPE(projected_points_in_camera_z1_plane, ["*", "2"])
    KORNIA_CHECK_SHAPE(params, ["*", "6"])

    output_dtype = torch.promote_types(projected_points_in_camera_z1_plane.dtype, params.dtype)
    compute_dtype = output_dtype
    if compute_dtype in (torch.float16, torch.bfloat16):
        compute_dtype = torch.float32

    points = projected_points_in_camera_z1_plane.to(dtype=compute_dtype)
    p = params.to(dtype=compute_dtype)

    x = points[..., 0]
    y = points[..., 1]

    fx = p[..., 0]
    fy = p[..., 1]
    cx = p[..., 2]
    cy = p[..., 3]
    xi = p[..., 4]
    alpha = p[..., 5]

    r2 = x * x + y * y
    d1 = torch.sqrt(r2 + 1.0)
    z1 = xi * d1 + 1.0
    d2 = torch.sqrt(r2 + z1 * z1)
    denom = alpha * d2 + (1.0 - alpha) * z1

    u = fx * (x / denom) + cx
    v = fy * (y / denom) + cy

    out = torch.stack([u, v], dim=-1)
    return out.to(dtype=output_dtype)


def undistort_points_double_sphere(
    distorted_points_in_camera: torch.Tensor,
    params: torch.Tensor,
) -> torch.Tensor:
    r"""Map distorted pixels back to normalized :math:`z = 1` points using closed-form algebraic unprojection.

    ``xy = undistort_points_double_sphere(uv, params)`` takes pixel coordinates ``(..., 2)`` and the ``(..., 6)``
    vector of :func:`distort_points_double_sphere` and returns the normalized coordinates ``(..., 2)`` on the
    :math:`z = 1` plane in closed form:

    .. math::
        m_x = \frac{u - c_x}{f_x}, \quad m_y = \frac{v - c_y}{f_y}, \quad r^2 = m_x^2 + m_y^2

    .. math::
        m_z = \frac{1 - \alpha^2 r^2}{\alpha \sqrt{1 - (2\alpha - 1) r^2} + 1 - \alpha}

    .. math::
        \text{scale} = \frac{m_z \xi + \sqrt{m_z^2 + (1 - \xi^2) r^2}}{m_z^2 + r^2}

    .. math::
        x_c = \text{scale} \cdot m_x, \quad y_c = \text{scale} \cdot m_y, \quad z_c = \text{scale} \cdot m_z - \xi

    .. math::
        x_{z1} = \frac{x_c}{z_c}, \quad y_{z1} = \frac{y_c}{z_c}

    Convention:
        - ``distorted_points_in_camera`` is in pixels, and the result is on the **normalized** :math:`z = 1` plane.
        - Unlike polynomial fisheye models (e.g. Kannala-Brandt), this unprojection is **non-iterative** and
          computed analytically in closed form.
        - Radicals are clamped at zero for numerical stability and safe autograd gradients.
        - ``float16`` and ``bfloat16`` inputs are computed in ``float32`` and rounded once to the promoted dtype.

    Args:
        distorted_points_in_camera: Tensor representing pixel coordinates with shape ``(..., 2)``.
        params: Tensor representing the parameters of the Double Sphere camera model with shape ``(..., 6)``.

    Returns:
        Tensor representing normalized points on the :math:`z = 1` plane with shape ``(..., 2)``.

    See Also:
        - :func:`distort_points_double_sphere`: the forward map, normalized points to pixels.
        - :func:`dx_distort_points_double_sphere`: the Jacobian of the forward map.
        - :class:`~kornia.sensors.camera.distortion_model.DoubleSphereTransform`: the same inverse on ``Vector2``.

    Example:
        >>> uv = torch.tensor([320.0, 240.0])
        >>> params = torch.tensor([500.0, 500.0, 320.0, 240.0, -0.18, 0.59])
        >>> undistort_points_double_sphere(uv, params)
        tensor([0., 0.])

    """
    KORNIA_CHECK_SHAPE(distorted_points_in_camera, ["*", "2"])
    KORNIA_CHECK_SHAPE(params, ["*", "6"])

    output_dtype = torch.promote_types(distorted_points_in_camera.dtype, params.dtype)
    compute_dtype = output_dtype
    if compute_dtype in (torch.float16, torch.bfloat16):
        compute_dtype = torch.float32

    points = distorted_points_in_camera.to(dtype=compute_dtype)
    p = params.to(dtype=compute_dtype)

    u = points[..., 0]
    v = points[..., 1]

    fx = p[..., 0]
    fy = p[..., 1]
    cx = p[..., 2]
    cy = p[..., 3]
    xi = p[..., 4]
    alpha = p[..., 5]

    mx = (u - cx) / fx
    my = (v - cy) / fy
    r2 = mx * mx + my * my

    # Differentiate positive safe arguments so boundary samples with det <= 0 or rad2 <= 0
    # have finite autograd derivatives and do not contaminate gradients across PyTorch versions.
    det_arg = 1.0 - (2.0 * alpha - 1.0) * r2
    det_is_nonpos = det_arg <= 0
    safe_det = torch.where(det_is_nonpos, torch.ones_like(det_arg), det_arg)
    sqrt_det = torch.where(det_is_nonpos, torch.zeros_like(det_arg), safe_det.sqrt())

    denom_mz = alpha * sqrt_det + (1.0 - alpha)
    mz = (1.0 - alpha * alpha * r2) / denom_mz

    rad2_arg = mz * mz + (1.0 - xi * xi) * r2
    rad2_is_nonpos = rad2_arg <= 0
    safe_rad2 = torch.where(rad2_is_nonpos, torch.ones_like(rad2_arg), rad2_arg)
    sqrt_rad2 = torch.where(rad2_is_nonpos, torch.zeros_like(rad2_arg), safe_rad2.sqrt())

    denom_scale = mz * mz + r2
    scale = (mz * xi + sqrt_rad2) / denom_scale

    xc = scale * mx
    yc = scale * my
    zc = scale * mz - xi

    out = torch.stack([xc / zc, yc / zc], dim=-1)
    return out.to(dtype=output_dtype)


def dx_distort_points_double_sphere(
    projected_points_in_camera_z1_plane: torch.Tensor,
    params: torch.Tensor,
) -> torch.Tensor:
    r"""Compute the 2x2 Jacobian of the Double Sphere model with respect to the point.

    ``J = dx_distort_points_double_sphere(xy, params)`` takes normalized points ``(..., 2)`` and the ``(..., 6)``
    vector of :func:`distort_points_double_sphere` and returns the Jacobian of that map at each point as
    ``(..., 2, 2)``.

    Convention:
        - The result has shape ``(..., 2, 2)``: rows are the output components ``(u, v)``, columns are the input
          components ``(x, y)``.
        - ``float16`` and ``bfloat16`` inputs are computed in ``float32`` and rounded once to the promoted dtype.

    Args:
        projected_points_in_camera_z1_plane: Tensor representing points on the normalized :math:`z = 1` plane
            with shape ``(..., 2)``.
        params: Tensor representing the parameters of the Double Sphere camera model with shape ``(..., 6)``.

    Returns:
        Tensor representing the derivative of the distortion with respect to the point with shape ``(..., 2, 2)``.

    See Also:
        - :func:`distort_points_double_sphere`: the map this differentiates.
        - :func:`undistort_points_double_sphere`: the inverse map.

    Example:
        >>> points = torch.tensor([0.1, 0.2])
        >>> params = torch.tensor([500.0, 500.0, 320.0, 240.0, -0.18, 0.59])
        >>> dx_distort_points_double_sphere(points, params)
        tensor([[596.2293,  -7.5706],
                [ -7.5706, 584.8734]])

    """
    KORNIA_CHECK_SHAPE(projected_points_in_camera_z1_plane, ["*", "2"])
    KORNIA_CHECK_SHAPE(params, ["*", "6"])

    output_dtype = torch.promote_types(projected_points_in_camera_z1_plane.dtype, params.dtype)
    compute_dtype = output_dtype
    if compute_dtype in (torch.float16, torch.bfloat16):
        compute_dtype = torch.float32

    points = projected_points_in_camera_z1_plane.to(dtype=compute_dtype)
    p = params.to(dtype=compute_dtype)

    x = points[..., 0]
    y = points[..., 1]

    fx = p[..., 0]
    fy = p[..., 1]
    xi = p[..., 4]
    alpha = p[..., 5]

    r2 = x * x + y * y
    d1 = torch.sqrt(r2 + 1.0)
    z1 = xi * d1 + 1.0
    d2 = torch.sqrt(r2 + z1 * z1)
    denom = alpha * d2 + (1.0 - alpha) * z1

    d_d1_dr2 = 0.5 / d1
    d_d2_dr2 = 0.5 / d2 * (1.0 + 2.0 * z1 * xi * d_d1_dr2)
    d_denom_dr2 = alpha * d_d2_dr2 + (1.0 - alpha) * xi * d_d1_dr2

    inv_denom = 1.0 / denom
    inv_denom2 = inv_denom * inv_denom

    du_dx = fx * (inv_denom - 2.0 * x * x * d_denom_dr2 * inv_denom2)
    du_dy = -fx * 2.0 * x * y * d_denom_dr2 * inv_denom2
    dv_dx = -fy * 2.0 * x * y * d_denom_dr2 * inv_denom2
    dv_dy = fy * (inv_denom - 2.0 * y * y * d_denom_dr2 * inv_denom2)

    row0 = torch.stack([du_dx, du_dy], dim=-1)
    row1 = torch.stack([dv_dx, dv_dy], dim=-1)
    out = torch.stack([row0, row1], dim=-2)

    return out.to(dtype=output_dtype)
