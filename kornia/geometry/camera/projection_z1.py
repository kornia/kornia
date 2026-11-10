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

"""Perspective projection of camera-frame points onto the normalized z = 1 plane, its inverse and its Jacobian."""

# inspired by: https://github.com/farm-ng/sophus-rs/blob/main/src/sensor/perspective_camera.rs
from __future__ import annotations

from typing import Optional

import torch

from kornia.core.check import KORNIA_CHECK_SHAPE


def project_points_z1(points_in_camera: torch.Tensor) -> torch.Tensor:
    r"""Divide camera-frame points by z onto the normalized z = 1 plane, without intrinsics.

    ``xy = project_points_z1(points)`` takes camera-frame points ``(..., 3)`` and returns ``(x / z, y / z)`` as
    ``(..., 2)``: normalized coordinates, not pixels. Map them to pixels with :func:`distort_points_affine`, or
    through a fisheye lens with :func:`distort_points_kannala_brandt`; for pixels from a ``3x3`` ``K`` in one call,
    use :func:`~kornia.geometry.camera.perspective.project_points` instead.

    .. math::

        \begin{bmatrix} u \\ v \\ w \end{bmatrix} =
        \begin{bmatrix} x \\ y \\ z \end{bmatrix} / z

    .. note::

        This function has a precondition that the points are in front of the camera, i.e. z > 0.
        If this is not the case, the points will be projected to the canonical plane, but the resulting
        points will be behind the camera and causing numerical issues for z == 0.

    Convention:
        - the input is a **camera-frame** point and the output its position on the canonical ``z = 1`` plane,
          a normalized coordinate rather than a pixel; applying ``K`` with
          :func:`~kornia.geometry.conversions.denormalize_points_with_intrinsics` gives the pixel.
        - the ``z > 0`` precondition above is not validated.

    .. warning::
        The divide is plain at every ``z``: at ``z = 0`` a component is infinite, or ``nan`` for a zero
        numerator. :func:`~kornia.geometry.camera.perspective.project_points` instead skips its divide at
        ``abs(z) <= 1e-8`` and returns finite pixels there, so the two disagree below that threshold:
        `#4267 <https://github.com/kornia/kornia/issues/4267>`_.

    Args:
        points_in_camera: torch.Tensor representing the points to project with shape ``(..., 3)``.

    Returns:
        torch.Tensor representing the projected points with shape ``(..., 2)``.

    See Also:
        - :func:`unproject_points_z1`: the reverse, scaling a normalized point by a depth.
        - :func:`dx_project_points_z1`: the ``(..., 2, 3)`` Jacobian of this projection.
        - :func:`project_points_orthographic`: drops ``z`` instead of dividing by it.
        - :func:`~kornia.geometry.camera.perspective.project_points`: pixels from a ``3x3`` ``K`` in one call,
          with the divide skipped at ``abs(z) <= 1e-8``.

    Example:
        >>> points = torch.tensor([1., 2., 3.])
        >>> project_points_z1(points)
        tensor([0.3333, 0.6667])

    """
    KORNIA_CHECK_SHAPE(points_in_camera, ["*", "3"])
    return points_in_camera[..., :2] / points_in_camera[..., 2:3]


def unproject_points_z1(
    points_in_cam_canonical: torch.Tensor, extension: Optional[torch.Tensor] = None
) -> torch.Tensor:
    r"""Scale normalized z = 1 points by a depth into camera-frame points, z = 1 by default.

    ``points = unproject_points_z1(xy, depth)`` takes normalized coordinates ``(..., 2)`` and their camera-frame
    depths ``d``, ``(...,)`` or ``(..., 1)``, and returns ``(x d, y d, d)`` as ``(..., 3)``; without a depth it
    appends ``z = 1``. For pixels and a ``3x3`` ``K``, use
    :func:`~kornia.geometry.camera.perspective.unproject_points` instead.

    .. math::
        \begin{bmatrix} x \\ y \\ z \end{bmatrix} =
        \begin{bmatrix} u \\ v \\ 1 \end{bmatrix} \cdot w

    Convention:
        - ``extension`` is the camera-frame ``z`` of the unprojected point: the canonical point is multiplied
          by it and it becomes the third component.
          :meth:`~kornia.sensors.camera.projection_model.Z1Projection.unproject` names the same argument
          ``depth``. Both ``(...,)`` and ``(..., 1)`` extensions are accepted, and their leading dimensions
          broadcast against the points'. An extension with one dimension fewer than the points is read as
          ``(...,)``, any other as ``(..., 1)``: for ``(2, 5, 2)`` points a ``(5, 1)`` extension is read as
          ``(...,)`` and raises, so pass ``(1, 5, 1)``. One depth scales both coordinates: any other trailing size
          raises ``ShapeError``.

    Args:
        points_in_cam_canonical: torch.Tensor representing the points to unproject with shape ``(..., 2)``.
        extension: torch.Tensor representing the extension (depth) of the points to unproject with shape
            ``(..., 1)`` or ``(...,)``, broadcastable against the points' leading dimensions. Defaults to unit
            depth.

    Returns:
        torch.Tensor representing the unprojected points with shape ``(..., 3)``.

    See Also:
        - :func:`project_points_z1`: the reverse, dividing by ``z``.
        - :func:`unproject_points_orthographic`: appends the depth without scaling ``x`` and ``y``.
        - :func:`~kornia.geometry.camera.perspective.unproject_points`: from pixels, with a ``3x3`` ``K``.

    Example:
        >>> points = torch.tensor([1., 2.])
        >>> extension = torch.tensor([3.])
        >>> unproject_points_z1(points, extension)
        tensor([3., 6., 3.])

    """
    KORNIA_CHECK_SHAPE(points_in_cam_canonical, ["*", "2"])

    if extension is None:
        # a ones column, not ``F.pad(..., value=1.0)``: see ``convert_points_to_homogeneous`` for MPS
        return torch.cat([points_in_cam_canonical, torch.ones_like(points_in_cam_canonical[..., :1])], dim=-1)
    if extension.ndim == points_in_cam_canonical.ndim - 1:
        extension = extension[..., None]  # (..., 1)
    KORNIA_CHECK_SHAPE(extension, ["*", "1"])
    xy = points_in_cam_canonical * extension
    # The depth is expanded to the product's leading shape, so a broadcast depth concatenates. That also keeps
    # TorchScript's CUDA fuser from compiling ``cat([points * w, w])`` into a group that returns an int32 tensor for
    # an empty batch once it has specialized on non-empty calls.
    return torch.cat([xy, extension.expand(xy[..., :1].shape)], dim=-1)


def dx_project_points_z1(points_in_camera: torch.Tensor) -> torch.Tensor:
    r"""Compute the 2x3 Jacobian of the z = 1 projection with respect to the camera-frame point.

    ``J = dx_project_points_z1(points)`` takes camera-frame points ``(..., 3)`` and returns the Jacobian of
    :func:`project_points_z1` at each, ``(..., 2, 3)``: the derivatives of ``(x / z, y / z)`` with respect to
    ``(x, y, z)``. At ``z = 0`` its nonzero entries are infinite, or ``nan`` where ``x`` or ``y`` is 0. Chain it
    with :func:`dx_distort_points_affine` or :func:`dx_distort_points_kannala_brandt` for the Jacobian of a whole
    camera model.

    .. math::
        \frac{\partial \pi}{\partial x} =
        \begin{bmatrix}
            \frac{1}{z} & 0 & -\frac{x}{z^2} \\
            0 & \frac{1}{z} & -\frac{y}{z^2}
        \end{bmatrix}

    .. note::
        This function has a precondition that the points are in front of the camera, i.e. z > 0.
        If this is not the case, the points will be projected to the canonical plane, but the resulting
        points will be behind the camera and causing numerical issues for z == 0.

    Convention:
        - the result is the full Jacobian of :func:`~kornia.geometry.camera.project_points_z1` with shape
          ``(..., 2, 3)``, laid out row-major in the output index: the ``u`` row then the ``v`` row, each
          holding the derivatives with respect to ``x``, ``y`` and ``z``.
        - the same ``z > 0`` precondition applies, and it is not validated either.

    Args:
        points_in_camera: torch.Tensor representing the points to project with shape ``(..., 3)``.

    Returns:
        torch.Tensor representing the Jacobian of the projected point :math:`(u, v)` with respect to
        :math:`(x, y, z)`, with shape ``(..., 2, 3)``.

    See Also:
        - :func:`project_points_z1`: the projection this differentiates.
        - :func:`dx_project_points_orthographic`: the orthographic counterpart, which returns ``du/dx`` alone.
        - :func:`dx_distort_points_affine`: the Jacobian of the next step of a camera model, the intrinsics.

    Example:
        >>> points = torch.tensor([1., 2., 3.])
        >>> dx_project_points_z1(points)
        tensor([[ 0.3333,  0.0000, -0.1111],
                [ 0.0000,  0.3333, -0.2222]])

    """
    KORNIA_CHECK_SHAPE(points_in_camera, ["*", "3"])

    x = points_in_camera[..., 0]
    y = points_in_camera[..., 1]
    z = points_in_camera[..., 2]

    z_inv = 1.0 / z
    z_sq = z_inv * z_inv
    zeros = torch.zeros_like(z_inv)
    return torch.stack(
        [
            torch.stack([z_inv, zeros, -x * z_sq], dim=-1),
            torch.stack([zeros, z_inv, -y * z_sq], dim=-1),
        ],
        dim=-2,
    )
