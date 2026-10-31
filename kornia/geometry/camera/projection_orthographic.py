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

"""Orthographic projection of camera-frame points: drop z, append it back, and the derivative."""

# inspired by: https://github.com/farm-ng/sophus-rs/blob/main/src/sensor/ortho_camera.rs
import torch

from kornia.core.check import KORNIA_CHECK_SHAPE


def project_points_orthographic(points_in_camera: torch.Tensor) -> torch.Tensor:
    r"""Project camera-frame points orthographically by dropping z: (x, y, z) -> (x, y).

    ``uv = project_points_orthographic(points)`` takes camera-frame points ``(..., 3)`` and returns their
    ``(x, y)`` as ``(..., 2)``, in the units of the points: no division and no intrinsics. Map them to pixels with
    :func:`distort_points_affine`. For a perspective camera, use :func:`project_points_z1`, or
    :func:`~kornia.geometry.camera.perspective.project_points` for pixels from a ``3x3`` ``K``.

    .. math::
        \begin{bmatrix} u \\ v \end{bmatrix} =
        \begin{bmatrix} x \\ y \end{bmatrix}


    Args:
        points_in_camera: torch.Tensor representing the points to project with shape ``(..., 3)``.

    Returns:
        torch.Tensor representing the projected points with shape ``(..., 2)``.

    See Also:
        - :func:`unproject_points_orthographic`: the reverse, appending a ``z``.
        - :func:`dx_project_points_orthographic`: its derivative ``du/dx``.
        - :func:`project_points_z1`: the perspective counterpart, dividing by ``z``.
        - :func:`~kornia.geometry.camera.perspective.project_points`: perspective pixels from a ``3x3`` ``K``.

    Example:
        >>> points = torch.tensor([1., 2., 3.])
        >>> project_points_orthographic(points)
        tensor([1., 2.])

    """
    KORNIA_CHECK_SHAPE(points_in_camera, ["*", "3"])
    return points_in_camera[..., :2]


def unproject_points_orthographic(points_in_camera: torch.Tensor, extension: torch.Tensor) -> torch.Tensor:
    r"""Unproject orthographic points by appending a z value: (u, v) -> (u, v, z).

    ``points = unproject_points_orthographic(uv, z)`` takes points ``(..., 2)`` and one ``z`` per point and returns
    ``(u, v, z)`` as ``(..., 3)``: ``u`` and ``v`` are not scaled. To scale normalized coordinates by a depth, as a
    perspective camera does, use :func:`unproject_points_z1` instead.

    .. math::
        \begin{bmatrix} x \\ y \\ z \end{bmatrix} =
        \begin{bmatrix} u \\ v \\ w \end{bmatrix}

    Convention:
        - ``extension`` becomes the ``z`` component of the result; both a ``(N,)`` and a ``(N, 1)`` extension
          are accepted.
        - ``extension`` must have the points' leading dimensions exactly: it does not broadcast, unlike in
          :func:`unproject_points_z1`, so a ``(1,)`` extension for ``(N, 2)`` points raises ``RuntimeError``.
          Its trailing size is not checked: a same-rank ``(N, 2)`` extension for ``(N, 2)`` points returns
          ``(N, 4)`` (`#6466 <https://github.com/kornia/kornia/issues/6466>`_), so pass one value per point.

    Args:
        points_in_camera: torch.Tensor representing the points to unproject with shape ``(..., 2)``.
        extension: torch.Tensor representing the extension of the points to unproject with shape ``(..., 1)``
            or ``(...,)``.

    Returns:
        torch.Tensor representing the unprojected points with shape ``(..., 3)``.

    See Also:
        - :func:`project_points_orthographic`: the reverse, dropping ``z``.
        - :func:`unproject_points_z1`: the perspective counterpart, scaling ``x`` and ``y`` by the depth.

    Example:
        >>> points = torch.tensor([1., 2.])
        >>> extension = torch.tensor([3.])
        >>> unproject_points_orthographic(points, extension)
        tensor([1., 2., 3.])

    """
    KORNIA_CHECK_SHAPE(points_in_camera, ["*", "2"])

    if len(points_in_camera.shape) != len(extension.shape):
        extension = extension[..., None]

    return torch.cat([points_in_camera, extension], dim=-1)


def dx_project_points_orthographic(points_in_camera: torch.Tensor) -> torch.Tensor:
    r"""Return the derivative du/dx of the orthographic projection: ones, one per point.

    ``d = dx_project_points_orthographic(points)`` takes camera-frame points ``(..., 3)`` and returns the
    derivative of ``u`` with respect to ``x`` at each, which is always 1, as ``(..., 1)``. It is not the full
    Jacobian: that of the orthographic projection is the constant ``[[1, 0, 0], [0, 1, 0]]``, while
    :func:`dx_project_points_z1` returns the full ``(..., 2, 3)`` Jacobian of the perspective projection.

    .. math::
        \frac{\partial u}{\partial x} = 1

    Args:
        points_in_camera: torch.Tensor representing the points to project with shape ``(..., 3)``.

    Returns:
        torch.Tensor representing the derivative of the x projection with respect to the x coordinate, with shape
        ``(..., 1)``.

    See Also:
        - :func:`project_points_orthographic`: the projection this differentiates.
        - :func:`dx_project_points_z1`: the full ``(..., 2, 3)`` Jacobian of the perspective projection.

    Example:
        >>> points = torch.tensor([1., 2., 3.])
        >>> dx_project_points_orthographic(points)
        tensor([1.])

    """
    KORNIA_CHECK_SHAPE(points_in_camera, ["*", "3"])
    return torch.ones_like(points_in_camera[..., 0:1])
