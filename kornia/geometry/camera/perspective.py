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

"""Perspective projection of camera-frame points to pixels with a 3x3 intrinsics matrix, and back at given depths."""

import torch
import torch.nn.functional as F

from kornia.geometry.conversions import (
    convert_points_from_homogeneous,
    convert_points_to_homogeneous,
    denormalize_points_with_intrinsics,
    normalize_points_with_intrinsics,
)


def project_points(point_3d: torch.Tensor, camera_matrix: torch.Tensor) -> torch.Tensor:
    r"""Project camera-frame 3D points to pixels with a 3x3 intrinsics matrix K.

    ``uv = project_points(points, K)`` takes camera-frame points ``(*, 3)``, with at least two dimensions, and the
    intrinsics ``(*, 3, 3)``, and returns their pixels ``(fx x / z + cx, fy y / z + cy)`` as ``(*, 2)``. ``K``
    broadcasts against the points as in :func:`~kornia.geometry.conversions.denormalize_points_with_intrinsics`:
    a ``(B, 3, 3)`` ``K`` pairs camera ``b`` with ``points[b]`` of ``(B, 3)`` or ``(B, N, 3)`` points; for
    ``(B, H, W, 3)`` points pass ``K[:, None, None]``. For world points, project through the extrinsics with
    :meth:`~kornia.geometry.camera.pinhole.PinholeCamera.project`; for a lens with Brown-Conrady distortion, pass
    the result to :func:`~kornia.geometry.calibration.distort_points`.

    Convention:
        - ``point_3d`` is a **camera-frame** point and ``camera_matrix`` the ``(*, 3, 3)`` ``K``; the
          function takes no extrinsics, so it does not move between frames. Pixel centres are integers; see the
          Convention block on :class:`~kornia.geometry.camera.pinhole.PinholeCamera`.
        - the input must be at least rank 2: an unbatched ``(3,)`` point raises :class:`ValueError`.
        - the skew entry ``K[0, 1]`` is ignored, as in
          :func:`~kornia.geometry.conversions.denormalize_points_with_intrinsics`.

    .. warning::
        At ``abs(z) <= 1e-8`` the divide is skipped and ``K`` is applied to the undivided point, giving the
        finite placeholder ``fx x + cx``; the other projection entry points do not follow this rule yet:
        `#4267 <https://github.com/kornia/kornia/issues/4267>`_. Neither that placeholder nor a point behind the
        camera is flagged as invalid: `#4555 <https://github.com/kornia/kornia/issues/4555>`_.

    Args:
        point_3d: tensor containing the 3d points to be projected
            to the camera plane. The shape of the tensor can be ``(*, 3)``.
        camera_matrix: tensor containing the intrinsics camera
            matrix. The tensor shape must be ``(*, 3, 3)``.

    Returns:
        tensor of (u, v) pixel coordinates with shape ``(*, 2)``.

    See Also:
        - :func:`unproject_points`: the reverse, pixels at given depths to camera-frame points.
        - :meth:`~kornia.geometry.camera.pinhole.PinholeCamera.project`: world points, through ``(B, 4, 4)``
          intrinsics and extrinsics.
        - :func:`~kornia.geometry.camera.project_points_z1`: the divide alone, to the normalized ``z = 1`` plane
          without intrinsics, with a plain divide at ``z = 0``.
        - :func:`~kornia.geometry.camera.project_points_orthographic`: drops ``z`` instead of dividing by it.
        - :func:`~kornia.geometry.calibration.distort_points`: adds Brown-Conrady lens distortion to the pixels.

    Example:
        >>> _ = torch.manual_seed(0)
        >>> X = torch.rand(1, 3)
        >>> K = torch.eye(3)[None]
        >>> project_points(X, K)
        tensor([[5.6088, 8.6827]])

    """
    # projection eq. [u, v, w]' = K * [x y z 1]'
    # u = fx * X / Z + cx
    # v = fy * Y / Z + cy
    # project back using depth dividing in a safe way
    xy_coords: torch.Tensor = convert_points_from_homogeneous(point_3d)
    return denormalize_points_with_intrinsics(xy_coords, camera_matrix)


def unproject_points(
    point_2d: torch.Tensor, depth: torch.Tensor, camera_matrix: torch.Tensor, normalize: bool = False
) -> torch.Tensor:
    r"""Unproject pixels at given depths to camera-frame 3D points with a 3x3 matrix K.

    ``points = unproject_points(uv, depth, K)`` takes pixels ``(*, 2)``, with at least two dimensions, their depths
    ``(*, 1)`` and the intrinsics ``(*, 3, 3)``, and returns the camera-frame points
    ``((u - cx) d / fx, (v - cy) d / fy, d)`` as ``(*, 3)``. ``K`` broadcasts against the pixels as in
    :func:`~kornia.geometry.conversions.normalize_points_with_intrinsics`: a ``(B, 3, 3)`` ``K`` pairs camera ``b``
    with ``uv[b]`` of ``(B, 2)`` or ``(B, N, 2)`` pixels; for ``(B, H, W, 2)`` pixels pass ``K[:, None, None]``.
    For a whole depth map, use :func:`~kornia.geometry.depth.depth_to_3d_v2`; for world points,
    :meth:`~kornia.geometry.camera.pinhole.PinholeCamera.unproject`.

    Convention:
        - ``point_2d`` is a pixel coordinate and the result is in the **camera** frame: the function takes no
          extrinsics, so it cannot return world coordinates.
          :meth:`~kornia.geometry.camera.pinhole.PinholeCamera.unproject` is the world-frame counterpart.
          Pixel centres are integers; see the Convention block on
          :class:`~kornia.geometry.camera.pinhole.PinholeCamera`.
        - ``depth`` is the camera-frame ``z`` of the result. With ``normalize=True`` it is read as the
          Euclidean ray length instead, so the result has that norm and a smaller ``z``.
        - ``depth`` must be a tensor (else :class:`TypeError`) whose last dimension is 1 (else
          :class:`ValueError`); its leading dimensions broadcast against the pixels'.

    Args:
        point_2d: tensor containing the 2d points to be projected to
            the camera frame. The shape of the tensor can be ``(*, 2)``.
        depth: tensor containing the depth value of each 2d
            points. The tensor shape must be equal to point2d ``(*, 1)``.
        camera_matrix: tensor containing the intrinsics camera
            matrix. The tensor shape must be ``(*, 3, 3)``.
        normalize: whether to normalize the pointcloud. This
            must be set to `True` when the depth is represented as the Euclidean
            ray length from the camera position.

    Returns:
        tensor of (x, y, z) camera coordinates with shape ``(*, 3)``.

    See Also:
        - :func:`project_points`: the reverse, camera-frame points to pixels.
        - :meth:`~kornia.geometry.camera.pinhole.PinholeCamera.unproject`: world points, through ``(B, 4, 4)``
          intrinsics and extrinsics.
        - :func:`~kornia.geometry.depth.depth_to_3d_v2`: every pixel of a depth map, channels last.
        - :func:`~kornia.geometry.camera.unproject_points_z1`: from the normalized ``z = 1`` plane, without
          intrinsics.
        - :func:`~kornia.geometry.calibration.undistort_points`: removes Brown-Conrady lens distortion from the
          pixels, to call first.

    Example:
        >>> _ = torch.manual_seed(0)
        >>> x = torch.rand(1, 2)
        >>> depth = torch.ones(1, 1)
        >>> K = torch.eye(3)[None]
        >>> unproject_points(x, depth, K)
        tensor([[0.4963, 0.7682, 1.0000]])

    """
    if not isinstance(depth, torch.Tensor):
        raise TypeError(f"Input depth type is not a torch.Tensor. Got {type(depth)}")

    if not depth.shape[-1] == 1:
        raise ValueError(f"Input depth must be in the shape of (*, 1). Got {depth.shape}")

    xy: torch.Tensor = normalize_points_with_intrinsics(point_2d, camera_matrix)
    xyz: torch.Tensor = convert_points_to_homogeneous(xy)
    if normalize:
        xyz = F.normalize(xyz, dim=-1, p=2.0)

    return xyz * depth
