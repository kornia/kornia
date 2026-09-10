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

"""nn.Module containing operators to work on RGB-Depth images."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from kornia.core.check import KORNIA_CHECK, KORNIA_CHECK_IS_TENSOR, KORNIA_CHECK_SHAPE
from kornia.filters.sobel import spatial_gradient
from kornia.geometry.grid import create_meshgrid

from .camera import PinholeCamera, cam2pixel, pixel2cam, project_points, unproject_points
from .conversions import normalize_pixel_coordinates, normalize_points_with_intrinsics
from .linalg import convert_points_to_homogeneous, transform_points

"""nn.Module containing operators to work on RGB-Depth images."""

__all__ = [
    "DepthWarper",
    "depth_from_disparity",
    "depth_from_plane_equation",
    "depth_to_3d",
    "depth_to_3d_v2",
    "depth_to_normals",
    "depth_warp",
    "unproject_meshgrid",
    "warp_frame_depth",
]


def unproject_meshgrid(
    height: int,
    width: int,
    camera_matrix: torch.Tensor,
    normalize_points: bool = False,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """Compute a 3d point per pixel given its depth value and the camera intrinsics.

    .. tip::

        This function should be used in conjunction with :py:func:`kornia.geometry.depth.depth_to_3d_v2` to cache
        the meshgrid computation when warping multiple frames with the same camera intrinsics.

    Convention:
        - the result is the ray through each pixel at depth 1, in the :math:`(B, H, W, 3)` layout, so
          multiplying it by a :math:`(B, H, W)` depth map reproduces
          :func:`~kornia.geometry.depth.depth_to_3d_v2` exactly -- which is what makes it usable as that
          function's ``xyz_grid`` cache.
        - the pixels are the integer pixel centres that :func:`~kornia.geometry.grid.create_meshgrid`
          enumerates, described in the Convention block on
          :class:`~kornia.geometry.camera.pinhole.PinholeCamera`: pixel ``(0, 0)`` is centred at ``(0, 0)``.
          ``camera_matrix`` is the :math:`(3, 3)` ``[[fx, 0, cx], [0, fy, cy], [0, 0, 1]]`` of that grid and
          there are no extrinsics, so the rays are in the **camera** frame.
        - ``camera_matrix`` is batched: the supported form is :math:`(B, 3, 3)`, which returns
          :math:`(B, H, W, 3)`. Extra leading dimensions are not part of the contract: today a singleton one
          passes through by accident -- a :math:`(2, 1, 3, 3)` intrinsics returns :math:`(2, 1, H, W, 3)` --
          and non-singleton ones either raise ``RuntimeError`` or broadcast against the pixel axes (an extra
          axis matching ``W`` applies different intrinsics to each column); both are the subject of the
          warning below. An unbatched :math:`(3, 3)` raises ``ShapeError`` (below).
        - ``normalize_points=True`` returns the unit ray instead of the ray whose ``z`` is 1, which is the form
          :func:`~kornia.geometry.depth.depth_to_3d_v2` needs when its depth is a Euclidean ray length rather
          than a camera-frame ``z``.

    .. warning::
        The shape guard is written ``["*", "3", "3"]``, so a bare :math:`(3, 3)` ``camera_matrix`` passes it
        and then raises a ``ShapeError`` further into the body, whose message describes a shape the caller
        never passed rather than the one it did. The same guard admits non-singleton extra leading dimensions,
        which then broadcast against the pixel axes instead of being rejected (above). Tracked as
        `#4271 <https://github.com/kornia/kornia/issues/4271>`_.

    Args:
        height: height of image.
        width: width of image.
        camera_matrix: tensor containing the camera intrinsics with shape :math:`(B, 3, 3)`.
        normalize_points: whether to normalize the pointcloud. This must be set to `True` when the depth is
          represented as the Euclidean ray length from the camera position.
        device: device to place the result on.
        dtype: dtype of the result.

    Return:
        tensor with a 3d point per pixel, with shape :math:`(B, H, W, 3)`.

    """
    KORNIA_CHECK_SHAPE(camera_matrix, ["*", "3", "3"])

    # create base coordinates grid. ``create_meshgrid`` returns ``(1, H, W, 2)``; drop only that leading
    # batch axis. A bare ``squeeze()`` would also drop ``H`` or ``W`` whenever either is 1, and the grid
    # would then broadcast across a phantom axis instead of keeping the documented ``(*, H, W, 3)`` shape.
    points_uv: torch.Tensor = create_meshgrid(
        height, width, normalized_coordinates=False, device=device, dtype=dtype
    ).squeeze(0)  # HxWx2

    # project pixels to camera frame
    camera_matrix_tmp: torch.Tensor = camera_matrix[:, None, None]  # Bx1x1x3x3

    points_xy = normalize_points_with_intrinsics(points_uv, camera_matrix_tmp)  # HxWx2

    # unproject pixels to camera frame
    points_xyz = convert_points_to_homogeneous(points_xy)  # HxWx3

    if normalize_points:
        points_xyz = F.normalize(points_xyz, dim=-1, p=2)

    return points_xyz


def depth_to_3d_v2(
    depth: torch.Tensor,
    camera_matrix: torch.Tensor,
    normalize_points: bool = False,
    xyz_grid: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    # NOTE: when this replaces the `depth_to_3d` behaviour, a deprecated function should be added here, instead
    # of just replace the other function.
    """Compute a 3d point per pixel given its depth value and the camera intrinsics.

    .. note::

        This is an alternative implementation of :py:func:`kornia.geometry.depth.depth_to_3d`
        that does not require the creation of a meshgrid.

    Convention:
        - ``depth`` is the camera-frame ``z`` of each pixel and the result is the camera-frame point
          ``((u - cx) z / fx, (v - cy) z / fy, z)``, laid out channels-**last** as :math:`(*, H, W, 3)`.
          :func:`~kornia.geometry.depth.depth_to_3d` computes the same points in the :math:`(B, 3, H, W)`
          layout, and the two are equal after ``permute(0, 2, 3, 1)``.
        - ``u`` and ``v`` are the column and the row of the integer-centre pixel grid that
          :func:`~kornia.geometry.depth.unproject_meshgrid` builds, described in the Convention block on
          :class:`~kornia.geometry.camera.pinhole.PinholeCamera`. There are no extrinsics, so the points are in
          the **camera** frame.
        - ``camera_matrix`` needs a leading batch dimension: a bare :math:`(3, 3)` passes this function's own
          guard and, when ``xyz_grid`` is not given, is then rejected inside
          :func:`~kornia.geometry.depth.unproject_meshgrid`
          (`#4271 <https://github.com/kornia/kornia/issues/4271>`_). When ``xyz_grid`` is given,
          ``camera_matrix`` is never read, so any matrix that passes the ``(*, 3, 3)`` guard -- a bare
          :math:`(3, 3)` included -- is silently accepted.
        - ``normalize_points=True`` reads ``depth`` as the Euclidean ray length from the camera centre instead
          of as ``z``, so the returned point has that norm rather than that ``z``.
        - passing ``xyz_grid`` skips the grid construction and uses the given rays instead; the two forms give
          the same result when ``xyz_grid`` is what
          :func:`~kornia.geometry.depth.unproject_meshgrid` returns for the same camera.

    Args:
        depth: image tensor containing a depth value per pixel with shape :math:`(*, H, W)`.
        camera_matrix: tensor containing the camera intrinsics with shape :math:`(*, 3, 3)`.
        normalize_points: whether to normalise the pointcloud. This must be set to `True` when the depth is
          represented as the Euclidean ray length from the camera position.
        xyz_grid: explicit xyz point values.

    Return:
        tensor with a 3d point per pixel of the same resolution as the input, :math:`(*, H, W, 3)`, whose
        leading dimensions are the broadcast of ``depth``'s and ``camera_matrix``'s: the Example broadcasts a
        :math:`(4, 4)` depth against a :math:`(2, 3, 3)` camera to :math:`(2, 4, 4, 3)`, and a
        :math:`(B, T, H, W)` depth with a :math:`(B, 1, 3, 3)` camera returns :math:`(B, T, H, W, 3)`.

    Example:
        >>> depth = torch.rand(4, 4)
        >>> K = torch.eye(3).repeat(2,1,1)
        >>> depth_to_3d_v2(depth, K).shape
        torch.Size([2, 4, 4, 3])

    """
    KORNIA_CHECK_SHAPE(depth, ["*", "H", "W"])
    KORNIA_CHECK_SHAPE(camera_matrix, ["*", "3", "3"])

    # create base grid if not provided
    height, width = depth.shape[-2:]
    points_xyz: torch.Tensor = (
        xyz_grid
        if xyz_grid is not None
        else unproject_meshgrid(height, width, camera_matrix, normalize_points, depth.device, depth.dtype)
    )

    KORNIA_CHECK_SHAPE(points_xyz, ["*", "H", "W", "3"])

    return points_xyz * depth[..., None]  # HxWx3


def depth_to_3d(depth: torch.Tensor, camera_matrix: torch.Tensor, normalize_points: bool = False) -> torch.Tensor:
    """Compute a 3d point per pixel given its depth value and the camera intrinsics.

    .. note::

        :py:func:`kornia.geometry.depth.depth_to_3d_v2` computes the same points without building a meshgrid,
        in the :math:`(B, H, W, 3)` layout, and is the newer of the two. Which of them survives,
        and on what deprecation path, is a coordinated decision that has not been taken: the two are kept side
        by side as they are and this function emits no ``DeprecationWarning``.

    Convention:
        - ``depth`` is the camera-frame ``z`` of each pixel and the result is the camera-frame point
          ``((u - cx) z / fx, (v - cy) z / fy, z)``, laid out channels-**first** as :math:`(B, 3, H, W)`.
          :func:`~kornia.geometry.depth.depth_to_3d_v2` computes the same points in the :math:`(B, H, W, 3)`
          layout, and the two are equal after ``permute(0, 2, 3, 1)``.
        - ``u`` and ``v`` are the column and the row of the integer pixel centres that
          :func:`~kornia.geometry.grid.create_meshgrid` enumerates, described in the Convention block on
          :class:`~kornia.geometry.camera.pinhole.PinholeCamera`: pixel ``(0, 0)`` is centred at ``(0, 0)``.
          ``camera_matrix`` is the :math:`(3, 3)` ``[[fx, 0, cx], [0, fy, cy], [0, 0, 1]]`` of that grid and
          there are no extrinsics, so the points are in the **camera** frame.
        - ``normalize_points=True`` reads ``depth`` as the Euclidean ray length from the camera centre instead
          of as ``z``, so the returned point has that norm rather than that ``z``.
        - an integer ``depth`` map is promoted through arithmetic with ``camera_matrix``: with floating-point
          intrinsics the point cloud follows their dtype (for example, ``float32`` or ``float64``).

    Args:
        depth: image tensor containing a depth value per pixel with shape :math:`(B, 1, H, W)`.
        camera_matrix: tensor containing the camera intrinsics with shape :math:`(B, 3, 3)`.
        normalize_points: whether to normalise the pointcloud. This must be set to `True` when the depth is
          represented as the Euclidean ray length from the camera position.

    Return:
        tensor with a 3d point per pixel of the same resolution as the input :math:`(B, 3, H, W)`.

    Example:
        >>> depth = torch.rand(1, 1, 4, 4)
        >>> K = torch.eye(3)[None]
        >>> depth_to_3d(depth, K).shape
        torch.Size([1, 3, 4, 4])

    """
    KORNIA_CHECK_IS_TENSOR(depth)
    KORNIA_CHECK_IS_TENSOR(camera_matrix)
    KORNIA_CHECK_SHAPE(depth, ["B", "1", "H", "W"])
    KORNIA_CHECK_SHAPE(camera_matrix, ["B", "3", "3"])

    # create base coordinates grid
    _, _, height, width = depth.shape
    points_2d: torch.Tensor = create_meshgrid(
        height, width, normalized_coordinates=False, device=depth.device, dtype=depth.dtype
    )  # 1xHxWx2

    # depth should come in Bx1xHxW
    points_depth: torch.Tensor = depth.permute(0, 2, 3, 1)  # 1xHxWx1

    # project pixels to camera frame
    camera_matrix_tmp: torch.Tensor = camera_matrix[:, None, None]  # Bx1x1x3x3
    points_3d: torch.Tensor = unproject_points(
        points_2d, points_depth, camera_matrix_tmp, normalize=normalize_points
    )  # BxHxWx3

    return points_3d.permute(0, 3, 1, 2)  # Bx3xHxW


def depth_to_normals(depth: torch.Tensor, camera_matrix: torch.Tensor, normalize_points: bool = False) -> torch.Tensor:
    """Compute the normal surface per pixel.

    Convention:
        - the normal is the cross product of the two spatial gradients of the unprojected point cloud, taken in
          the order ``d/dx`` cross ``d/dy`` and then normalized to unit length, so a fronto-parallel plane gets
          the normal ``(0, 0, 1)``: ``+z`` points **away** from the camera, along the viewing direction, and
          not back towards it.
        - the same order fixes the two in-plane signs: a depth that grows along the column axis tilts the
          normal towards ``-x``, and a depth that grows along the row axis tilts it towards ``-y``.
        - ``depth``, ``camera_matrix`` and ``normalize_points`` mean here what they mean for
          :func:`~kornia.geometry.depth.depth_to_3d`, whose Convention block states the pixel grid, the camera
          frame and the two readings of ``depth``; the result carries that function's :math:`(B, 3, H, W)`
          layout, with the three normal components on the channel axis.

    Args:
        depth: image tensor containing a depth value per pixel with shape :math:`(B, 1, H, W)`.
        camera_matrix: tensor containing the camera intrinsics with shape :math:`(B, 3, 3)`.
        normalize_points: whether to normalize the pointcloud. This must be set to `True` when the depth is
        represented as the Euclidean ray length from the camera position.

    Return:
        tensor with a normal surface vector per pixel of the same resolution as the input :math:`(B, 3, H, W)`.

    Example:
        >>> depth = torch.rand(1, 1, 4, 4)
        >>> K = torch.eye(3)[None]
        >>> depth_to_normals(depth, K).shape
        torch.Size([1, 3, 4, 4])

    """
    KORNIA_CHECK_IS_TENSOR(depth)
    KORNIA_CHECK_IS_TENSOR(camera_matrix)
    KORNIA_CHECK_SHAPE(depth, ["B", "1", "H", "W"])
    KORNIA_CHECK_SHAPE(camera_matrix, ["B", "3", "3"])

    # compute the 3d points from depth; permute to channel-first for spatial_gradient
    xyz: torch.Tensor = depth_to_3d_v2(depth.squeeze(1), camera_matrix, normalize_points).permute(0, 3, 1, 2)  # Bx3xHxW

    # compute the pointcloud spatial gradients
    gradients: torch.Tensor = spatial_gradient(xyz)  # Bx3x2xHxW

    # Rearrange to (B, H, W, 3) before cross product so the 3 XYZ components are
    # contiguous in memory.  Cross product along dim=1 on a (B,3,H,W) tensor strides
    # H*W elements between components, causing severe cache thrashing on CPU.
    a: torch.Tensor = gradients[:, :, 0].permute(0, 2, 3, 1).contiguous()  # BxHxWx3
    b: torch.Tensor = gradients[:, :, 1].permute(0, 2, 3, 1).contiguous()  # BxHxWx3

    normals: torch.Tensor = torch.linalg.cross(a, b, dim=-1)  # BxHxWx3
    return F.normalize(normals, dim=-1, p=2).permute(0, 3, 1, 2)  # Bx3xHxW


def depth_from_plane_equation(
    plane_normals: torch.Tensor,
    plane_offsets: torch.Tensor,
    points_uv: torch.Tensor,
    camera_matrix: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    r"""Compute depth values from plane equations and pixel coordinates.

    Convention:
        - the plane is given in Hessian form :math:`n \cdot X = d`, with the normal ``plane_normals`` and the
          offset ``plane_offsets`` in the **camera** frame: ``n = (0, 0, 1)`` with ``d = 2`` is the plane
          ``z = 2``, and every pixel on it has depth 2.
        - ``points_uv`` are pixel coordinates on the integer-centre grid described in the Convention block on
          :class:`~kornia.geometry.camera.pinhole.PinholeCamera`; this function normalizes them with
          ``camera_matrix`` itself, so they are pixels and not normalized coordinates.
        - the result is the camera-frame ``z`` of each of those pixels, one value per pixel, in the
          :math:`(B, N)` layout of ``points_uv`` -- a list of depths rather than a depth map.
        - the ray-plane dot product is clamped, in a masked branch, to :math:`\pm` ``eps`` when it falls
          strictly inside :math:`(-eps, eps)`, keeping the sign of the denominator, so a nearly-grazing ray
          returns a large signed depth. An exactly zero denominator is replaced by positive ``eps``.
          The result is finite when representable in the input dtype; small ``eps`` can still underflow
          or produce overflow in ``float16``. Nothing outside the mask is touched.

    Args:
        plane_normals (torch.Tensor): Plane normal vectors of shape (B, 3).
        plane_offsets (torch.Tensor): Plane offsets of shape (B, 1).
        points_uv (torch.Tensor): Pixel coordinates of shape (B, N, 2).
        camera_matrix (torch.Tensor): Camera intrinsic matrix of shape (B, 3, 3).
        eps: epsilon for numerical stability.

    Returns:
        torch.Tensor: Computed depth values at the given pixels, shape (B, N).

    """
    KORNIA_CHECK_SHAPE(plane_normals, ["B", "3"])
    KORNIA_CHECK_SHAPE(plane_offsets, ["B", "1"])
    KORNIA_CHECK_SHAPE(points_uv, ["B", "N", "2"])
    KORNIA_CHECK_SHAPE(camera_matrix, ["B", "3", "3"])

    # Normalize pixel coordinates
    points_xy = normalize_points_with_intrinsics(points_uv, camera_matrix)  # (B, N, 2)
    rays = convert_points_to_homogeneous(points_xy)  # (B, N, 3)

    # Reshape plane normals to match rays
    plane_normals_exp = plane_normals.unsqueeze(1)  # (B, 1, 3)
    # No need to unsqueeze plane_offsets; it is already (B, 1)

    # Compute the denominator of the depth equation
    denom = torch.sum(rays * plane_normals_exp, dim=-1)  # (B, N)
    denom_abs = torch.abs(denom)
    zero_mask = denom_abs < eps
    # The guard was `eps * sign(denom)`, and `sign` is zero at zero, so the
    # multiplication cancelled the guard at the exact singularity it exists for
    # and a ray parallel to the plane returned inf. Choose the sign with a
    # comparison instead: it has no hole at zero, and keeps the branch's sign
    # for the small non-zero denominators the guard already handled.
    # `torch.copysign` would read the same but is not exportable -- the legacy
    # ONNX exporter has no `aten::copysign` and the dynamo one has no ONNX
    # function for the `prims.signbit` it decomposes to -- and this function is
    # in the documented export surface (docs/export_support/cases_geomB.py).
    signed_eps = torch.where(denom < 0, torch.full_like(denom, -eps), torch.full_like(denom, eps))
    denom = torch.where(zero_mask, signed_eps, denom)

    # Compute depth from plane equation
    depth = plane_offsets / denom  # plane_offsets: (B, 1), denom: (B, N) -> depth: (B, N)
    return depth


def warp_frame_depth(
    image_src: torch.Tensor,
    depth_dst: torch.Tensor,
    src_trans_dst: torch.Tensor,
    camera_matrix: torch.Tensor,
    normalize_points: bool = False,
) -> torch.Tensor:
    """Warp a tensor from a source to destination frame by the depth in the destination.

    Compute 3d points from the depth, transform them using given transformation, then project the point cloud to an
    image plane.

    Convention:
        - the depth belongs to the **destination** frame (``depth_dst``) and the image to the **source** frame
          (``image_src``), and ``src_trans_dst`` maps destination-frame points into the source frame. With
          ``fx = fy = 1`` and a unit depth, a :math:`+1` translation in ``x`` therefore samples ``image_src``
          one pixel to the right of each destination pixel.
        - the unprojection is :func:`~kornia.geometry.depth.depth_to_3d_v2` and the reprojection
          :func:`~kornia.geometry.camera.perspective.project_points`, so the pixel grid, the camera frame and
          the two readings of ``depth`` selected by ``normalize_points`` are the ones stated in that function's
          Convention block.
        - the sampling is ``grid_sample`` with ``align_corners=True`` and the default
          ``padding_mode="zeros"``, both baked in: the function exposes neither. Bilinear interpolation blends in-bounds
          neighbors with the zero extension outside the image, so subpixel samples just beyond the border
          can be nonzero; samples whose entire interpolation footprint is outside return 0.
        - the result carries ``image_src``'s channel count, whatever it is: the output is :math:`(B, D, H, W)`.

    .. warning::
        :class:`~kornia.geometry.depth.DepthWarper` performs the same warp under the **opposite** naming: the
        frame this function calls ``dst`` (the one holding the depth) is that class's ``src``, and the image it
        calls ``image_src`` is that class's ``patch_dst``. The Convention block on
        :class:`~kornia.geometry.depth.DepthWarper` states the mapping in full. Tracked as
        `#4273 <https://github.com/kornia/kornia/issues/4273>`_.

    .. warning::
        An empty batch (:math:`B = 0`) raises ``ZeroDivisionError`` from ``transform_points`` -- its
        batch-repeat count is ``0 // 0`` -- before any sampling runs, rather than returning an empty result,
        although the shape guards on the way in accept it and
        :func:`~kornia.geometry.depth.depth_to_3d` -- the same unprojection in the other layout -- returns an
        empty point cloud. Tracked as `#4281 <https://github.com/kornia/kornia/issues/4281>`_.

    Args:
        image_src: image tensor in the source frame with shape :math:`(B,D,H,W)`.
        depth_dst: depth tensor in the destination frame with shape :math:`(B,1,H,W)`.
        src_trans_dst: transformation matrix from destination to source with shape :math:`(B,4,4)`.
        camera_matrix: tensor containing the camera intrinsics with shape :math:`(B,3,3)`.
        normalize_points: whether to normalize the pointcloud. This must be set to ``True`` when the depth
           is represented as the Euclidean ray length from the camera position.

    Return:
        ``image_src`` resampled onto the destination pixel grid, with shape :math:`(B,D,H,W)`.

    """
    KORNIA_CHECK_SHAPE(image_src, ["B", "D", "H", "W"])
    KORNIA_CHECK_SHAPE(depth_dst, ["B", "1", "H", "W"])
    KORNIA_CHECK_SHAPE(src_trans_dst, ["B", "4", "4"])
    KORNIA_CHECK_SHAPE(camera_matrix, ["B", "3", "3"])

    # unproject source points to camera frame as (B, H, W, 3) directly — avoids two permutes
    points_3d_dst: torch.Tensor = depth_to_3d_v2(depth_dst.squeeze(1), camera_matrix, normalize_points)  # BxHxWx3

    # apply transformation to the 3d points
    points_3d_src = transform_points(src_trans_dst[:, None], points_3d_dst)  # BxHxWx3

    # project back to pixels
    camera_matrix_tmp: torch.Tensor = camera_matrix[:, None, None]  # Bx1x1xHxW
    points_2d_src: torch.Tensor = project_points(points_3d_src, camera_matrix_tmp)  # BxHxWx2

    # normalize points between [-1 / 1]
    height, width = depth_dst.shape[-2:]
    points_2d_src_norm: torch.Tensor = normalize_pixel_coordinates(points_2d_src, height, width)  # BxHxWx2

    return F.grid_sample(image_src, points_2d_src_norm, align_corners=True)


class DepthWarper(nn.Module):
    r"""Warp a patch by depth.

    .. math::
        P_{src}^{\{dst\}} = K_{dst} * T_{src}^{\{dst\}}

        I_{src} = \\omega(I_{dst}, P_{src}^{\{dst\}}, D_{src})

    Convention:
        - the depth lives in the **source** frame and the image in the **destination** frame:
          :meth:`forward` takes ``(depth_src, patch_dst)``, samples ``patch_dst`` at the pixels the
          source-frame depth projects to, and returns a :math:`(B, C, H, W)` tensor carrying ``patch_dst``'s
          channel count.
        - this is a two-step API: :meth:`compute_projection_matrix` has to be called first. Until it has
          been, :meth:`warp_grid` and :meth:`forward` raise ``ValueError`` and :meth:`compute_subpixel_step`
          raises ``RuntimeError``.
        - :meth:`compute_projection_matrix` stores exactly the matrix of the equation above,
          ``K_dst @ (E_dst @ inverse(E_src))``, with ``K_dst`` and ``E_dst`` the ``intrinsics`` and
          ``extrinsics`` of the ``pinhole_dst`` given to the constructor and ``E_src`` the ``extrinsics`` of
          the ``pinhole_src`` given to that method. The source extrinsics are inverted, not transposed, so the
          translation is carried through.
        - both cameras are :class:`~kornia.geometry.camera.pinhole.PinholeCamera` objects, so their
          ``intrinsics`` are the :math:`(B, 4, 4)` matrix and their ``extrinsics`` the world-to-camera
          transform that class's Convention block describes.
        - the ``grid`` attribute is this instance's own grid of integer pixel centres -- the grid
          :func:`~kornia.geometry.grid.create_meshgrid` enumerates and the Convention block on
          :class:`~kornia.geometry.camera.pinhole.PinholeCamera` describes -- in homogeneous form: pixel
          ``(0, 0)`` is ``(0, 0, 1)``. :meth:`warp_grid` returns something else: the sampling positions in
          ``grid_sample``'s ``align_corners=True`` normalized coordinates, in which the **destination**
          image's integer-centre pixel ``(0, 0)`` is ``(-1, -1)`` and its pixel ``(H - 1, W - 1)`` is
          ``(1, 1)``. Where each destination pixel actually lands is wherever the reprojection sends it: an
          identity camera pair leaves pixel ``(0, 0)`` at ``(-1, -1)``, a rotated and translated one sends
          it elsewhere.
        - ``align_corners`` defaults to ``True``; it is handed to ``grid_sample`` unchanged, together with
          ``mode`` and ``padding_mode``.
        - :func:`~kornia.geometry.depth.depth_warp` is the functional form of this class -- it builds one,
          calls :meth:`compute_projection_matrix` and forwards -- and returns a result equal to it bit for bit.
          It exposes ``align_corners`` only; ``mode`` and ``padding_mode`` keep their defaults there.

    .. warning::
        :func:`~kornia.geometry.depth.warp_frame_depth` performs the same warp under the **opposite** naming.
        The frame this class calls ``src``, the one holding the depth, is that function's ``dst``
        (``depth_dst``), and the image this class takes as ``patch_dst`` is that function's ``image_src``;
        reading "dst" as "dst" across the two APIs gives the inverse warp. The two also differ arithmetically:
        they build the sampling grid by different routes -- :func:`~kornia.geometry.depth.depth_to_3d_v2` and
        :func:`~kornia.geometry.camera.perspective.project_points` here, ``pixel2cam`` and ``cam2pixel`` there
        -- so the grid, and with it the resampled image, can differ in the last bits. Where the two grids come
        out bit-identical, so do the images. Wherever the transformed points keep a camera-frame ``z`` away
        from zero, the two agree at the working dtype's tolerance in float32 and float64; in float16 and
        bfloat16 the gap is wider than that tolerance, which is why the agreement is claimed for the two
        single- and double-precision dtypes only. At ``z = 0`` the two split outright, because their two
        projection routes guard the singularity differently:
        :func:`~kornia.geometry.camera.perspective.project_points` skips the homogeneous divide when
        ``abs(z) <= 1e-8``, so :func:`~kornia.geometry.depth.warp_frame_depth` samples ``image_src`` at the
        undivided ``(x, y)`` and returns image content, while ``cam2pixel`` divides by ``z + 1e-12`` and sends
        the same pixel to a coordinate of order ``1e12``, far outside the image. That split is one instance of
        `#4267 <https://github.com/kornia/kornia/issues/4267>`_, the namespace-wide ``z = 0`` conflict. The
        naming conflict is tracked as `#4273 <https://github.com/kornia/kornia/issues/4273>`_.

    Args:
        pinhole_dst: the pinhole model for the destination frame.
        height: the height of the image to warp.
        width: the width of the image to warp.
        mode: interpolation mode to calculate output values ``'bilinear'`` | ``'nearest'``.
        padding_mode: padding mode for outside grid values ``'zeros'`` | ``'border'`` | ``'reflection'``.
        align_corners: interpolation flag.

    """

    # All per-instance, not global (thread safe, multiple warps)
    def __init__(
        self,
        pinhole_dst: PinholeCamera,
        height: int,
        width: int,
        mode: str = "bilinear",
        padding_mode: str = "zeros",
        align_corners: bool = True,
    ) -> None:
        super().__init__()
        self.width: int = width
        self.height: int = height
        self.mode: str = mode
        self.padding_mode: str = padding_mode
        self.eps = 1e-6
        self.align_corners: bool = align_corners

        # state members
        # _pinhole_dst is Type[PinholeCamera], enforce in constructor
        if not isinstance(pinhole_dst, PinholeCamera):
            raise TypeError(f"Expected pinhole_dst as PinholeCamera, got {type(pinhole_dst)}")
        self._pinhole_dst: PinholeCamera = pinhole_dst
        self._pinhole_src: PinholeCamera | None = None
        self._dst_proj_src: torch.Tensor | None = None

        # Meshgrid only depends on (height, width), can be staticmethod cached
        self.grid: torch.Tensor = self._create_meshgrid(height, width)

    @staticmethod
    def _create_meshgrid(height: int, width: int) -> torch.Tensor:
        grid: torch.Tensor = create_meshgrid(height, width, normalized_coordinates=False)  # 1xHxWx2
        return convert_points_to_homogeneous(grid)  # append ones to last dim

    def compute_projection_matrix(self, pinhole_src: PinholeCamera) -> DepthWarper:
        """Compute the projection matrix from the source to destination frame.

        See the Convention block on :class:`~kornia.geometry.depth.DepthWarper`.
        """
        # Inline type checks for faster fail-fast
        if type(self._pinhole_dst) is not PinholeCamera:
            raise TypeError(
                f"Member self._pinhole_dst expected to be of class PinholeCamera. Got {type(self._pinhole_dst)}"
            )
        if type(pinhole_src) is not PinholeCamera:
            raise TypeError(f"Argument pinhole_src expected to be of class PinholeCamera. Got {type(pinhole_src)}")
        # Compute transformation matrix: dst_extrinsics @ inv(src_extrinsics)
        batch_shape = pinhole_src.extrinsics.shape[:-2]
        device = pinhole_src.extrinsics.device
        dtype = pinhole_src.extrinsics.dtype

        # Create 4x4 identity matrices efficiently
        inv_extr = torch.eye(4, device=device, dtype=dtype).expand(*batch_shape, 4, 4).contiguous()
        dst_trans_src = torch.eye(4, device=device, dtype=dtype).expand(*batch_shape, 4, 4).contiguous()

        # Inline inverse transformation
        src_rmat = pinhole_src.extrinsics[..., :3, :3]
        src_tvec = pinhole_src.extrinsics[..., :3, 3:]
        inv_rmat = torch.transpose(src_rmat, -1, -2)
        inv_tvec = torch.matmul(-inv_rmat, src_tvec)

        # Set rotation and translation parts
        inv_extr[..., :3, :3] = inv_rmat
        inv_extr[..., :3, 3:] = inv_tvec

        # Compose with dst extrinsics
        dst_rmat = self._pinhole_dst.extrinsics[..., :3, :3]
        dst_tvec = self._pinhole_dst.extrinsics[..., :3, 3:]
        composed_rmat = torch.matmul(dst_rmat, inv_rmat)
        composed_tvec = torch.matmul(dst_rmat, inv_tvec) + dst_tvec

        dst_trans_src[..., :3, :3] = composed_rmat
        dst_trans_src[..., :3, 3:] = composed_tvec

        # intrinsics (Nx3x3) @ extrinsics (Nx4x4)
        dst_proj_src = torch.matmul(self._pinhole_dst.intrinsics, dst_trans_src)

        self._pinhole_src = pinhole_src
        self._dst_proj_src = dst_proj_src
        return self

    def _compute_projection(self, x: float, y: float, invd: float) -> torch.Tensor:
        if self._dst_proj_src is None or self._pinhole_src is None:
            raise ValueError("Please, call compute_projection_matrix.")

        point = torch.tensor(
            [[[x], [y], [invd], [1.0]]], device=self._dst_proj_src.device, dtype=self._dst_proj_src.dtype
        )
        flow = torch.matmul(self._dst_proj_src, point)
        z = 1.0 / flow[:, 2]
        _x = flow[:, 0] * z
        _y = flow[:, 1] * z
        return torch.cat([_x, _y], 1)

    def compute_subpixel_step(self) -> torch.Tensor:
        """Compute the inverse depth step for sub pixel accurate sampling of the depth cost volume, per camera.

        See the Convention block on :class:`~kornia.geometry.depth.DepthWarper`.

        Szeliski, Richard, and Daniel Scharstein. "Symmetric sub-pixel stereo matching." European Conference on Computer
        Vision. Springer Berlin Heidelberg, 2002.
        """
        if self._dst_proj_src is None:
            raise RuntimeError("Expected torch.Tensor, but got None Type from the projection matrix")

        delta_d = 0.01
        center_x = self.width / 2
        center_y = self.height / 2

        # Batch both invds in one call (for potential fused kernels in future) for efficiency
        invds = (1.0 - delta_d, 1.0 + delta_d)
        # Instead of two calls, process both at once with minimal tensor construction
        points = (
            torch.tensor(
                [[center_x, center_y, invds[0], 1.0], [center_x, center_y, invds[1], 1.0]],
                dtype=self._dst_proj_src.dtype,
                device=self._dst_proj_src.device,
            )
            .transpose(0, 1)
            .unsqueeze(0)
        )  # (1, 4, 2)
        # Repeat projection matrix for batch
        proj = self._dst_proj_src
        flow = torch.matmul(proj, points)  # (N, 3/4, 2)
        zs = 1.0 / flow[:, 2]  # (N, 2)
        xs = flow[:, 0] * zs
        ys = flow[:, 1] * zs
        xys = torch.stack((xs, ys), dim=-1)  # (N, 2, 2)
        dxy = torch.norm(xys[:, 1] - xys[:, 0], p=2, dim=1) / 2.0
        dxdd = dxy / delta_d
        # half pixel sampling, min for all cameras
        return torch.min(0.5 / dxdd)

    def warp_grid(self, depth_src: torch.Tensor) -> torch.Tensor:
        """Compute a grid for warping a given the depth from the reference pinhole camera.

        See the Convention block on :class:`~kornia.geometry.depth.DepthWarper`.

        The function `compute_projection_matrix` has to be called beforehand in order to have precomputed the relative
        projection matrices encoding the relative pose and the intrinsics between the reference and a non reference
        camera.
        """
        # TODO: add type and value checkings
        if self._dst_proj_src is None or self._pinhole_src is None:
            raise ValueError("Please, call compute_projection_matrix.")

        if len(depth_src.shape) != 4:
            raise ValueError(f"Input depth_src has to be in the shape of Bx1xHxW. Got {depth_src.shape}")

        # unpack depth attributes
        batch_size, _, _, _ = depth_src.shape
        device: torch.device = depth_src.device
        dtype: torch.dtype = depth_src.dtype

        # expand the base coordinate grid according to the input batch size
        pixel_coords: torch.Tensor = self.grid.to(device=device, dtype=dtype).expand(batch_size, -1, -1, -1)  # BxHxWx3

        # reproject the pixel coordinates to the camera frame
        cam_coords_src: torch.Tensor = pixel2cam(
            depth_src, self._pinhole_src.intrinsics_inverse().to(device=device, dtype=dtype), pixel_coords
        )  # BxHxWx3

        # reproject the camera coordinates to the pixel
        pixel_coords_src: torch.Tensor = cam2pixel(
            cam_coords_src, self._dst_proj_src.to(device=device, dtype=dtype)
        )  # (B*N)xHxWx2

        # normalize between -1 and 1 the coordinates
        pixel_coords_src_norm: torch.Tensor = normalize_pixel_coordinates(pixel_coords_src, self.height, self.width)
        return pixel_coords_src_norm

    def forward(self, depth_src: torch.Tensor, patch_dst: torch.Tensor) -> torch.Tensor:
        """Warp a tensor from destination frame to reference given the depth in the reference frame.

        See the Convention block on :class:`~kornia.geometry.depth.DepthWarper`.

        Args:
            depth_src: the depth in the reference frame. The tensor must have a shape :math:`(B, 1, H, W)`.
            patch_dst: the patch in the destination frame. The tensor must have a shape :math:`(B, C, H, W)`.

        Return:
            the warped patch from destination frame to reference.

        Shape:
            - Input: :math:`(B, 1, H, W)` and :math:`(B, C, H, W)`.
            - Output: :math:`(B, C, H, W)` where C = number of channels.

        Example:
            >>> # pinholes camera models
            >>> pinhole_dst = PinholeCamera(torch.randn(1, 4, 4), torch.randn(1, 4, 4),
            ... torch.tensor([32]), torch.tensor([32]))
            >>> pinhole_src = PinholeCamera(torch.randn(1, 4, 4), torch.randn(1, 4, 4),
            ... torch.tensor([32]), torch.tensor([32]))
            >>> # create the depth warper, compute the projection matrix
            >>> warper = DepthWarper(pinhole_dst, 32, 32)
            >>> _ = warper.compute_projection_matrix(pinhole_src)
            >>> # warp the destination frame to reference by depth
            >>> depth_src = torch.ones(1, 1, 32, 32)  # Nx1xHxW
            >>> image_dst = torch.rand(1, 3, 32, 32)  # NxCxHxW
            >>> image_src = warper(depth_src, image_dst)  # NxCxHxW

        """
        return F.grid_sample(
            patch_dst,
            self.warp_grid(depth_src),
            mode=self.mode,
            padding_mode=self.padding_mode,
            align_corners=self.align_corners,
        )


def depth_warp(
    pinhole_dst: PinholeCamera,
    pinhole_src: PinholeCamera,
    depth_src: torch.Tensor,
    patch_dst: torch.Tensor,
    height: int,
    width: int,
    align_corners: bool = True,
) -> torch.Tensor:
    """Warp a tensor from destination frame to reference given the depth in the reference frame.

    See the Convention block on :class:`~kornia.geometry.depth.DepthWarper`.

    This function is that class's functional form: it constructs a
    :class:`~kornia.geometry.depth.DepthWarper`, calls its ``compute_projection_matrix`` and returns its
    output, which is equal to the class's bit for bit under the default ``mode`` and ``padding_mode``, the only
    ones this function offers (``align_corners`` is forwarded).

    Example:
        >>> # pinholes camera models
        >>> pinhole_dst = PinholeCamera(torch.randn(1, 4, 4), torch.randn(1, 4, 4),
        ... torch.tensor([32]), torch.tensor([32]))
        >>> pinhole_src = PinholeCamera(torch.randn(1, 4, 4), torch.randn(1, 4, 4),
        ... torch.tensor([32]), torch.tensor([32]))
        >>> # warp the destination frame to reference by depth
        >>> depth_src = torch.ones(1, 1, 32, 32)  # Nx1xHxW
        >>> image_dst = torch.rand(1, 3, 32, 32)  # NxCxHxW
        >>> image_src = depth_warp(pinhole_dst, pinhole_src, depth_src, image_dst, 32, 32)  # NxCxHxW

    """
    # Cache and reuse warper and projection matrix (single use/call)
    # Inlined for performance, use local variables and freed objects
    # instead of class members where possible.
    warper = DepthWarper(pinhole_dst, height, width, align_corners=align_corners)
    # projection matrix is required for each call, avoid double checking in class
    warper.compute_projection_matrix(pinhole_src)
    # __call__ implemented by nn.Module (likely calls forward, not shown).
    return warper(depth_src, patch_dst)


def depth_from_disparity(
    disparity: torch.Tensor, baseline: float | torch.Tensor, focal: float | torch.Tensor
) -> torch.Tensor:
    """Compute depth from disparity.

    Convention:
        - the depth is ``baseline * focal / disparity``, elementwise: ``baseline`` is the distance between the
          two camera centres and ``focal`` the focal length in pixels, so a disparity of 2 with a baseline of
          0.5 and a focal length of 100 gives a depth of 25.
        - ``baseline`` and ``focal`` are each a python ``float`` or a tensor of shape :math:`(1,)`. A 0-dim
          tensor -- what a reduction produces -- and a per-batch-element :math:`(B,)` tensor both raise
          ``ShapeError``, and a python ``int`` is rejected by the type check, so one value is shared by the
          whole batch.
        - ``disparity`` is :math:`(*, H, W)` and the result has its shape. Its sign is not checked, so a
          negative disparity gives a negative depth.

    .. warning::
        The epsilon is inside the arithmetic -- the divisor is ``disparity + 1e-8`` -- instead of selecting a
        branch, so a zero disparity, which is what a stereo matcher writes where it found no match, returns a
        large finite depth rather than ``inf`` in float32, float64 and bfloat16: ``baseline * focal / 1e-8``,
        ``5e9`` for a baseline of ``0.5`` and a focal length of ``100``, a value set by the epsilon as much as
        by the camera and one no caller can threshold against. In float16 the ``1e-8`` itself rounds to zero,
        so the same call divides by zero and returns ``inf`` after all. Tracked as
        `#4272 <https://github.com/kornia/kornia/issues/4272>`_.

    Args:
        disparity: Disparity tensor of shape :math:`(*, H, W)`.
        baseline: a python ``float`` or a tensor of shape :math:`(1,)` containing the distance between the two
          lenses.
        focal: a python ``float`` or a tensor of shape :math:`(1,)` containing the focal length.

    Return:
        Depth map of the shape :math:`(*, H, W)`.

    Example:
        >>> disparity = torch.rand(4, 1, 4, 4)
        >>> baseline = torch.rand(1)
        >>> focal = torch.rand(1)
        >>> depth_from_disparity(disparity, baseline, focal).shape
        torch.Size([4, 1, 4, 4])

    """
    KORNIA_CHECK_IS_TENSOR(disparity, f"Input disparity type is not a torch.Tensor. Got {type(disparity)}.")
    KORNIA_CHECK_SHAPE(disparity, ["*", "H", "W"])
    KORNIA_CHECK(
        isinstance(baseline, (float, torch.Tensor)),
        f"Input baseline should be either a float or torch.Tensor. Got {type(baseline)}",
    )
    KORNIA_CHECK(
        isinstance(focal, (float, torch.Tensor)),
        f"Input focal should be either a float or torch.Tensor. Got {type(focal)}",
    )

    if isinstance(baseline, torch.Tensor):
        KORNIA_CHECK_SHAPE(baseline, ["1"])

    if isinstance(focal, torch.Tensor):
        KORNIA_CHECK_SHAPE(focal, ["1"])

    return baseline * focal / (disparity + 1e-8)
