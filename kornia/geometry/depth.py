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

"""Depth maps: unproject them to 3D points and normals, get depth from stereo or planes, and warp images by depth.

Functions that take ``camera_matrix`` expect pixel intrinsics of shape ``(B, 3, 3)`` with integer pixel
centres, and read ``depth`` as the camera-frame ``z`` (the Euclidean ray length with ``normalize_points=True``).

Which function to use:

- 3D points from a depth map: :func:`depth_to_3d` returns ``(B, 3, H, W)``; :func:`depth_to_3d_v2` returns
  ``(*, H, W, 3)`` and can reuse the rays from :func:`unproject_meshgrid` for every frame of one camera.
- Surface normals from a depth map: :func:`depth_to_normals`.
- Depth from a rectified stereo disparity map: :func:`depth_from_disparity`. Depth of given pixels on a known
  plane: :func:`depth_from_plane_equation`.
- An image rendered from another viewpoint by depth: :func:`warp_frame_depth` (a 4x4 pose and one camera
  matrix), or :class:`DepthWarper` and :func:`depth_warp` (two :class:`~kornia.geometry.camera.pinhole.PinholeCamera`
  models).
- Selected pixels instead of whole maps: :func:`~kornia.geometry.camera.perspective.unproject_points` and
  :func:`~kornia.geometry.camera.perspective.project_points`.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from kornia.core.check import KORNIA_CHECK, KORNIA_CHECK_IS_TENSOR, KORNIA_CHECK_SHAPE, KORNIA_CHECK_SIZES
from kornia.core.exceptions import ShapeError
from kornia.core.utils import _autocast_disabled, _torch_grid_sample
from kornia.filters.sobel import spatial_gradient
from kornia.geometry.grid import create_meshgrid

from .camera import PinholeCamera, cam2pixel, pixel2cam, project_points, unproject_points
from .camera.pinhole import _embed_intrinsics
from .conversions import normalize_pixel_coordinates, normalize_points_with_intrinsics
from .linalg import convert_points_to_homogeneous, transform_points

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


def _depth_grid_dtype(depth: torch.Tensor, camera_matrix: torch.Tensor) -> torch.dtype:
    """Return the dtype the pixel grid is built in before unprojection.

    The grid holds the integer pixel indices at the precision of the promoted inputs: a bfloat16 or
    float16 grid rounds them, and an integer grid cannot represent them. Integer inputs promote to
    the default floating dtype instead: ``promote_types`` does not support unsigned/signed integer
    pairs. A complex promotion keeps a real grid, because ``arange`` has no complex kernels and the
    subsequent arithmetic promotes to complex anyway, which also unblocks complex inputs. An all-half
    call keeps its half grid.
    """
    depth_is_float = depth.is_floating_point()
    camera_is_float = camera_matrix.is_floating_point()
    if not depth_is_float and not camera_is_float and not depth.is_complex() and not camera_matrix.is_complex():
        return torch.empty(0, device=depth.device).dtype
    promoted = torch.promote_types(depth.dtype, camera_matrix.dtype)
    # A complex32 promotion is checked through the tensors, not the promoted dtype: TorchScript sees a
    # dtype as an integer, so a membership test on dtype attributes would not compile.
    if depth.is_complex() or camera_matrix.is_complex():
        return torch.empty(0, device=depth.device, dtype=promoted).real.dtype
    return promoted


def unproject_meshgrid(
    height: int,
    width: int,
    camera_matrix: torch.Tensor,
    normalize_points: bool = False,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """Compute the camera ray through every pixel, to reuse across frames with depth_to_3d_v2.

    ``rays = unproject_meshgrid(H, W, camera_matrix)`` returns the camera-frame ray of each pixel as a
    ``(B, H, W, 3)`` tensor, scaled so that ``z`` is 1 by default (unit length with ``normalize_points=True``). It
    takes no depth: multiply the rays by ``depth[..., None]`` for a ``(B, H, W)`` depth map, or pass them to
    :func:`depth_to_3d_v2` as ``xyz_grid``, to get the 3D points of many frames from one camera without rebuilding
    the grid each time.

    Convention:
        - pixel centres are integers; see :class:`~kornia.geometry.camera.pinhole.PinholeCamera`.
        - ``camera_matrix`` must be ``(B, 3, 3)``; an unbatched matrix or extra leading dimensions raise
          ``ShapeError``.

    Args:
        height: height of image.
        width: width of image.
        camera_matrix: tensor containing the camera intrinsics with shape ``(B, 3, 3)``.
        normalize_points: whether to normalize the pointcloud. This must be set to ``True`` when the depth is
          represented as the Euclidean ray length from the camera position.
        device: device to place the result on.
        dtype: dtype of the pixel grid; the output dtype follows promotion with ``camera_matrix``.

    Return:
        the ray through each pixel, with ``z`` equal to 1 (or unit length with ``normalize_points=True``), with
        shape ``(B, H, W, 3)``.

    See Also:
        - :func:`depth_to_3d_v2`: takes these rays as ``xyz_grid`` and scales them by a depth map.
        - :func:`~kornia.geometry.camera.perspective.unproject_points`: the camera-frame point of selected pixels.

    """
    KORNIA_CHECK_SHAPE(camera_matrix, ["B", "3", "3"])

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
        points_xyz = F.normalize(points_xyz, dim=-1, p=2.0)

    return points_xyz


def depth_to_3d_v2(
    depth: torch.Tensor,
    camera_matrix: torch.Tensor,
    normalize_points: bool = False,
    xyz_grid: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    # NOTE: when this replaces the `depth_to_3d` behaviour, a deprecated function should be added here, instead
    # of just replace the other function.
    """Unproject a depth map to a 3D point per pixel, channels last.

    ``points = depth_to_3d_v2(depth, camera_matrix)`` takes a ``(*, H, W)`` map of camera-frame depths and
    ``(B, 3, 3)`` pixel intrinsics, and returns the camera-frame point of every pixel as a ``(*, H, W, 3)``
    tensor. The ``(B, H, W, 3)`` rays of the ``B`` cameras broadcast against ``depth`` by torch's rules, so the
    camera batch lines up with the axis of ``depth`` just before ``H``, not with its first axis: an ``(H, W)`` depth
    gives one map per camera, a ``(B, H, W)`` depth pairs camera ``b`` with ``depth[b]``, and a ``(B, T, H, W)``
    depth pairs camera ``t`` with ``depth[:, t]`` when ``T == B``. With ``T == 1`` it pairs every camera with every
    map and returns ``(B, B, H, W, 3)``, so pass ``depth[:, 0]``; any other ``T`` raises, unless there is a single
    camera, which then serves every map. For one camera per ``(B, T, H, W)`` batch element, pass
    ``xyz_grid=unproject_meshgrid(H, W, camera_matrix)[:, None]``; the same grid also saves rebuilding the pixel
    rays for many frames of one camera. For a channels-first ``(B, 3, H, W)`` result, use :func:`depth_to_3d`.

    Convention:
        - pixel ``(u, v)``, with ``u`` the column and ``v`` the row, maps to ``((u - cx) z / fx, (v - cy) z / fy, z)``.
          Pixel centres are integers, as in :class:`~kornia.geometry.camera.pinhole.PinholeCamera`.
          ``depth_to_3d_v2(depth[:, 0], K)`` equals ``depth_to_3d(depth, K).permute(0, 2, 3, 1)``.
        - without ``xyz_grid``, ``camera_matrix`` must be exactly ``(B, 3, 3)`` (another shape raises
          ``ShapeError``); with ``xyz_grid`` only its shape is checked.
        - ``normalize_points=True`` reads ``depth`` as the Euclidean ray length instead of ``z``. With ``xyz_grid`` it
          is ignored: build the grid with the same ``normalize_points``.
        - without ``xyz_grid``, the pixel grid is built in ``promote_types(depth.dtype, camera_matrix.dtype)``
          when either input is floating (the default floating dtype when both are integer), so half-precision
          depth does not round the pixel indices below the intrinsics' precision.

    Args:
        depth: image tensor containing a depth value per pixel with shape ``(*, H, W)``.
        camera_matrix: tensor containing the camera intrinsics with shape ``(B, 3, 3)``.
        normalize_points: whether to normalize the pointcloud. This must be set to ``True`` when the depth is
          represented as the Euclidean ray length from the camera position.
        xyz_grid: explicit xyz point values.

    Return:
        tensor with a 3d point per pixel of the same resolution as the input, ``(*, H, W, 3)``, whose
        leading dimensions are the broadcast of ``depth``'s and the grid's: the Example broadcasts a
        ``(4, 4)`` depth against a ``(2, 3, 3)`` camera to ``(2, 4, 4, 3)``, and a
        ``(B, T, H, W)`` depth with an ``xyz_grid`` of shape ``(B, 1, H, W, 3)`` returns
        ``(B, T, H, W, 3)``.

    See Also:
        - :func:`depth_to_3d`: the same points channels first, ``(B, 3, H, W)``.
        - :func:`unproject_meshgrid`: the per-pixel rays to pass as ``xyz_grid``.
        - :func:`depth_to_normals`: per-pixel surface normals, from a ``(B, 1, H, W)`` depth.

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
        else unproject_meshgrid(
            height, width, camera_matrix, normalize_points, depth.device, _depth_grid_dtype(depth, camera_matrix)
        )
    )

    KORNIA_CHECK_SHAPE(points_xyz, ["*", "H", "W", "3"])

    return points_xyz * depth[..., None]  # HxWx3


def depth_to_3d(depth: torch.Tensor, camera_matrix: torch.Tensor, normalize_points: bool = False) -> torch.Tensor:
    """Unproject a depth map to a 3D point per pixel, channels first.

    ``points = depth_to_3d(depth, camera_matrix)`` takes a ``(B, 1, H, W)`` map of camera-frame depths and
    ``(B, 3, 3)`` pixel intrinsics, and returns the camera-frame point of every pixel as a ``(B, 3, H, W)``
    tensor. For a channels-last ``(*, H, W, 3)`` result, or to reuse the pixel rays across frames, use
    :func:`depth_to_3d_v2` on ``depth[:, 0]``. To unproject selected pixels rather than a whole map, use
    :func:`~kornia.geometry.camera.perspective.unproject_points`.

    Convention:
        - pixel ``(u, v)``, with ``u`` the column and ``v`` the row, maps to ``((u - cx) z / fx, (v - cy) z / fy, z)``.
          Pixel centres are integers, as in :class:`~kornia.geometry.camera.pinhole.PinholeCamera`.
          ``depth_to_3d(depth, K).permute(0, 2, 3, 1)`` equals ``depth_to_3d_v2(depth[:, 0], K)``.
        - ``normalize_points=True`` reads ``depth`` as the Euclidean ray length instead of ``z``.
        - an integer ``depth`` is not rejected; the result takes the floating dtype of ``camera_matrix``.
          The pixel grid is built in ``promote_types(depth.dtype, camera_matrix.dtype)`` when either input is
          floating (the default floating dtype when both are integer), so half-precision ``depth`` does not
          round the pixel indices below the intrinsics' precision.

    Args:
        depth: image tensor containing a depth value per pixel with shape ``(B, 1, H, W)``.
        camera_matrix: tensor containing the camera intrinsics with shape ``(B, 3, 3)``.
        normalize_points: whether to normalize the pointcloud. This must be set to ``True`` when the depth is
          represented as the Euclidean ray length from the camera position.

    Return:
        tensor with a 3d point per pixel of the same resolution as the input ``(B, 3, H, W)``.

    See Also:
        - :func:`depth_to_3d_v2`: the same points channels last, with an optional cached grid.
        - :func:`depth_to_normals`: per-pixel surface normals from the same inputs.
        - :func:`depth_from_disparity`: depth from a stereo disparity map, to feed this function.
        - :func:`~kornia.geometry.camera.perspective.unproject_points`: the camera-frame point of selected pixels.

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
        height,
        width,
        normalized_coordinates=False,
        device=depth.device,
        dtype=_depth_grid_dtype(depth, camera_matrix),
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
    r"""Estimate a unit surface normal per pixel from a depth map.

    ``normals = depth_to_normals(depth, camera_matrix)`` takes the same ``(B, 1, H, W)`` depth map and
    ``(B, 3, 3)`` intrinsics as :func:`depth_to_3d`, differentiates the unprojected points with a Sobel
    :func:`~kornia.filters.spatial_gradient`, and returns the normalized cross product of the two tangents as a
    ``(B, 3, H, W)`` tensor. Normals point away from the camera: a fronto-parallel plane gets ``(0, 0, 1)``.

    Convention:
        - both spatial dimensions must be at least 2: a surface normal requires two tangent directions.
        - a depth growing along the columns tilts the normal towards ``-x``, one growing along the rows towards
          ``-y``.
        - float16 and bfloat16 inputs are widened to float32 first. The result has the promoted dtype of ``depth`` and
          ``camera_matrix``, or the default floating dtype when both are integer, and it does not follow an
          enclosing autocast region: its dtype and values are the same inside autocast as outside.

    Args:
        depth: image tensor containing a depth value per pixel with shape ``(B, 1, H, W)``, with
            :math:`H \geq 2` and :math:`W \geq 2`.
        camera_matrix: tensor containing the camera intrinsics with shape ``(B, 3, 3)``.
        normalize_points: whether to normalize the pointcloud. This must be set to ``True`` when the depth is
            represented as the Euclidean ray length from the camera position.

    Return:
        tensor with a normal surface vector per pixel of the same resolution as the input ``(B, 3, H, W)``.

    Raises:
        ShapeError: if either spatial dimension of ``depth`` is smaller than 2.

    See Also:
        - :func:`depth_to_3d`: the point map whose tangents give the normals.

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

    height, width = depth.shape[-2:]
    if height < 2 or width < 2:
        raise ShapeError(
            f"depth_to_normals requires H >= 2 and W >= 2. Got H={height}, W={width}.",
            actual_shape=list(depth.shape),
            expected_shape=["B", "1", "H >= 2", "W >= 2"],
        )

    dtype = torch.promote_types(depth.dtype, camera_matrix.dtype)
    # Compute in float32 when either input is half precision: bfloat16 cannot hold the pixel grid past 256 and the
    # float16 cross product of small pixel tangents underflows. The grid takes the promoted dtype, so cast each input.
    if depth.dtype in (torch.float16, torch.bfloat16):
        depth = depth.float()
    if camera_matrix.dtype in (torch.float16, torch.bfloat16):
        camera_matrix = camera_matrix.float()

    # compute the 3d points from depth; permute to channel-first for spatial_gradient
    xyz: torch.Tensor = depth_to_3d_v2(depth.squeeze(1), camera_matrix, normalize_points).permute(0, 3, 1, 2)  # Bx3xHxW
    dtype = dtype if dtype.is_floating_point else xyz.dtype

    # Keep the gradient convolution in the point-cloud dtype under autocast.
    with _autocast_disabled(xyz.device.type):
        gradients: torch.Tensor = spatial_gradient(xyz)  # Bx3x2xHxW

    # Rearrange to (B, H, W, 3) before cross product so the 3 XYZ components are
    # contiguous in memory.  Cross product along dim=1 on a (B,3,H,W) tensor strides
    # H*W elements between components, causing severe cache thrashing on CPU.
    a: torch.Tensor = gradients[:, :, 0].permute(0, 2, 3, 1).contiguous()  # BxHxWx3
    b: torch.Tensor = gradients[:, :, 1].permute(0, 2, 3, 1).contiguous()  # BxHxWx3

    normals: torch.Tensor = torch.linalg.cross(a, b, dim=-1)  # BxHxWx3
    return F.normalize(normals, dim=-1, p=2).to(dtype).permute(0, 3, 1, 2)  # Bx3xHxW


def depth_from_plane_equation(
    plane_normals: torch.Tensor,
    plane_offsets: torch.Tensor,
    points_uv: torch.Tensor,
    camera_matrix: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    r"""Compute the depth of given pixels that lie on a known plane.

    ``depth = depth_from_plane_equation(plane_normals, plane_offsets, points_uv, camera_matrix)`` intersects the
    camera ray of each pixel in ``points_uv`` of shape ``(B, N, 2)`` with the camera-frame plane
    :math:`n \cdot X = d` and returns the ``z`` of each intersection as a ``(B, N)`` tensor: a list of depths,
    not a depth map.

    Convention:
        - the plane is in Hessian form, with the normal ``plane_normals`` of shape ``(B, 3)`` and
          the offset ``plane_offsets`` of shape ``(B, 1)``: ``n = (0, 0, 1)`` with ``d = 2`` is the plane ``z = 2``,
          and every pixel on it has depth 2.
        - ``points_uv`` are **pixels** (integer centres; see
          :class:`~kornia.geometry.camera.pinhole.PinholeCamera`), normalized here with ``camera_matrix``.
        - a ray-plane dot product inside :math:`(-eps, eps)` is replaced by :math:`\pm` ``eps`` with its sign
          (``+eps`` for an exact zero), so a grazing ray returns the large signed depth ``d / eps`` rather than
          ``inf``. In float16 a positive ``eps`` is floored at the smallest subnormal, ``2**-24``, because the
          default ``1e-8`` rounds to zero there; ``d / 2**-24`` is still ``inf`` in float16 once ``|d|`` exceeds
          about ``3.9e-3``, so pass a larger ``eps`` for a finite depth (``eps=1e-4`` keeps ``|d| <= 6.5`` finite).

    Args:
        plane_normals (torch.Tensor): Plane normal vectors of shape ``(B, 3)``.
        plane_offsets (torch.Tensor): Plane offsets of shape ``(B, 1)``.
        points_uv (torch.Tensor): Pixel coordinates of shape ``(B, N, 2)``.
        camera_matrix (torch.Tensor): Camera intrinsic matrix of shape ``(B, 3, 3)``.
        eps: epsilon for numerical stability.

    Returns:
        torch.Tensor: Computed depth values at the given pixels, shape ``(B, N)``.

    See Also:
        - :func:`~kornia.geometry.camera.perspective.unproject_points`: turn these pixels and depths into 3D points.
        - :func:`depth_from_disparity`: a whole depth map from a rectified stereo disparity map, not from a plane.

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
    # A positive eps below float16's smallest subnormal rounds to zero there, which would turn the guard into
    # a division by zero, so floor it at that subnormal (as normalize_quaternion does).
    if denom.dtype == torch.float16 and eps > 0.0:
        eps = max(eps, 5.960464477539063e-08)
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
    return plane_offsets / denom  # plane_offsets: (B, 1), denom: (B, N) -> depth: (B, N)


def warp_frame_depth(
    image_src: torch.Tensor,
    depth_dst: torch.Tensor,
    src_trans_dst: torch.Tensor,
    camera_matrix: torch.Tensor,
    normalize_points: bool = False,
) -> torch.Tensor:
    """Render an image from another viewpoint, given that viewpoint's depth map and the relative pose.

    ``image_dst = warp_frame_depth(image_src, depth_dst, src_trans_dst, camera_matrix)`` shows the source image
    ``(B, D, H, W)`` as the destination camera sees it. Each destination pixel is unprojected with its depth
    from ``depth_dst`` of shape ``(B, 1, H, W)``, moved into the source frame by the ``(B, 4, 4)`` pose
    ``src_trans_dst``, projected with the ``(B, 3, 3)`` intrinsics ``camera_matrix`` that both views share, and
    sampled bilinearly. :class:`DepthWarper` and :func:`depth_warp` do the same warp from two
    :class:`~kornia.geometry.camera.pinhole.PinholeCamera` models; for a planar scene,
    :func:`~kornia.geometry.transform.warp_perspective` with a homography needs no depth.

    Convention:
        - ``src_trans_dst`` maps destination-frame points into the source frame. With ``fx = fy = 1`` and a
          unit depth, a translation of 1 along ``x`` samples ``image_src`` one pixel to the right of each
          destination pixel.
        - the unprojection is :func:`depth_to_3d_v2` and the reprojection
          :func:`~kornia.geometry.camera.perspective.project_points`, whose conventions apply.
        - the sampling is bilinear ``grid_sample`` with a fixed ``align_corners=True`` and
          ``padding_mode="zeros"``; the output keeps ``image_src``'s channels.
        - :class:`DepthWarper` and :func:`depth_warp` name the frames the other way round: this function's
          ``dst`` (the frame holding the depth) is their ``src``, and ``image_src`` is their ``patch_dst``.

    Args:
        image_src: image tensor in the source frame with shape ``(B, D, H, W)``.
        depth_dst: depth tensor in the destination frame with shape ``(B, 1, H, W)``.
        src_trans_dst: transformation matrix from destination to source with shape ``(B, 4, 4)``.
        camera_matrix: tensor containing the camera intrinsics with shape ``(B, 3, 3)``.
        normalize_points: whether to normalize the pointcloud. This must be set to ``True`` when the depth
           is represented as the Euclidean ray length from the camera position.

    Return:
        ``image_src`` resampled onto the destination pixel grid, with shape ``(B, D, H, W)``.

    See Also:
        - :class:`DepthWarper`, :func:`depth_warp`: the same warp from two pinhole camera models.
        - :func:`depth_to_3d_v2`, :func:`~kornia.geometry.camera.perspective.project_points`: the two halves of the
          warp.
        - :func:`~kornia.geometry.transform.warp_perspective`: warp by a homography, without depth.

    """
    KORNIA_CHECK_SHAPE(image_src, ["B", "D", "H", "W"])
    KORNIA_CHECK_SHAPE(depth_dst, ["B", "1", "H", "W"])
    KORNIA_CHECK_SHAPE(src_trans_dst, ["B", "4", "4"])
    KORNIA_CHECK_SHAPE(camera_matrix, ["B", "3", "3"])
    KORNIA_CHECK(
        image_src.shape[-2:] == depth_dst.shape[-2:],
        f"image_src and depth_dst must have the same height and width. "
        f"Got {list(image_src.shape[-2:])} and {list(depth_dst.shape[-2:])}.",
    )

    if (
        image_src.shape[0] == 0
        and depth_dst.shape[0] == 0
        and src_trans_dst.shape[0] == 0
        and camera_matrix.shape[0] == 0
    ):
        output_shape = (0, image_src.shape[1], depth_dst.shape[-2], depth_dst.shape[-1])
        output_zero = image_src.reshape(-1)[:1].sum() * 0.0
        return output_zero.reshape(1, 1, 1, 1).expand(output_shape)

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

    return _torch_grid_sample(image_src, points_2d_src_norm, align_corners=True)


class DepthWarper(nn.Module):
    r"""Render an image from another viewpoint by depth, given two PinholeCamera models (module form).

    Build ``warper = DepthWarper(pinhole_dst, H, W)``, call ``warper.compute_projection_matrix(pinhole_src)``,
    then ``image = warper(depth_src, patch_dst)`` samples the destination image ``patch_dst`` of shape
    ``(B, C, H, W)`` at the pixels where the source-frame depth ``depth_src`` projects, so the ``(B, C, H, W)``
    result shows ``patch_dst`` as the source camera sees it. Calling the warper before
    :meth:`compute_projection_matrix` raises ``ValueError``.
    :func:`depth_warp` is the one-call form; :func:`warp_frame_depth` takes a 4x4 pose and one intrinsics
    matrix instead of two cameras.

    .. math::
        P_{src}^{\{dst\}} = K_{dst} * T_{src}^{\{dst\}}

        I_{src} = \omega(I_{dst}, P_{src}^{\{dst\}}, D_{src})

    Convention:
        - the depth lives in the **source** frame and the image in the **destination** frame.
        - until :meth:`compute_projection_matrix` has been called, :meth:`warp_grid` raises ``ValueError`` and
          :meth:`compute_subpixel_step` raises ``RuntimeError``.
        - :meth:`compute_projection_matrix` stores ``K_dst @ (E_dst @ inverse(E_src))``, the matrix above, from
          the constructor's ``pinhole_dst`` and that method's ``pinhole_src`` (both
          :class:`~kornia.geometry.camera.pinhole.PinholeCamera`, world-to-camera extrinsics).
          Both cameras use only the top-left 3x3 block of their intrinsics, embedded in
          homogeneous 4x4 form; the stored fourth row and column are ignored.
        - the ``grid`` attribute is the homogeneous integer-centre pixel grid (pixel ``(0, 0)`` is
          ``(0, 0, 1)``); :meth:`warp_grid` returns sampling positions in ``grid_sample``'s
          ``align_corners=True`` normalized coordinates of the **destination** image.
        - ``align_corners`` (default ``True``), ``mode`` and ``padding_mode`` are forwarded to ``grid_sample``,
          but :meth:`warp_grid` normalizes for ``align_corners=True`` whatever the flag, so ``False`` misaligns
          the warp and an identity warp no longer returns its input
          (`#6317 <https://github.com/kornia/kornia/issues/6317>`_).
        - :func:`warp_frame_depth` names the frames the other way round: its ``dst`` (the frame holding the depth)
          is this class's ``src``, and its ``image_src`` is ``patch_dst``.

    .. warning::
        The two warps agree to rounding for float32/float64 away from ``z = 0``, and split at ``z = 0``, where
        ``cam2pixel`` sends every pixel with a nonzero projected numerator far outside the image:
        `#4267 <https://github.com/kornia/kornia/issues/4267>`_.

    Args:
        pinhole_dst: the pinhole model for the destination frame.
        height: the height of the image to warp.
        width: the width of the image to warp.
        mode: interpolation mode to calculate output values ``'bilinear'`` | ``'nearest'``.
        padding_mode: padding mode for outside grid values ``'zeros'`` | ``'border'`` | ``'reflection'``.
        align_corners: interpolation flag.

    See Also:
        - :func:`depth_warp`: this warp in one call.
        - :func:`warp_frame_depth`: the same warp from a 4x4 pose and one camera matrix.

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

        See the Convention block on :class:`DepthWarper`.
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

        dst_proj_src = torch.matmul(_embed_intrinsics(self._pinhole_dst.camera_matrix), dst_trans_src)

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

        See the Convention block on :class:`DepthWarper`.

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
        """Compute where each source pixel samples the destination image, given the depth in the source frame.

        See the Convention block on :class:`DepthWarper`.

        The method ``compute_projection_matrix`` has to be called beforehand in order to have precomputed the
        projection matrix, which moves a source-frame point into the destination camera frame and applies the
        destination intrinsics.
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
        """Warp a tensor from the destination frame to the source frame, given the depth in the source frame.

        See the Convention block on :class:`DepthWarper`.

        Args:
            depth_src: the depth in the source frame. The tensor must have a shape ``(B, 1, H, W)``.
            patch_dst: the patch in the destination frame. The tensor must have a shape ``(B, C, H, W)``.

        Return:
            the patch warped from the destination frame to the source frame.

        Shape:
            - Input: ``(B, 1, H, W)`` and ``(B, C, H, W)``.
            - Output: ``(B, C, H, W)`` where C = number of channels.

        Example:
            >>> # pinholes camera models
            >>> pinhole_dst = PinholeCamera(torch.randn(1, 4, 4), torch.randn(1, 4, 4),
            ... torch.tensor([32]), torch.tensor([32]))
            >>> pinhole_src = PinholeCamera(torch.randn(1, 4, 4), torch.randn(1, 4, 4),
            ... torch.tensor([32]), torch.tensor([32]))
            >>> # create the depth warper, compute the projection matrix
            >>> warper = DepthWarper(pinhole_dst, 32, 32)
            >>> _ = warper.compute_projection_matrix(pinhole_src)
            >>> # warp the destination frame to the source frame by depth
            >>> depth_src = torch.ones(1, 1, 32, 32)  # Nx1xHxW
            >>> image_dst = torch.rand(1, 3, 32, 32)  # NxCxHxW
            >>> image_src = warper(depth_src, image_dst)  # NxCxHxW

        """
        return _torch_grid_sample(
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
    """Render an image from another viewpoint by depth, given two PinholeCamera models (one call).

    ``image = depth_warp(pinhole_dst, pinhole_src, depth_src, patch_dst, H, W)`` samples the destination image
    ``patch_dst`` of shape ``(B, C, H, W)`` at the pixels where the source-frame depth ``depth_src`` projects, so the
    result shows ``patch_dst`` as the source camera sees it. It is :class:`DepthWarper` in one call, with that
    class's default ``mode`` and ``padding_mode`` and the same output bit for bit; see its Convention block.
    :func:`warp_frame_depth` takes a 4x4 pose and one intrinsics matrix instead of two cameras.

    Args:
        pinhole_dst: the pinhole model for the destination frame, the one ``patch_dst`` is seen from.
        pinhole_src: the pinhole model for the source frame, the one ``depth_src`` belongs to.
        depth_src: the depth in the source frame, with shape ``(B, 1, H, W)``.
        patch_dst: the image in the destination frame, with shape ``(B, C, H, W)``.
        height: the height of the images, ``H``.
        width: the width of the images, ``W``.
        align_corners: forwarded to ``grid_sample`` only; the sampling grid is always normalized for
            ``align_corners=True``, so ``False`` misaligns the warp
            (`#6317 <https://github.com/kornia/kornia/issues/6317>`_). The default ``True`` is what
            :class:`DepthWarper` uses.

    Return:
        ``patch_dst`` resampled onto the source pixel grid, with shape ``(B, C, H, W)``.

    See Also:
        - :class:`DepthWarper`: the module form, with ``mode`` and ``padding_mode``.
        - :func:`warp_frame_depth`: the same warp from a 4x4 pose and one camera matrix.

    Example:
        >>> # pinholes camera models
        >>> pinhole_dst = PinholeCamera(torch.randn(1, 4, 4), torch.randn(1, 4, 4),
        ... torch.tensor([32]), torch.tensor([32]))
        >>> pinhole_src = PinholeCamera(torch.randn(1, 4, 4), torch.randn(1, 4, 4),
        ... torch.tensor([32]), torch.tensor([32]))
        >>> # warp the destination frame to the source frame by depth
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


def _per_sample_camera_parameter(parameter: float | torch.Tensor, disparity: torch.Tensor) -> float | torch.Tensor:
    """Shape a per-batch-element ``(B,)`` camera parameter so each value meets its own disparity sample.

    A Python number, a scalar tensor and a ``(1,)`` tensor already broadcast over the whole disparity and are
    returned untouched, so every input accepted before ``(B,)`` support existed takes the same path as it did.
    A ``(B,)`` tensor pairs element ``b`` with ``disparity[b]``, which needs a batch axis: ``disparity`` must
    have rank 3 or more and a leading size of ``B``. Anything else raises ``ShapeError``.
    """
    if not isinstance(parameter, torch.Tensor) or parameter.ndim == 0 or parameter.shape == (1,):
        return parameter
    if parameter.ndim == 1 and disparity.ndim >= 3:
        # Compared as a number: a string would turn a symbolic batch size into a graph break under torch.compile.
        KORNIA_CHECK_SIZES(parameter, [disparity.shape[0]])
        return parameter.reshape(disparity.shape[0], *([1] * (disparity.ndim - 1)))
    KORNIA_CHECK_SHAPE(parameter, ["1"])
    return parameter


def depth_from_disparity(
    disparity: torch.Tensor, baseline: float | torch.Tensor, focal: float | torch.Tensor
) -> torch.Tensor:
    """Convert a rectified stereo disparity map to depth.

    ``depth = depth_from_disparity(disparity, baseline, focal)`` returns ``baseline * focal / disparity`` for every
    pixel of a ``(*, H, W)`` disparity map, with ``baseline`` the distance between the two camera centres and
    ``focal`` the focal length in pixels, so the depth comes out in the unit of ``baseline``. A zero disparity
    gives a large finite depth, not ``inf``, except in float16 (see below). To go from disparity straight to 3D
    points with a calibrated stereo rig, use
    :meth:`~kornia.geometry.camera.stereo.StereoCamera.reproject_disparity_to_3D`, which takes a ``(B, H, W, 1)``
    disparity.

    Convention:
        - the exact formula is ``baseline * focal / (disparity + 1e-8)``: a disparity of 2 with a baseline of
          0.5 and a focal length of 100 gives a depth of 25. The ``1e-8`` lowers the depth by a
          relative ``1e-8 / disparity``. Float32 rounds that away from a disparity of 0.25 up; below it the bias
          shows, ``1e-6`` relative (10 float32 ulps) at a disparity of 0.01.
        - a zero disparity (no stereo match, or a point at infinity) gives a finite far depth,
          ``baseline * focal / 1e-8`` (``5e9`` for baseline 0.5 and focal 100), not ``inf``. Such a point warps
          as a point at infinity, along the rotation-only homography, in
          :func:`warp_frame_depth`, whereas an ``inf`` depth would turn into ``NaN`` in
          :func:`depth_to_3d` wherever a ray component is zero. The far depth is a
          placeholder, not a measurement: mask ``disparity > 0`` to tell matched pixels apart.
        - in float16 the ``1e-8`` rounds to ``0`` and the far depth would overflow anyway, so a zero disparity
          gives ``inf``. A float32 far depth turns back into ``inf`` wherever the warp casts it to float16, as a
          float16 pose or float16 autocast does, so run the whole warp (depth, image, pose and camera matrix) in
          float32, outside float16 autocast, when zero disparities flow into it.
        - ``baseline`` and ``focal`` are each a scalar (Python number, 0-d or ``(1,)`` tensor) shared by the
          whole disparity, or a ``(B,)`` tensor paired with ``disparity[b]``, which then needs a leading
          batch axis of size ``B`` (otherwise ``ShapeError``). The two may mix.
        - ``disparity`` is ``(*, H, W)`` and the result has its shape. Its sign is not checked.

    Args:
        disparity: Disparity tensor of shape ``(*, H, W)``.
        baseline: Distance between the two lenses, as an int, float, or tensor of shape ``()`` or ``(1,)``,
            or of shape ``(B,)`` for one value per batch element.
        focal: Focal length, as an int, float, or tensor of shape ``()`` or ``(1,)``, or of shape ``(B,)`` for
            one value per batch element.

    Return:
        Depth map of the shape ``(*, H, W)``.

    See Also:
        - :func:`depth_to_3d`, :func:`depth_to_3d_v2`: unproject the depth map to 3D points.
        - :meth:`~kornia.geometry.camera.stereo.StereoCamera.reproject_disparity_to_3D`: disparity to 3D points in one
          step.
        - :func:`depth_from_plane_equation`: the depth of given pixels on a known plane, ``(B, N)``, not a map.

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
        isinstance(baseline, (int, float, torch.Tensor)) and not isinstance(baseline, bool),
        f"Input baseline should be an int, float or torch.Tensor. Got {type(baseline)}",
    )
    KORNIA_CHECK(
        isinstance(focal, (int, float, torch.Tensor)) and not isinstance(focal, bool),
        f"Input focal should be an int, float or torch.Tensor. Got {type(focal)}",
    )

    baseline = _per_sample_camera_parameter(baseline, disparity)
    focal = _per_sample_camera_parameter(focal, disparity)

    return baseline * focal / (disparity + 1e-8)
