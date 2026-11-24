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

"""Rectify calibrated stereo pairs onto a shared horizontal pixel grid."""

from typing import NamedTuple, cast

import torch

from kornia.core.check import KORNIA_CHECK_SHAPE
from kornia.core.utils import _autocast_disabled
from kornia.geometry.grid import create_meshgrid

from .distort import _distort_normalized

__all__ = ["StereoRectification", "stereo_rectify"]


class StereoRectification(NamedTuple):
    """Sampling maps, masks, and camera matrices of a horizontally rectified pair.

    See Also:
        - :func:`~kornia.geometry.calibration.stereo_rectify`: build the maps and matrices from two calibrated cameras.
    """

    map_left: torch.Tensor
    map_right: torch.Tensor
    valid_left: torch.Tensor
    valid_right: torch.Tensor
    projection_left: torch.Tensor
    projection_right: torch.Tensor
    rotation_left: torch.Tensor
    rotation_right: torch.Tensor


def _rectification_map(
    rays: torch.Tensor,
    rotation: torch.Tensor,
    intrinsics: torch.Tensor,
    distortion: torch.Tensor,
    rig_valid: torch.Tensor,
    image_size: tuple[int, int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Project rectified rays into one original, distorted camera."""
    height, width = image_size
    # Row vectors multiply the rectification rotation to apply its inverse.
    original = rays @ rotation
    z = original[..., 2:3]
    front = (z > 1e-8) & rig_valid[:, None, None]
    normalized = torch.where(front, original[..., :2], 0) / torch.where(front, z, 1)
    x, y = normalized.unbind(-1)
    xd, yd, valid = _distort_normalized(x, y, distortion, front.squeeze(-1))
    u = xd * intrinsics[:, 0:1, 0] + intrinsics[:, 0:1, 2]
    v = yd * intrinsics[:, 1:2, 1] + intrinsics[:, 1:2, 2]
    # Integer-centre pixels cover a half-pixel on either side of their centre.
    visible = cast(torch.Tensor, valid) & (u >= -0.5) & (u < width - 0.5) & (v >= -0.5) & (v < height - 0.5)
    coords = torch.where(visible[..., None], torch.stack((u, v), dim=-1), 0)
    return coords.reshape(rotation.shape[0], height, width, 2), visible.reshape(rotation.shape[0], height, width)


def stereo_rectify(
    K_left: torch.Tensor,
    K_right: torch.Tensor,
    dist_left: torch.Tensor,
    dist_right: torch.Tensor,
    R_right_left: torch.Tensor,
    t_right_left: torch.Tensor,
    K_rectified: torch.Tensor,
    image_size: tuple[int, int],
) -> StereoRectification:
    """Build differentiable rectification maps and projection matrices for a calibrated stereo pair.

    Both views use the caller's shared rectified intrinsics. The common x axis points from the left optical
    centre to the right. The common z axis is the projection of the mean optical axis perpendicular to x.
    In this frame, a point visible in both rectified images has equal vertical coordinates and positive
    left-minus-right disparity. Each output pixel has a backward map into its original distorted image.

    Convention:
        - The pose maps left-camera coordinates to right-camera coordinates, X_right = R X_left + t.
        - Maps have shape (B, H, W, 2) in integer-centre pixel coordinates (column, row). Split the last
          dimension to pass them to remap with align_corners=True.
        - Boolean masks of shape (B, H, W) identify nonsingular front-facing rays inside the source pixel area.
          Invalid locations contain zero coordinates. Use the mask to suppress their sampled values.
        - Projection matrices of shape (B, 3, 4) use the rectified left-camera frame, as StereoCamera expects.
          The rotations map vectors from the respective original camera frames to their rectified frames.
        - A zero baseline or an undefined common optical axis yields all-false masks and zero matrices.
        - All tensors must be finite and on the same device. Computation takes place outside autocast in
          float64 if any input is float64, otherwise float32.

    Args:
        K_left: Left pinhole intrinsics (B, 3, 3), with positive focal lengths and zero skew.
        K_right: Right pinhole intrinsics (B, 3, 3), with positive focal lengths and zero skew.
        dist_left: Left Brown-Conrady coefficients (B, N), as in distort_points.
        dist_right: Right Brown-Conrady coefficients (B, N), as in distort_points.
        R_right_left: Right-from-left rotation matrices (B, 3, 3).
        t_right_left: Right-from-left translations (B, 3).
        K_rectified: Common pinhole intrinsics (B, 3, 3), with positive focal lengths and zero skew.
        image_size: Height and width of both source and rectified images.

    Returns:
        Rectification maps, validity masks, projection matrices and rotations.

    See Also:
        - :func:`~kornia.geometry.calibration.undistort_image`: remove distortion from one image.
        - :func:`~kornia.geometry.transform.remap`: sample an image with the returned pixel maps.
    """
    for matrix in (K_left, K_right, R_right_left, K_rectified):
        KORNIA_CHECK_SHAPE(matrix, ["B", "3", "3"])
    KORNIA_CHECK_SHAPE(t_right_left, ["B", "3"])
    for dist in (dist_left, dist_right):
        KORNIA_CHECK_SHAPE(dist, ["B", "N"])

    height, width = image_size
    if height <= 0 or width <= 0:
        raise ValueError("Image height and width must be positive.")
    inputs = (K_left, K_right, dist_left, dist_right, R_right_left, t_right_left, K_rectified)
    if any(t.shape[0] != K_left.shape[0] for t in inputs):
        raise ValueError("All calibration tensors must have the same batch size.")
    if any(t.device != K_left.device or not t.is_floating_point() for t in inputs):
        raise ValueError("All calibration tensors must be floating point and on the same device.")

    dtype = torch.float64 if any(t.dtype == torch.float64 for t in inputs) else torch.float32
    K_left, K_right, dist_left, dist_right, R, t, K_rectified = (v.to(dtype) for v in inputs)
    with _autocast_disabled(K_left.device.type):
        centre_right = (-R.mT @ t.unsqueeze(-1)).squeeze(-1)
        baseline = torch.linalg.vector_norm(centre_right, dim=-1, keepdim=True)
        x = centre_right / torch.where(baseline > 0, baseline, 1)
        optical_axis = R[..., 2, :] + R.new_tensor([0, 0, 1])
        y = torch.linalg.cross(optical_axis, x)
        y_length = torch.linalg.vector_norm(y, dim=-1, keepdim=True)
        rig_valid = ((baseline > 0) & (y_length > 1e-6)).squeeze(-1)
        y = y / torch.where(rig_valid[:, None], y_length, 1)
        rotation_left = torch.stack((x, y, torch.linalg.cross(x, y)), dim=-2)
        rotation_left = torch.where(rig_valid[:, None, None], rotation_left, 0)
        rotation_right = rotation_left @ R.mT

        grid = create_meshgrid(height, width, normalized_coordinates=False, device=K_left.device, dtype=dtype)
        u, v = grid.reshape(1, height * width, 2).unbind(-1)
        u = (u - K_rectified[:, 0:1, 2]) / K_rectified[:, 0:1, 0]
        v = (v - K_rectified[:, 1:2, 2]) / K_rectified[:, 1:2, 1]
        rays = torch.stack((u, v, torch.ones_like(u)), dim=-1)
        map_left, valid_left = _rectification_map(rays, rotation_left, K_left, dist_left, rig_valid, image_size)
        map_right, valid_right = _rectification_map(rays, rotation_right, K_right, dist_right, rig_valid, image_size)

        K_rectified = torch.where(rig_valid[:, None, None], K_rectified, 0)
        left_column = torch.zeros_like(K_rectified[..., :1])
        right_column = -baseline[..., None] * K_rectified[..., :1]
        projection_left = torch.cat((K_rectified, left_column), dim=-1)
        projection_right = torch.cat((K_rectified, right_column), dim=-1)
        return StereoRectification(
            map_left,
            map_right,
            valid_left,
            valid_right,
            projection_left,
            projection_right,
            rotation_left,
            rotation_right,
        )
