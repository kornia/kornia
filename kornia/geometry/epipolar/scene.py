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

"""Generate synthetic 3d scenes."""

import math
from typing import Dict

import torch

from kornia.geometry.conversions import axis_angle_to_rotation_matrix
from kornia.geometry.linalg import transform_points

from .projection import projection_from_KRt, random_intrinsics


def generate_scene(num_views: int, num_points: int) -> Dict[str, torch.Tensor]:
    """Generate a random 3d scene seen by ``num_views`` cameras that share one camera matrix.

    Convention:
        - Every point has positive depth in every view; depth and focal length are not bounded away from zero.
        - Draws from torch's global generator (see :doc:`Conventions & Pitfalls </get-started/conventions>`).

    Args:
        num_views: the number of views :math:`V`.
        num_points: the number of 3d points :math:`N`.

    Returns:
        A dictionary of tensors on the CPU in the default dtype:

        - ``"K"``: the camera matrix shared by all views, with shape :math:`(1, 3, 3)`.
        - ``"R"``: the rotation of each view, with shape :math:`(V, 3, 3)`.
        - ``"t"``: the translation of each view, with shape :math:`(V, 3, 1)`.
        - ``"P"``: the projection matrix :math:`K [R|t]` of each view, with shape :math:`(V, 3, 4)`.
        - ``"points3d"``: the 3d points, in :math:`[0, 1)` per coordinate, with shape :math:`(1, N, 3)`.
        - ``"points2d"``: the projection of ``points3d`` by each ``P``, with shape :math:`(V, N, 2)`.

    """
    # Generate the 3d points
    points3d = torch.rand(1, num_points, 3)  # NxMx3

    # Create random camera matrix
    K = random_intrinsics(0.0, 100.0)  # 1x3x3

    # Create random rotation per view
    ang = torch.rand(num_views, 1) * math.pi * 2.0

    rvec = torch.rand(num_views, 3)
    rvec = ang * rvec / torch.norm(rvec, dim=1, keepdim=True)  # Nx3
    rot_mat = axis_angle_to_rotation_matrix(rvec)  # Nx3x3
    # matches with cv2.Rodrigues -> yay !

    # Create random translation per view
    tx = torch.empty(num_views).uniform_(-0.5, 0.5)
    ty = torch.empty(num_views).uniform_(-0.5, 0.5)
    tz = torch.empty(num_views).uniform_(-1.0, 2.0)
    tvec = torch.stack([tx, ty, tz], dim=1)[..., None]

    # Make sure the shape is in front of the camera
    points3d_trans = (rot_mat @ points3d.transpose(-2, -1)) + tvec
    min_dist = torch.min(points3d_trans[:, 2], dim=1)[0]
    tvec[:, 2, 0] = torch.where(min_dist < 0, tz - min_dist + 1.0, tz)

    # compute projection matrices
    P = projection_from_KRt(K, rot_mat, tvec)

    # project points3d and backproject to image plane
    points2d = transform_points(P, points3d.expand(num_views, -1, -1))

    return {"K": K, "R": rot_mat, "t": tvec, "P": P, "points3d": points3d, "points2d": points2d}
