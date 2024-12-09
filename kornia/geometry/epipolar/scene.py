"""Module to generate synthetic 3d scenes."""

import math
from typing import Dict

import torch

from kornia.core import rand, stack, where
from kornia.geometry.conversions import axis_angle_to_rotation_matrix
from kornia.geometry.linalg import transform_points

from .projection import projection_from_KRt, random_intrinsics


def generate_scene(num_views: int, num_points: int) -> Dict[str, torch.Tensor]:
    # Generate the 3d points
    points3d = rand(1, num_points, 3)  # NxMx3

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
    tvec = stack([tx, ty, tz], dim=1)[..., None]

    # Make sure the shape is in front of the camera
    points3d_trans = (rot_mat @ points3d.transpose(-2, -1)) + tvec
    min_dist = torch.min(points3d_trans[:, 2], dim=1)[0]
    tvec[:, 2, 0] = where(min_dist < 0, tz - min_dist + 1.0, tz)

    # compute projection matrices
    P = projection_from_KRt(K, rot_mat, tvec)

    # project points3d and backproject to image plane
    points2d = transform_points(P, points3d.expand(num_views, -1, -1))

    return {"K": K, "R": rot_mat, "t": tvec, "P": P, "points3d": points3d, "points2d": points2d}
