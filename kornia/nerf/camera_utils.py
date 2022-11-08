import math
from typing import List

import torch

from kornia.core import tensor
from kornia.geometry.camera import PinholeCamera


def cameras_for_ids(cameras: PinholeCamera, camera_ids: List[int]) -> PinholeCamera:
    r"""Takes a PinholeCamera camera object and a set of camera indices and creates a new PinholeCamera object for
    the requested cameras.

    Args:
        cameras: Scene camera object: PinholeCamera
        camera_ids: List of camera indices to copy: List[int]

    Return:
        A new PinholeCamera object with a sub-set of cameras: PinholeCamera
    """
    intrinsics = cameras.intrinsics[camera_ids]
    extrinsics = cameras.extrinsics[camera_ids]
    height = cameras.height[camera_ids]
    width = cameras.width[camera_ids]
    return PinholeCamera(intrinsics, extrinsics, height, width)


def create_spiral_path(cameras: PinholeCamera, rad: float, num_views: int, num_circles: int) -> PinholeCamera:
    r"""Creates a PinholeCamera object with cameras that follow a spiral path. Used for novel view synthesis for
    face facing models.

    Args:
        cameras: Scene cameras used to train the NeRF model: PinholeCamera
        rad: Spiral radius: float
        num_views: Number of created cameras: int
        num_circles: Number of spiral circles: int
    """

    # Average locations over all cameras
    mean_center = cameras.translation_vector.mean(0, False).squeeze(-1)
    device = cameras.intrinsics.device
    t = torch.linspace(0, 2 * math.pi * num_circles, num_views, device=device)
    cos_t = torch.cos(t) * rad
    sin_t = -torch.sin(t) * rad
    sin_05t = -torch.sin(0.5 * t) * rad
    translation_vector = torch.unsqueeze(mean_center, dim=0) + torch.stack((cos_t, sin_t, sin_05t)).permute((1, 0))
    mean_intrinsics = cameras.intrinsics.mean(0, True).repeat(num_views, 1, 1)
    mean_extrinsics = cameras.extrinsics.mean(0, True).repeat(num_views, 1, 1)
    extrinsics = mean_extrinsics
    extrinsics[:, :3, 3] = translation_vector
    height = tensor([cameras.height[0]] * num_views, device=device)
    width = tensor([cameras.width[0]] * num_views, device=device)
    return PinholeCamera(mean_intrinsics, extrinsics, height, width)
