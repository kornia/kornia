import math
from typing import List

import torch
from scipy.interpolate import splev, splprep

from kornia.core import Tensor
from kornia.geometry.camera import PinholeCamera
from kornia.geometry.conversions import (
    QuaternionCoeffOrder,
    quaternion_to_rotation_matrix,
    rotation_matrix_to_quaternion,
)


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
    height = torch.tensor([cameras.height[0]] * num_views, device=device)
    width = torch.tensor([cameras.width[0]] * num_views, device=device)
    return PinholeCamera(mean_intrinsics, extrinsics, height, width)


def create_spline_curve(cameras: PinholeCamera, num_views: int) -> PinholeCamera:
    r"""Create a PinholeCamera object with cameras that are positioned along a path that is calculated from the
    locations of the input camera centers. Parameteric spline is used for the tranjectory. Output camera intrinsics
    are also interpolated from those of the input cameras.

    Args:
        cameras: Scene cameras used to train the NeRF model: PinholeCamera
        num_views: Number of created cameras: int
    """
    # Extrinsics
    origins = cameras.origins().cpu().squeeze().numpy()
    ox = origins[:, 0]
    oy = origins[:, 1]
    oz = origins[:, 2]

    rotation_matrix = torch.clone(cameras.rotation_matrix).cpu()  # Cloning to restore memory contingency
    q = rotation_matrix_to_quaternion(rotation_matrix, order=QuaternionCoeffOrder.WXYZ).numpy()
    qw = q[:, 0]
    qx = q[:, 1]
    qy = q[:, 2]
    qz = q[:, 3]

    # Intrinsics
    fx_new = torch.mean(cameras.fx).item()  # Use average of camera focals
    fy_new = torch.mean(cameras.fy).item()
    width_new = cameras.width[0].item()  # For new camera sizes use first input camera
    height_new = cameras.height[0].item()
    cx_new = width_new / 2.0  # Assign principal point to mid-image
    cy_new = height_new / 2.0

    # Interpolate output camera extrinsic parameters
    tck, _ = splprep([ox, oy, oz, qw, qx, qy, qz], s=0)
    u2 = torch.linspace(0, 1, num_views, device='cpu', dtype=cameras.dtype).numpy()
    new_points = splev(u2, tck)

    device = cameras.device
    dtype = cameras.dtype
    intrinsics: List[Tensor] = []
    extrinsics: List[Tensor] = []
    for i in range(num_views):
        intrinsic = torch.eye(4, device=device, dtype=dtype)
        intrinsic[0, 0] = fx_new
        intrinsic[1, 1] = fy_new
        intrinsic[0, 2] = cx_new
        intrinsic[1, 2] = cy_new
        intrinsics.append(intrinsic)

        q_new = torch.tensor([new_points[3][i], new_points[4][i], new_points[5][i], new_points[6][i]], device=device)
        R_new = quaternion_to_rotation_matrix(q_new, order=QuaternionCoeffOrder.WXYZ)
        o_new = torch.tensor([new_points[0][i], new_points[1][i], new_points[2][i]], device=device)
        t_new = -R_new @ o_new
        extrinsic = torch.eye(4, device=device, dtype=dtype)
        extrinsic[:3, :3] = R_new
        extrinsic[:3, 3] = t_new
        extrinsics.append(extrinsic)
    cameras_new = PinholeCamera(
        torch.stack(intrinsics),
        torch.stack(extrinsics),
        torch.tensor([height_new] * num_views, device=device),
        torch.tensor([width_new] * num_views, device=device),
    )
    return cameras_new
