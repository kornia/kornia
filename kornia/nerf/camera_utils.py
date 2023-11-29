import math
from typing import List, Tuple, Union

import torch

from kornia.core import Device, Tensor, cos, sin, stack
from kornia.geometry.camera import PinholeCamera
from kornia.geometry.conversions import quaternion_to_rotation_matrix


def parse_colmap_output(
    cameras_path: str, images_path: str, device: Device, dtype: torch.dtype
) -> Tuple[List[str], PinholeCamera]:
    r"""Parses colmap output to create an PinholeCamera for aligned scene cameras.

    Args:
        cameras_path: Path to camera.txt Colmap file with camera intrinsics: str
        images_path: Path to images.txt Colmap file with camera extrinsics for each image: str
        device: device for created camera object: Union[str, torch.device]

    Returns:
        image names: List[str]
        scene camera object: PinholeCamera
    """

    # Parse camera intrinsics
    with open(cameras_path) as f:
        lines = f.readlines()

    class CameraParams:
        def __init__(self, line: str) -> None:
            split_line = line.split(" ")
            model = split_line[1]
            if model == "SIMPLE_PINHOLE":
                self._width = int(split_line[2])
                self._height = int(split_line[3])
                self._fx = float(split_line[4])
                self._fy = self._fx
                self._cx = int(split_line[5])
                self._cy = int(split_line[6])
            elif model == "PINHOLE":
                self._width = int(split_line[2])
                self._height = int(split_line[3])
                self._fx = float(split_line[4])
                self._fy = float(split_line[5])
                self._cx = int(split_line[6])
                self._cy = int(split_line[7])

    cameras_params: List[CameraParams] = []
    for line in lines:
        if line.startswith("#"):
            continue
        camera_params = CameraParams(line)
        cameras_params.append(camera_params)

    # Parse camera quaternions and translation vectors
    with open(images_path) as f:
        lines = f.readlines()
    intrinsics: List[Tensor] = []
    extrinsics: List[Tensor] = []
    heights: List[int] = []
    widths: List[int] = []
    img_names: List[str] = []
    for line in lines:
        if line.startswith("#"):
            continue

        # Read line with camera quaternion
        line = line.strip()
        if line.endswith(("jpg", "png")):
            split_line = line.split(" ")
            qw = float(split_line[1])
            qx = float(split_line[2])
            qy = float(split_line[3])
            qz = float(split_line[4])
            tx = float(split_line[5])
            ty = float(split_line[6])
            tz = float(split_line[7])
            camera_ind = int(split_line[8]) - 1
            img_name = split_line[9]
            img_names.append(img_name)

            # Intrinsic
            camera_params = cameras_params[camera_ind]
            intrinsic = torch.eye(4, device=device, dtype=dtype)
            intrinsic[0, 0] = camera_params._fx
            intrinsic[1, 1] = camera_params._fy
            intrinsic[0, 2] = camera_params._cx
            intrinsic[1, 2] = camera_params._cy
            intrinsics.append(intrinsic)

            heights.append(camera_params._height)
            widths.append(camera_params._width)

            # Extrinsic
            q = torch.tensor([qw, qx, qy, qz], device=device)
            R = quaternion_to_rotation_matrix(q)
            t = torch.tensor([tx, ty, tz], device=device)
            extrinsic = torch.eye(4, device=device, dtype=dtype)
            extrinsic[:3, :3] = R
            extrinsic[:3, 3] = t
            extrinsics.append(extrinsic)

    cameras = PinholeCamera(
        torch.stack(intrinsics),
        torch.stack(extrinsics),
        torch.tensor(heights, device=device),
        torch.tensor(widths, device=device),
    )
    return img_names, cameras


def cameras_for_ids(cameras: PinholeCamera, camera_ids: Union[List[int], Tensor]) -> PinholeCamera:
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
    cos_t = cos(t) * rad
    sin_t = -sin(t) * rad
    sin_05t = -sin(0.5 * t) * rad
    translation_vector = torch.unsqueeze(mean_center, dim=0) + stack((cos_t, sin_t, sin_05t)).permute((1, 0))
    mean_intrinsics = cameras.intrinsics.mean(0, True).repeat(num_views, 1, 1)
    mean_extrinsics = cameras.extrinsics.mean(0, True).repeat(num_views, 1, 1)
    extrinsics = mean_extrinsics
    extrinsics[:, :3, 3] = translation_vector
    height = torch.tensor([cameras.height[0]] * num_views, device=device)
    width = torch.tensor([cameras.width[0]] * num_views, device=device)
    return PinholeCamera(mean_intrinsics, extrinsics, height, width)
