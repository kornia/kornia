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

import math
from typing import List, Tuple, Union

import torch

from kornia.core import Device, Tensor, cos, sin, stack
from kornia.geometry.camera import PinholeCamera
from kornia.geometry.conversions import quaternion_to_rotation_matrix


def parse_colmap_output(
    cameras_path: str, images_path: str, device: Device, dtype: torch.dtype
) -> Tuple[List[str], PinholeCamera]:
    r"""Parse colmap output to create an PinholeCamera for aligned scene cameras.

    Args:
        cameras_path: Path to camera.txt Colmap file with camera intrinsics: str
        images_path: Path to images.txt Colmap file with camera extrinsics for each image: str
        device: device for created camera object: Union[str, torch.device]
        dtype: Intrinsics and extrinsics dtype.

    Returns:
        image names: List[str]
        scene camera object: PinholeCamera

    """
    # Parse camera intrinsics
    with open(cameras_path) as f:
        lines = [line.strip() for line in f if not line.startswith("#")]
    class CameraParams:
        def __init__(self, line: str) -> None:
            split_line = line.split(" ")
            if len(split_line) < 7:
                raise ValueError(f"Invalid camera line: {line}")
            model = split_line[1]
            if model == "SIMPLE_PINHOLE":
                self.width = int(split_line[2])
                self.height = int(split_line[3])
                self.fx = float(split_line[4])
                self.fy = self.fx
                self.cx = float(split_line[5])
                self.cy = float(split_line[6])
            elif model == "PINHOLE":
                if len(split_line) < 8:
                    raise ValueError(f"Invalid PINHOLE camera line: {line}")
                self.width = int(split_line[2])
                self.height = int(split_line[3])
                self.fx = float(split_line[4])
                self.fy = float(split_line[5])
                self.cx = float(split_line[6])
                self.cy = float(split_line[7])
            else:
                raise ValueError(f"Unsupported camera model: {model}")
    cameras_params: List[CameraParams] = [CameraParams(line) for line in lines]
    with open(images_path) as f:
        lines = [l for l in (line.strip() for line in f if not line.startswith("#")) if l.endswith(("jpg", "png"))]
    num_images = len(lines)
    if num_images == 0:
        raise ValueError("No valid images found in images.txt")
    img_names: List[str] = []
    camera_inds: List[int] = []
    quats_list: List[List[float]] = []
    ts_list: List[List[float]] = []
    for line in lines:
        split_line = line.split(" ")
        if len(split_line) < 10:
            raise ValueError(f"Invalid image line: {line}")
        qw, qx, qy, qz = map(float, split_line[1:5])
        tx, ty, tz = map(float, split_line[5:8])
        camera_ind = int(split_line[8]) - 1
        img_name = split_line[9]
        if camera_ind >= len(cameras_params):
            raise ValueError(f"Invalid camera index {camera_ind + 1} for image {img_name}")
        img_names.append(img_name)
        camera_inds.append(camera_ind)
        quats_list.append([qw, qx, qy, qz])
        ts_list.append([tx, ty, tz])
    quats = torch.tensor(quats_list, device=device, dtype=dtype)
    ts = torch.tensor(ts_list, device=device, dtype=dtype)
    Rs = quaternion_to_rotation_matrix(quats)
    extrinsics = torch.eye(4, device=device, dtype=dtype).unsqueeze(0).repeat(num_images, 1, 1)
    extrinsics[:, :3, :3] = Rs
    extrinsics[:, :3, 3] = ts
    fxs = torch.tensor([cameras_params[i].fx for i in camera_inds], device=device, dtype=dtype)
    fys = torch.tensor([cameras_params[i].fy for i in camera_inds], device=device, dtype=dtype)
    cxs = torch.tensor([cameras_params[i].cx for i in camera_inds], device=device, dtype=dtype)
    cys = torch.tensor([cameras_params[i].cy for i in camera_inds], device=device, dtype=dtype)
    intrinsics = torch.eye(4, device=device, dtype=dtype).unsqueeze(0).repeat(num_images, 1, 1)
    intrinsics[:, 0, 0] = fxs
    intrinsics[:, 1, 1] = fys
    intrinsics[:, 0, 2] = cxs
    intrinsics[:, 1, 2] = cys
    heights = torch.tensor([cameras_params[i].height for i in camera_inds], device=device)
    widths = torch.tensor([cameras_params[i].width for i in camera_inds], device=device)
    cameras = PinholeCamera(
        intrinsics,
        extrinsics,
        heights,
        widths,
    )
    return img_names, cameras


def cameras_for_ids(cameras: PinholeCamera, camera_ids: Union[List[int], Tensor]) -> PinholeCamera:
    r"""Take a PinholeCamera camera and camera indices to create a new PinholeCamera for requested cameras.

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
    r"""Create a PinholeCamera object with cameras that follow a spiral path.

    Used for novel view synthesis for face facing models.

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
