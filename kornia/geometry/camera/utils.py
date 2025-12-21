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

from __future__ import annotations

import math

import torch
from torch import Tensor

from kornia.core import Device
from kornia.geometry.camera import PinholeCamera


def create_camera_dimensions(
    device: Device, dtype: torch.dtype, n_cams1: int = 3, n_cams2: int = 2
) -> tuple[Tensor, Tensor, Tensor]:
    """Create camera dimensions for ray sampling.

    Args:
        device: Device for tensors
        dtype: Data type for tensors
        n_cams1: Number of cameras in first group (default: 3)
        n_cams2: Number of cameras in second group (default: 2)

    Returns:
        Tuple of (heights, widths, num_img_rays) tensors
    """
    heights: torch.Tensor = torch.cat(
        (
            torch.tensor([200] * n_cams1, device=device, dtype=dtype),
            torch.tensor([100] * n_cams2, device=device, dtype=dtype),
        )
    )
    widths: torch.Tensor = torch.cat(
        (
            torch.tensor([300] * n_cams1, device=device, dtype=dtype),
            torch.tensor([400] * n_cams2, device=device, dtype=dtype),
        )
    )
    num_img_rays: torch.Tensor = torch.cat(
        (
            torch.tensor([10] * n_cams1, device=device, dtype=dtype),
            torch.tensor([15] * n_cams2, device=device, dtype=dtype),
        )
    )
    return heights, widths, num_img_rays


def create_intrinsics(
    fxs: list[float | int],
    fys: list[float | int],
    cxs: Tensor | list[float | int],
    cys: Tensor | list[float | int],
    device: Device,
    dtype: torch.dtype,
) -> Tensor:
    """Create intrinsic camera matrices from focal lengths and principal points.

    Args:
        fxs: Focal length in x direction
        fys: Focal length in y direction
        cxs: Principal point x coordinate
        cys: Principal point y coordinate
        device: Device for tensors
        dtype: Data type for tensors

    Returns:
        Stacked intrinsic matrices of shape (N, 4, 4)
    """
    intrinsics_batch: list[Tensor] = []
    # Convert cxs and cys to lists if they are tensors
    cxs_list = cxs.tolist() if isinstance(cxs, Tensor) else cxs
    cys_list = cys.tolist() if isinstance(cys, Tensor) else cys
    for fx, fy, cx, cy in zip(fxs, fys, cxs_list, cys_list):
        intrinsics = torch.eye(4, device=device, dtype=dtype)
        intrinsics[0, 0] = fx
        intrinsics[1, 1] = fy
        intrinsics[0, 2] = cx
        intrinsics[1, 2] = cy
        intrinsics_batch.append(intrinsics)
    return torch.stack(intrinsics_batch)


def create_extrinsics_with_rotation(
    alphas: list[float],
    betas: list[float],
    gammas: list[float],
    txs: list[float],
    tys: list[float],
    tzs: list[float],
    device: Device,
    dtype: torch.dtype,
) -> Tensor:
    """Create extrinsic camera matrices with rotation and translation.

    Args:
        alphas: Rotation angles around x-axis
        betas: Rotation angles around y-axis
        gammas: Rotation angles around z-axis
        txs: Translation in x direction
        tys: Translation in y direction
        tzs: Translation in z direction
        device: Device for tensors
        dtype: Data type for tensors

    Returns:
        Stacked extrinsic matrices of shape (N, 4, 4)
    """
    extrinsics_batch: list[Tensor] = []
    for alpha, beta, gamma, tx, ty, tz in zip(alphas, betas, gammas, txs, tys, tzs):
        Rx = torch.eye(3, device=device, dtype=dtype)
        Rx[1, 1] = math.cos(alpha)
        Rx[1, 2] = math.sin(alpha)
        Rx[2, 1] = -Rx[1, 2]
        Rx[2, 2] = Rx[1, 1]

        Ry = torch.eye(3, device=device, dtype=dtype)
        Ry[0, 0] = math.cos(beta)
        Ry[0, 2] = -math.sin(beta)
        Ry[2, 0] = -Ry[0, 2]
        Ry[2, 2] = Ry[0, 0]

        Rz = torch.eye(3, device=device, dtype=dtype)
        Rz[0, 0] = math.cos(gamma)
        Rz[0, 1] = math.sin(gamma)
        Rz[1, 0] = -Rz[0, 1]
        Rz[1, 1] = Rz[0, 0]

        Ryz = torch.matmul(Ry, Rz)
        R = torch.matmul(Rx, Ryz)

        extrinsics = torch.eye(4, device=device, dtype=dtype)
        extrinsics[..., 0, -1] = tx
        extrinsics[..., 1, -1] = ty
        extrinsics[..., 2, -1] = tz
        extrinsics[:3, :3] = R

        extrinsics_batch.append(extrinsics)
    return torch.stack(extrinsics_batch)


def create_pinhole_camera(height: float, width: float, device: Device, dtype: torch.dtype) -> PinholeCamera:
    """Create a single PinholeCamera with default parameters.

    Args:
        height: Camera image height
        width: Camera image width
        device: Device for tensors
        dtype: Data type for tensors

    Returns:
        PinholeCamera: A PinholeCamera instance
    """
    fx = width
    fy = height
    cx = (width - 1.0) / 2.0
    cy = (height - 1.0) / 2.0

    tx = 0.0
    ty = 0.0
    tz = 1.0

    alpha = math.pi / 2.0
    beta = 0.0
    gamma = -math.pi / 2.0

    intrinsics = create_intrinsics([fx], [fy], [cx], [cy], device=device, dtype=dtype)
    extrinsics = create_extrinsics_with_rotation([alpha], [beta], [gamma], [tx], [ty], [tz], device=device, dtype=dtype)

    return PinholeCamera(
        intrinsics,
        extrinsics,
        torch.tensor([height], device=device, dtype=dtype),
        torch.tensor([width], device=device, dtype=dtype),
    )


def create_four_cameras(device: Device, dtype: torch.dtype) -> PinholeCamera:
    """Create four PinholeCameras with predefined parameters.

    Args:
        device: Device for tensors
        dtype: Data type for tensors

    Returns:
        PinholeCamera: A PinholeCamera instance with 4 cameras in batch
    """
    height = torch.tensor([5, 4, 4, 4], device=device, dtype=dtype)
    width = torch.tensor([9, 7, 7, 7], device=device, dtype=dtype)

    fx = width.tolist()
    fy = height.tolist()

    cx = (width - 1.0) / 2.0
    cy = (height - 1.0) / 2.0

    tx = [0.0, 0.0, 0.0, 0.0]
    ty = [0.0, 0.0, 0.0, 0.0]
    tz = [11.0, 11.0, 11.0, 5.0]

    pi = math.pi
    alpha = [pi / 2.0, pi / 2.0, pi / 2.0, 0.0]
    beta = [0.0, 0.0, 0.0, pi]
    gamma = [-pi / 2.0, 0.0, pi / 2.0, 0.0]

    intrinsics = create_intrinsics(fx, fy, cx, cy, device=device, dtype=dtype)
    extrinsics = create_extrinsics_with_rotation(alpha, beta, gamma, tx, ty, tz, device=device, dtype=dtype)

    cameras = PinholeCamera(intrinsics, extrinsics, height, width)
    return cameras


def create_random_images_for_cameras(cameras: PinholeCamera) -> list[Tensor]:
    """Create random images for a given set of cameras."""
    torch.manual_seed(112)
    imgs: list[Tensor] = []
    for height, width in zip(cameras.height.tolist(), cameras.width.tolist()):
        image_data = torch.randint(0, 255, (3, int(height), int(width)), dtype=torch.uint8)  # (C, H, W)
        imgs.append(image_data)  # (C, H, W)
    return imgs


def create_red_images_for_cameras(cameras: PinholeCamera, device: Device) -> list[Tensor]:
    """Create red images for a given set of cameras."""
    imgs: list[Tensor] = []
    for height, width in zip(cameras.height.tolist(), cameras.width.tolist()):
        image_data = torch.zeros(3, int(height), int(width), dtype=torch.uint8)  # (C, H, W)
        image_data[0, ...] = 255  # Red channel
        imgs.append(image_data.to(device=device))
    return imgs
