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

import torch

from kornia.core import Device, Tensor
from kornia.geometry.camera import PinholeCamera
from kornia.nerf.data_utils import RayDataset, instantiate_ray_dataloader

from testing.base import assert_close

from tests.nerf.test_rays import create_four_cameras


def create_random_images_for_cameras(cameras: PinholeCamera) -> list[Tensor]:
    """Creates random images for a given set of cameras."""
    torch.manual_seed(112)
    imgs: list[Tensor] = []
    for height, width in zip(cameras.height.tolist(), cameras.width.tolist()):
        image_data = torch.randint(0, 255, (3, int(height), int(width)), dtype=torch.uint8)  # (C, H, W)
        imgs.append(image_data)  # (C, H, W)
    return imgs


def create_red_images_for_cameras(cameras: PinholeCamera, device: Device) -> list[Tensor]:
    """Creates red images for a given set of cameras."""
    imgs: list[Tensor] = []
    for height, width in zip(cameras.height.tolist(), cameras.width.tolist()):
        image_data = torch.zeros(3, int(height), int(width), dtype=torch.uint8)  # (C, H, W)
        image_data[0, ...] = 255  # Red channel
        imgs.append(image_data.to(device=device))
    return imgs


class TestDataset:
    def test_uniform_ray_dataset(self, device, dtype):
        cameras = create_four_cameras(device, dtype)
        imgs = create_random_images_for_cameras(cameras)
        dataset = RayDataset(cameras, 1, 2, False, device=device, dtype=dtype)
        dataset.init_ray_dataset()
        dataset.init_images_for_training(imgs)

        batch_size = 32
        data_loader = instantiate_ray_dataloader(dataset, batch_size=batch_size, shuffle=False)
        d = next(iter(data_loader))  # First batch of 32 labeled rays

        # Check dimensions
        assert d[0].shape == (batch_size, 3)  # Ray origins
        assert d[1].shape == (batch_size, 3)  # Ray directions
        assert d[2].shape == (batch_size, 3)  # Ray rgbs

        # Comparing RGB values between sampled rays and original images
        assert_close(d[2][0].cpu().to(dtype), (imgs[0][:, 0, 0] / 255.0).to(dtype))
        assert_close(
            d[2][1].cpu().to(dtype), (imgs[0][:, 0, 1] / 255.0).to(dtype)
        )  # First row, second column in the image (1 sample point index)
        assert_close(
            d[2][9].cpu().to(dtype), (imgs[0][:, 1, 0] / 255.0).to(dtype)
        )  # Second row, first column in the image (9 sample point index)
