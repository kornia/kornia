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

import torch

from kornia.core import Device, Tensor
from kornia.nerf.volume_renderer import IrregularRenderer, RegularRenderer
from kornia.utils._compat import torch_meshgrid

from testing.base import assert_close


def _create_regular_point_cloud(
    height: int, width: int, num_ray_points: int, device: Device, dtype: torch.dtype
) -> Tensor:
    x = torch.linspace(0, width, steps=width, device=device, dtype=dtype)
    y = torch.linspace(0, height, steps=height, device=device, dtype=dtype)
    xy = torch_meshgrid([y, x], indexing="ij")
    z = torch.linspace(1, 11, steps=num_ray_points, device=device, dtype=dtype)
    points3d = torch.zeros(height, width, num_ray_points, 3, device=device, dtype=dtype)
    points3d[..., 0] = xy[0].unsqueeze(-1)
    points3d[..., 1] = xy[1].unsqueeze(-1)
    points3d[..., 2] = z
    return points3d


class TestRenderer:
    def test_dimensions(self, device, dtype):
        height = 5
        width = 4
        num_ray_points = 7
        rgbs = torch.rand((height, width, num_ray_points, 3), dtype=dtype, device=device)
        densities = torch.rand((height, width, num_ray_points, 1), dtype=dtype, device=device)

        points3d = _create_regular_point_cloud(height, width, num_ray_points, device=device, dtype=dtype)

        irreg_renderer = IrregularRenderer()
        rgb_rendered = irreg_renderer(rgbs, densities, points3d)
        assert rgb_rendered.shape == (height, width, 3)

    def test_only_red(self, device, dtype):
        height = 5
        width = 4
        num_ray_points = 7
        rgbs = torch.zeros((height, width, num_ray_points, 3), dtype=dtype, device=device)
        rgbs[..., 0] = 1
        densities = torch.rand((height, width, num_ray_points, 1), dtype=dtype, device=device)

        points3d = _create_regular_point_cloud(height, width, num_ray_points, device=device, dtype=dtype)

        irreg_renderer = IrregularRenderer()
        rgbs_rendered = irreg_renderer(rgbs, densities, points3d)
        only_reds = torch.zeros_like(rgbs_rendered, dtype=dtype, device=device)
        only_reds[..., 0] = 1.0
        assert_close(rgbs_rendered, only_reds)

    def test_shell(self, device, dtype):
        height = 5
        width = 4
        num_ray_points = 11
        rgbs = torch.rand((height, width, num_ray_points, 3), dtype=dtype, device=device)
        densities = torch.zeros((height, width, num_ray_points, 1), dtype=dtype, device=device)
        densities[..., 0, :] = 1

        points3d = _create_regular_point_cloud(height, width, num_ray_points, device=device, dtype=dtype)

        regular_renderer = RegularRenderer()
        rgbs_rendered = regular_renderer(rgbs, densities, points3d)

        weight = 1 - math.exp(-1.0)
        assert_close(rgbs_rendered, weight * rgbs[..., 0, :])

    def test_regular_vs_irregular(self, device, dtype):
        height = 5
        width = 4
        num_ray_points = 11
        rgbs = torch.rand((height, width, num_ray_points, 3), dtype=dtype, device=device)
        densities = torch.zeros((height, width, num_ray_points, 1), dtype=dtype, device=device)
        densities[..., 0, :] = 10

        points3d = _create_regular_point_cloud(height, width, num_ray_points, device=device, dtype=dtype)

        irreg_renderer = IrregularRenderer()
        irreg_rgb_rendered = irreg_renderer(rgbs, densities, points3d)

        regular_renderer = RegularRenderer()
        regular_rgbs_rendered = regular_renderer(rgbs, densities, points3d)

        assert irreg_rgb_rendered.shape == regular_rgbs_rendered.shape
        assert_close(irreg_rgb_rendered, regular_rgbs_rendered)
