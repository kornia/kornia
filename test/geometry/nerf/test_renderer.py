import math

import torch

from kornia.geometry.nerf.renderer import IrregularRenderer, RegularRenderer
from kornia.testing import assert_close


def _create_regular_point_cloud(height: int, width: int, num_ray_points: int) -> torch.tensor:
    x = torch.linspace(0, width, steps=width)
    y = torch.linspace(0, height, steps=height)
    xy = torch.meshgrid(y, x)
    z = torch.linspace(1, 11, steps=num_ray_points)
    points3d = torch.zeros(height, width, num_ray_points, 3)
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
        densities = torch.rand((height, width, num_ray_points), dtype=dtype, device=device)

        points3d = _create_regular_point_cloud(height, width, num_ray_points)

        irreg_renderer = IrregularRenderer()
        rgb_rendered = irreg_renderer(rgbs, densities, points3d)
        assert rgb_rendered.shape == (height, width, 3)

    def test_only_red(self, device, dtype):
        height = 5
        width = 4
        num_ray_points = 7
        rgbs = torch.zeros((height, width, num_ray_points, 3), dtype=dtype, device=device)
        rgbs[..., 0] = 1
        densities = torch.rand((height, width, num_ray_points), dtype=dtype, device=device)

        points3d = _create_regular_point_cloud(height, width, num_ray_points)

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
        densities = torch.zeros((height, width, num_ray_points), dtype=dtype, device=device)
        densities[..., 0] = 1

        points3d = _create_regular_point_cloud(height, width, num_ray_points)

        regular_renderer = RegularRenderer()
        rgbs_rendered = regular_renderer(rgbs, densities, points3d)

        weight = 1 - math.exp(-1.0)
        assert_close(rgbs_rendered, weight * rgbs[..., 0, :])

    def test_regular_vs_irregular(self, device, dtype):
        height = 5
        width = 4
        num_ray_points = 11
        rgbs = torch.rand((height, width, num_ray_points, 3), dtype=dtype, device=device)
        densities = torch.zeros((height, width, num_ray_points), dtype=dtype, device=device)
        densities[..., 0] = 10

        points3d = _create_regular_point_cloud(height, width, num_ray_points)

        irreg_renderer = IrregularRenderer()
        irreg_rgb_rendered = irreg_renderer(rgbs, densities, points3d)

        regular_renderer = RegularRenderer()
        regular_rgbs_rendered = regular_renderer(rgbs, densities, points3d)

        assert irreg_rgb_rendered.shape == regular_rgbs_rendered.shape
        assert_close(irreg_rgb_rendered, regular_rgbs_rendered)
