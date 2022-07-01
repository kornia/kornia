import math

import torch

from kornia.geometry.nerf.renderer import IrregularRenderer, RegularRenderer
from kornia.testing import assert_close


class TestIrregularRenderer:
    def test_dimensions(self, device, dtype):
        height = 5
        width = 4
        num_ray_points = 7
        rgbs = torch.rand((height, width, num_ray_points, 3), dtype=dtype, device=device)
        densities = torch.rand((height, width, num_ray_points), dtype=dtype, device=device)
        t_vals = torch.linspace(3, 10, steps=num_ray_points).repeat(height, width, 1)
        irreg_renderer = IrregularRenderer()
        rgb_rendered = irreg_renderer(rgbs, densities, t_vals)
        assert rgb_rendered.shape == (height, width, 3)

    def test_only_red(self, device, dtype):
        height = 5
        width = 4
        num_ray_points = 7
        rgbs = torch.zeros((height, width, num_ray_points, 3), dtype=dtype, device=device)
        rgbs[..., 0] = 1
        densities = torch.rand((height, width, num_ray_points), dtype=dtype, device=device)
        t_vals = torch.linspace(3, 10, steps=num_ray_points).repeat(height, width, 1)
        irreg_renderer = IrregularRenderer()
        rgbs_rendered = irreg_renderer(rgbs, densities, t_vals)
        only_reds = torch.zeros_like(rgbs_rendered, dtype=dtype, device=device)
        only_reds[..., 0] = 1.0
        assert_close(rgbs_rendered, only_reds)

    def test_shell(self, device, dtype):
        height = 5
        width = 4
        num_ray_points = 7
        rgbs = torch.rand((height, width, num_ray_points, 3), dtype=dtype, device=device)
        densities = torch.zeros((height, width, num_ray_points), dtype=dtype, device=device)
        densities[..., 0] = 10
        deltas = torch.ones((height, width), dtype=dtype, device=device) * 0.1
        regular_renderer = RegularRenderer()
        rgbs_rendered = regular_renderer(rgbs, densities, deltas)
        weight = 1 - math.exp(-1.0)
        assert_close(rgbs_rendered, weight * rgbs[..., 0, :])
