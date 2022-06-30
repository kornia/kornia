import torch

# from kornia.geometry.nerf.renderer import IrregularRenderer


class TestIrregularRenderer:
    def test_dimensions(self, device, dtype):
        # irreg_renderer = IrregularRenderer()
        height = 5
        width = 4
        num_ray_points = 7
        rgbs = torch.zeros((height, width, num_ray_points, 3), dtype=dtype, device=device)
        rgbs[..., 0] = 1
        # densities = torch.zeros((height, width, num_ray_points), dtype=dtype, device=device)
