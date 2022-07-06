import torch

from kornia.geometry.nerf.positional_encoder import PositionalEncoder


class TestPositionalEncoder:
    def test_dimensions(self, device, dtype):
        num_rays = 15
        num_ray_points = 11
        x = torch.rand(num_rays, num_ray_points, 3)
        num_freqs = 10
        pos_encoder = PositionalEncoder(num_freqs)
        x_encoded = pos_encoder(x)
        assert x_encoded.shape == (num_rays, num_ray_points, 3 * (2 * num_freqs + 1))
