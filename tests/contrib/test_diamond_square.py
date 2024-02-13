import torch

import kornia

from testing.base import BaseTester


class TestDiamondSquare(BaseTester):
    def test_smoke(self, device, dtype):
        torch.manual_seed(0)
        output_size = (1, 1, 3, 4)
        roughness = 0.5
        random_scale = 1.0
        out = kornia.contrib.diamond_square(output_size, roughness, random_scale, device=device, dtype=dtype)
        assert out.shape == output_size
        assert out.device == device
        assert out.dtype == dtype

    def test_normalize(self, device, dtype):
        torch.manual_seed(0)
        output_size = (1, 1, 3, 4)
        roughness = 0.5
        random_scale = 1.0
        normalize_range = (0.0, 1.0)
        expected_min = torch.tensor(normalize_range[0], device=device, dtype=dtype)
        expected_max = torch.tensor(normalize_range[1], device=device, dtype=dtype)
        out = kornia.contrib.diamond_square(
            output_size, roughness, random_scale, normalize_range=normalize_range, device=device, dtype=dtype
        )
        self.assert_close(out.min(), expected_min)
        self.assert_close(out.max(), expected_max)
