import pytest
import torch

from kornia.feature.steerers import (
    DiscreteSteerer,
)

from testing.base import BaseTester

class TestDiscreteSteerer(BaseTester):
    @pytest.mark.parametrize("num_desc, desc_dim, steerer_power", [(1, 4, 1), (2, 128, 7), (32, 128, 11)])
    def test_shape(self, num_desc, desc_dim, steerer_power, device):
        desc = torch.rand(num_desc, desc_dim, device=device)
        generator = torch.rand(desc_dim, desc_dim, device=device)

        steerer = DiscreteSteerer(generator)

        desc = steerer.steer_descriptions(desc, steerer_power=steerer_power)
        assert desc.shape == (num_desc, desc_dim)

    def test_steering(self, device):
        generator = torch.tensor([[0., 1], [-1, 0]], device=device)
        desc = torch.rand(16, 2, device=device)
        steerer = DiscreteSteerer(generator)
        desc_out = steerer.steer_descriptions(desc, steerer_power=3)

        desc = desc[:, [1, 0]]
        desc[:, 0] = -desc[:, 0]
        assert torch.allclose(desc, desc_out)
