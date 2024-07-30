import pytest
import torch

from kornia.feature.steerers import DiscreteSteerer

from testing.base import BaseTester


class TestDiscreteSteerer(BaseTester):
    @pytest.mark.parametrize("num_desc, desc_dim, steerer_power", [(1, 4, 1), (2, 128, 7), (32, 128, 11)])
    def test_shape(self, num_desc, desc_dim, steerer_power, device):
        desc = torch.rand(num_desc, desc_dim, device=device)
        generator = torch.rand(desc_dim, desc_dim, device=device)

        steerer = DiscreteSteerer(generator)

        desc = steerer.steer_descriptions(desc, steerer_power=steerer_power)
        assert desc.shape == (num_desc, desc_dim)

    @pytest.mark.parametrize("normalize", [True, False])
    def test_steering(self, device, normalize):
        generator = torch.tensor([[0.0, 1], [-1, 0]], device=device)
        desc = torch.rand(16, 2, device=device)
        steerer = DiscreteSteerer(generator)
        desc_out = steerer.steer_descriptions(desc, steerer_power=3, normalize=normalize)

        if normalize:
            desc = torch.nn.functional.normalize(desc, dim=-1)
        desc = desc[:, [1, 0]]
        desc[:, 0] = -desc[:, 0]
        assert torch.allclose(desc, desc_out)

    @pytest.mark.parametrize("generator_type", ["C4", "SO2"])
    @pytest.mark.parametrize("steerer_order", [2, 14])
    def test_default(self, device, generator_type, steerer_order):
        steerer = DiscreteSteerer.create_dedode_default(
            generator_type=generator_type,
            steerer_order=steerer_order,
        ).to(device)
        assert isinstance(steerer, DiscreteSteerer)

        shape = (96, 256)
        desc = torch.randn(*shape, device=device)
        desc = steerer(desc)

        assert desc.shape == shape
