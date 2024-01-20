import logging

import pytest
import torch
from torch import nn, optim

import kornia
from kornia.testing import assert_close

logger = logging.getLogger(__name__)


class TestIntegrationSoftArgmax2d:
    # optimization
    lr = 1e-3
    num_iterations = 500

    # data params
    height = 240
    width = 320

    def generate_sample(self, base_target, std_val=1.0):
        """Generate a random sample around the given point.

        The standard deviation is in pixel.
        """
        noise = std_val * torch.rand_like(base_target)
        return base_target + noise

    @pytest.mark.slow
    def test_regression_2d(self, device):
        # create the parameters to estimate: the heatmap
        params = nn.Parameter(torch.rand(1, 1, self.height, self.width).to(device))

        # generate base sample
        target = torch.zeros(1, 1, 2).to(device)
        target[..., 0] = self.width / 2
        target[..., 1] = self.height / 2

        # create the optimizer and pass the heatmap
        optimizer = optim.Adam([params], lr=self.lr)

        # loss criterion
        criterion = nn.MSELoss()

        # spatial soft-argmax2d module
        soft_argmax2d = kornia.geometry.SpatialSoftArgmax2d(normalized_coordinates=False)

        # NOTE: check where this comes from
        temperature = (self.height * self.width) ** (0.5)

        for _ in range(self.num_iterations):
            x = params
            sample = self.generate_sample(target).to(device)
            pred = soft_argmax2d(temperature * x)
            loss = criterion(pred, sample)
            logger.debug(f"Loss: {loss.item():.3f} Pred: {pred}")

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        assert_close(pred[..., 0], target[..., 0], rtol=1e-2, atol=1e-2)
        assert_close(pred[..., 1], target[..., 1], rtol=1e-2, atol=1e-2)
