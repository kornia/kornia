import logging
import pytest

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import kornia as kornia

logger = logging.getLogger(__name__)


class TestIntegrationSoftArgmax2d:
    # optimization
    lr = 1e-3
    num_iterations = 500

    # data params
    height = 240
    width = 320

    def generate_sample(self, base_target, std_val=1.0):
        """Generates a random sample around the given point.
        The standard deviation is in pixel.
        """
        noise = std_val * torch.rand_like(base_target)
        return base_target + noise

    def test_regression_2d(self):
        # create the parameters to estimate: the heatmap
        params = nn.Parameter(torch.rand(1, 1, self.height, self.width))

        # generate base sample
        target = torch.zeros(1, 1, 2)
        target[..., 0] = self.width / 2
        target[..., 1] = self.height / 2

        # create the optimizer and pass the heatmap
        optimizer = optim.Adam([params], lr=self.lr)

        # loss criterion
        criterion = nn.MSELoss()

        # spatial soft-argmax2d module
        soft_argmax2d = kornia.geometry.SpatialSoftArgmax2d(
            normalized_coordinates=False)

        # NOTE: check where this comes from
        temperature = (self.height * self.width) ** (0.5)

        for iter_id in range(self.num_iterations):
            x = params
            sample = self.generate_sample(target)
            pred = soft_argmax2d(temperature * x)
            loss = criterion(pred, sample)
            logger.debug("Loss: {0:.3f} Pred: {1}".format(loss.item(), pred))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        assert pytest.approx(pred[..., 0].item(), target[..., 0].item())
        assert pytest.approx(pred[..., 1].item(), target[..., 1].item())
