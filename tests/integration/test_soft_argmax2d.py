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

import logging

import pytest
import torch
from torch import nn, optim

import kornia

from testing.base import BaseTester

logger = logging.getLogger(__name__)


class TestIntegrationSoftArgmax2d(BaseTester):
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

        self.assert_close(pred[..., 0], target[..., 0], rtol=1e-2, atol=1e-2)
        self.assert_close(pred[..., 1], target[..., 1], rtol=1e-2, atol=1e-2)
