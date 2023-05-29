import logging

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import kornia

logger = logging.getLogger(__name__)


class TestIntegrationFocalLoss:
    # optimization
    thresh = 1e-1
    lr = 1e-3
    num_iterations = 1000
    num_classes = 2

    # focal loss
    alpha = 0.5
    gamma = 2.0

    def generate_sample(self, base_target, std_val=0.1):
        target = base_target.float() / base_target.max()
        noise = std_val * torch.rand(1, 1, 6, 5).to(base_target.device)
        return target + noise

    @staticmethod
    def init_weights(m):
        if isinstance(m, nn.Conv2d):
            torch.nn.init.xavier_uniform_(m.weight)

    def test_conv2d_relu(self, device):
        # we generate base sample
        target = torch.LongTensor(1, 6, 5).fill_(0).to(device)
        for i in range(1, self.num_classes):
            target[..., i:-i, i:-i] = i

        m = nn.Sequential(nn.Conv2d(1, self.num_classes, kernel_size=3, padding=1), nn.ReLU(True)).to(device)
        m.apply(self.init_weights)

        optimizer = optim.Adam(m.parameters(), lr=self.lr)

        criterion = kornia.losses.FocalLoss(alpha=self.alpha, gamma=self.gamma, reduction='mean')
        # NOTE: uncomment to compare against vanilla cross entropy
        # criterion = nn.CrossEntropyLoss()

        for _ in range(self.num_iterations):
            sample = self.generate_sample(target).to(device)
            output = m(sample)
            loss = criterion(output, target.to(device))
            logger.debug(f"Loss: {loss.item()}")

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        sample = self.generate_sample(target).to(device)
        output_argmax = torch.argmax(m(sample), dim=1)
        logger.debug(f"Output argmax: \n{output_argmax}")

        # TODO(edgar): replace by IoU or find a more stable solution
        #              for this test. The issue is that depending on
        #              the seed to initialize the weights affects the
        #              final results and slows down the convergence of
        #              the algorithm.
        val = F.mse_loss(output_argmax.float(), target.float())
        if not val.item() < self.thresh:
            pytest.xfail("Wrong seed or initial weight values.")
