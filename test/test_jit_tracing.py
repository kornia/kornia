import pytest
from common import device_type

import torch
import kornia as kornia
import torch.nn as nn


class WarpPerspective(nn.Module):

    def __init__(self, output_size):
        super(WarpPerspective, self).__init__()

        self.output_height, self.output_width = output_size

    def forward(self, x, M):
        x = kornia.warp_perspective(x, M,
                                    dsize=(self.output_height,
                                           self.output_width))
        return x


class WarpAffine(nn.Module):

    def __init__(self, output_size):
        super(WarpAffine, self).__init__()

        self.output_height, self.output_width = output_size

    def forward(self, x, M):
        x = kornia.warp_affine(x, M,
                               dsize=(self.output_height,
                                      self.output_width))
        return x


class LocalizationNetwork(nn.Module):

    def __init__(self, output_size, pretrained=True, affine=True):
        super(LocalizationNetwork, self).__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(2, 3, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.affine = affine

        if self.affine:
            self.linear = nn.Linear(3 * 2 * 2, 6)
            self.warper = WarpAffine(output_size)
        else:
            self.linear = nn.Linear(3 * 2 * 2, 9)
            self.warper = WarpPerspective(output_size)

    def forward(self, x):
        batch_size = x.size(0)

        M = self.linear(self.cnn(x).view(batch_size, -1))
        if self.affine:
            M = M.view(-1, 2, 3)
        else:
            M = M.view(-1, 3, 3)

        out = self.warper(x, M)
        return (out, M)


# TODO(wizaron): need the double check what's going on here
@pytest.mark.skip(reason="in PyTorch >= v1.0.0 it crashes.")
@pytest.mark.parametrize("affine", [True, False])
def test_jit_tracing(device_type, affine):
    net = LocalizationNetwork((4, 4), pretrained=False, affine=affine)
    net.eval()

    net = net.to(torch.device(device_type))

    dummy_input = torch.randn(1, 2, 4, 4)
    dummy_input = dummy_input.to(torch.device(device_type))

    output = net(dummy_input)

    traced_net = torch.jit.trace(net, dummy_input)
    traced_output = traced_net(dummy_input)

    out_comp1 = float(torch.mean((output[0] == traced_output[0]).float()))
    out_comp2 = float(torch.mean((output[1] == traced_output[1]).float()))

    assert(out_comp1 == 1.0)
    assert(out_comp2 == 1.0)
