import pytest
from common import TEST_DEVICES

import torch
import torchgeometry as tgm
import torch.nn as nn
import torchvision.models as models


class WarpPerspective(nn.Module):

    def __init__(self, output_size):
        super(WarpPerspective, self).__init__()

        self.output_height, self.output_width = output_size

    def forward(self, x, M):

        x = tgm.warp_perspective(x, M,
                                 dsize=(self.output_height,
                                        self.output_width))

        return x


class WarpAffine(nn.Module):

    def __init__(self, output_size):
        super(WarpAffine, self).__init__()

        self.output_height, self.output_width = output_size

    def forward(self, x, M):

        x = tgm.warp_affine(x, M,
                            dsize=(self.output_height,
                                   self.output_width))

        return x


class LocalizationNetwork(nn.Module):

    def __init__(self, output_size, pretrained=True, affine=True):
        super(LocalizationNetwork, self).__init__()

        self.cnn = models.__dict__['alexnet'](pretrained=pretrained)
        self.cnn = nn.Sequential(*list(self.cnn.children())[0])
        self.cnn = nn.Sequential(*list(self.cnn.children())[:-1])

        self.affine = affine

        if self.affine:
            self.linear = nn.Linear(256 * 3 * 3, 6)
            self.warper = WarpAffine(output_size)
        else:
            self.linear = nn.Linear(256 * 3 * 3, 9)
            self.warper = WarpPerspective(output_size)

    def forward(self, x):
        bs = x.size(0)

        inp_height = x.size(2)
        inp_width = x.size(3)

        x_down = nn.functional.interpolate(x, size=(64, 64), mode='bilinear')

        M = self.linear(self.cnn(x_down).view(bs, -1))
        if self.affine:
            M = M.view(-1, 2, 3)
        else:
            M = M.view(-1, 3, 3)

        out = self.warper(x, M)

        return (out, M)


@pytest.mark.parametrize("device_type", TEST_DEVICES)
@pytest.mark.parametrize("affine", [True, False])
def test_jit_tracing(device_type, affine):

    net = LocalizationNetwork((128, 128), pretrained=False, affine=affine)
    net.eval()

    net = net.to(torch.device(device_type))

    dummy_input = torch.randn(1, 3, 128, 128)
    dummy_input = dummy_input.to(torch.device(device_type))

    output = net(dummy_input)

    traced_net = torch.jit.trace(net, dummy_input)
    traced_output = traced_net(dummy_input)

    out_comp1 = float(torch.mean((output[0] == traced_output[0]).float()))
    out_comp2 = float(torch.mean((output[1] == traced_output[1]).float()))

    assert(out_comp1 == 1.0)
    assert(out_comp2 == 1.0)
