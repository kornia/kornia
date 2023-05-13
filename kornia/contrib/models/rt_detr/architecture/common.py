from __future__ import annotations

import torch.nn.functional as F
from torch import nn

from kornia.core import Module, Tensor


class ConvNormAct(nn.Sequential):
    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, act: str = "relu", groups: int = 1
    ):
        super().__init__()
        if kernel_size % 2 == 0:  # even kernel_size -> asymmetric padding
            # NOTE: check paddlepaddle's SAME padding
            # NOTE: this does not account for stride=2
            p2 = (kernel_size - 1) // 2
            p1 = kernel_size - 1 - p2
            self.pad = nn.ZeroPad2d((p1, p2, p1, p2))
            padding = 0
        else:
            padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, 1, groups, False)
        self.norm = nn.BatchNorm2d(out_channels)
        self.act = dict(relu=nn.ReLU, silu=nn.SiLU, none=nn.Identity)[act](inplace=True)


# NOTE: can be merged with sam.architecture.mask_decoder.MLP
class MLP(Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(num_layers - 1):
            self.layers.append(nn.Linear(input_dim, hidden_dim))
            input_dim = hidden_dim
        self.layers.append(nn.Linear(input_dim, output_dim))

    def forward(self, x: Tensor) -> Tensor:
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x), True) if i < len(self.layers) - 1 else layer(x)
        return x
