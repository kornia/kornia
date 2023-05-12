from __future__ import annotations

import torch.nn.functional as F
from torch import nn

from kornia.core import Module, Tensor


class ConvNormAct(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, act: str = "relu"):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, (kernel_size - 1) // 2, bias=False)
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
