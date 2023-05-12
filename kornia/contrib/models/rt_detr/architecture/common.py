from __future__ import annotations

from torch import nn


class ConvNormAct(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, act: str = "relu"):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, (kernel_size - 1) // 2, bias=False)
        self.norm = nn.BatchNorm2d(out_channels)
        self.act = dict(relu=nn.ReLU, silu=nn.SiLU, none=nn.Identity)[act](inplace=True)


class MLP(nn.Sequential):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int):
        layers = []
        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True)])
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, output_dim))
        super().__init__(*layers)
