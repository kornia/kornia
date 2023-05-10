from __future__ import annotations

from torch import nn

from kornia.core import Module


def conv_norm_act(in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, act: str = "relu") -> Module:
    if act == "relu":
        act_module = nn.ReLU(inplace=True)
    elif act == "silu":
        act_module = nn.SiLU(inplace=True)
    elif act == "none":
        act_module = nn.Identity()

    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, (kernel_size - 1) // 2, bias=False),
        nn.BatchNorm2d(out_channels),
        act_module,
    )


class MLP(nn.Sequential):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int):
        layers = []
        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True)])
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, output_dim))
        super().__init__(*layers)
