from __future__ import annotations

import torch.nn.functional as F
from torch import nn

from kornia.core import Module, Tensor


class ConvNormAct(nn.Sequential):
    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, act: str = "relu", groups: int = 1
    ):
        super().__init__()
        if kernel_size % 2 == 0:
            # even kernel_size -> asymmetric padding
            # PPHGNetV2 (for RT-DETR) uses kernel 2
            # follow TensorFlow/PaddlePaddle: bottom/right side is padded 1 more than top/left
            # NOTE: this does not account for stride=2
            p1 = (kernel_size - 1) // 2
            p2 = kernel_size - 1 - p1
            self.pad = nn.ZeroPad2d((p1, p2, p1, p2))
            padding = 0
        else:
            padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, 1, groups, False)
        self.norm = nn.BatchNorm2d(out_channels)
        self.act = dict(relu=nn.ReLU, silu=nn.SiLU, none=nn.Identity)[act](inplace=True)


# Lightly adapted from
# https://github.com/facebookresearch/MaskFormer/blob/main/mask_former/modeling/transformer/transformer_predictor.py
class MLP(Module):
    def __init__(
        self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int, sigmoid_output: bool = False
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))
        self.sigmoid_output = sigmoid_output

    def forward(self, x: Tensor) -> Tensor:
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        if self.sigmoid_output:
            x = F.sigmoid(x)
        return x
