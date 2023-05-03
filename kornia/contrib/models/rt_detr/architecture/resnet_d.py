"""ResNet-D as described in https://arxiv.org/abs/1812.01187.

Based on the code from https://github.com/pytorch/vision/blob/v0.15.1/torchvision/models/resnet.py
https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.6/ppdet/modeling/backbones/resnet.py
"""
from __future__ import annotations

from torch import nn

from kornia.core import Module, Tensor
from kornia.core.check import KORNIA_CHECK


def conv_norm_act(in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, act: bool = True) -> Module:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, (kernel_size - 1) // 2, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True) if act else nn.Identity(),
    )


class BottleNeckD(Module):
    expansion = 4

    def __init__(self, in_channels: int, out_channels: int, stride: int, base_width: int):
        KORNIA_CHECK(stride in (1, 2))
        super().__init__()
        width = out_channels * base_width // 64
        expanded_out_channels = out_channels * self.expansion

        self.convs = nn.Sequential(
            conv_norm_act(in_channels, width, 1),
            conv_norm_act(width, width, 3, stride=stride),
            conv_norm_act(width, expanded_out_channels, 1, act=False),
        )

        if stride == 2:
            self.shortcut = nn.Sequential(
                nn.AvgPool2d(2, 2), conv_norm_act(in_channels, expanded_out_channels, 1, act=False)
            )
        elif in_channels != out_channels * self.expansion:
            self.shortcut = conv_norm_act(in_channels, expanded_out_channels, 1, act=False)
        else:
            self.shortcut = nn.Identity()

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        return self.relu(self.convs(x) + self.shortcut(x))


class ResNetD(Module):
    base_width = 64

    def __init__(self, n_blocks: list[int]):
        KORNIA_CHECK(len(n_blocks) == 4)
        super().__init__()
        in_channels = 64
        self.stem = nn.Sequential(
            conv_norm_act(3, in_channels // 2, 3, stride=2),
            conv_norm_act(in_channels // 2, in_channels // 2, 3),
            conv_norm_act(in_channels // 2, in_channels, 3),
            nn.MaxPool2d(3, stride=2, padding=1),
        )

        self.res2, in_channels = self.make_stage(in_channels, 64, 1, n_blocks[0])
        self.res3, in_channels = self.make_stage(in_channels, 128, 2, n_blocks[1])
        self.res4, in_channels = self.make_stage(in_channels, 256, 2, n_blocks[2])
        self.res5, in_channels = self.make_stage(in_channels, 512, 2, n_blocks[3])

    def make_stage(self, in_channels: int, out_channels: int, stride: int, n_blocks: int) -> tuple[Module, int]:
        stage = nn.Sequential()
        stage.append(BottleNeckD(in_channels, out_channels, stride, self.base_width))
        for _ in range(n_blocks - 1):
            stage.append(BottleNeckD(out_channels * BottleNeckD.expansion, out_channels, 1, self.base_width))
        return stage, out_channels * BottleNeckD.expansion

    def forward(self, x: Tensor) -> list[Tensor]:
        out = self.stem(x)
        fmaps = [self.res2(out)]
        fmaps.append(self.res3(fmaps[-1]))
        fmaps.append(self.res4(fmaps[-1]))
        fmaps.append(self.res5(fmaps[-1]))
        return fmaps
