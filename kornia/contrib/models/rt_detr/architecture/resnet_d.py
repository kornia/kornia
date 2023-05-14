"""ResNet-D as described in https://arxiv.org/abs/1812.01187.

Based on the code from https://github.com/pytorch/vision/blob/v0.15.1/torchvision/models/resnet.py
https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.6/ppdet/modeling/backbones/resnet.py
"""
from __future__ import annotations

from torch import nn

from kornia.contrib.models.common import ConvNormAct
from kornia.core import Module, Tensor
from kornia.core.check import KORNIA_CHECK


class BottleNeckD(Module):
    expansion = 4

    def __init__(self, in_channels: int, out_channels: int, stride: int, base_width: int):
        KORNIA_CHECK(stride in (1, 2))
        super().__init__()
        width = out_channels * base_width // 64
        expanded_out_channels = out_channels * self.expansion

        self.convs = nn.Sequential(
            ConvNormAct(in_channels, width, 1),
            ConvNormAct(width, width, 3, stride=stride),
            ConvNormAct(width, expanded_out_channels, 1, act="none"),
        )

        if stride == 2:
            self.shortcut = nn.Sequential(
                nn.AvgPool2d(2, 2), ConvNormAct(in_channels, expanded_out_channels, 1, act="none")
            )
        elif in_channels != out_channels * self.expansion:
            self.shortcut = ConvNormAct(in_channels, expanded_out_channels, 1, act="none")
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
        self.conv1 = nn.Sequential(
            ConvNormAct(3, in_channels // 2, 3, stride=2),
            ConvNormAct(in_channels // 2, in_channels // 2, 3),
            ConvNormAct(in_channels // 2, in_channels, 3),
            nn.MaxPool2d(3, stride=2, padding=1),
        )

        self.res2, in_channels = self.make_stage(in_channels, 64, 1, n_blocks[0])
        self.res3, in_channels = self.make_stage(in_channels, 128, 2, n_blocks[1])
        self.res4, in_channels = self.make_stage(in_channels, 256, 2, n_blocks[2])
        self.res5, in_channels = self.make_stage(in_channels, 512, 2, n_blocks[3])

        self.out_channels = [ch * BottleNeckD.expansion for ch in [128, 256, 512]]

    def make_stage(self, in_channels: int, out_channels: int, stride: int, n_blocks: int) -> tuple[Module, int]:
        layers = []
        layers.append(BottleNeckD(in_channels, out_channels, stride, self.base_width))
        for _ in range(n_blocks - 1):
            layers.append(BottleNeckD(out_channels * BottleNeckD.expansion, out_channels, 1, self.base_width))
        return nn.Sequential(*layers), out_channels * BottleNeckD.expansion

    def forward(self, x: Tensor) -> list[Tensor]:
        x = self.conv1(x)
        res2 = self.res2(x)
        res3 = self.res3(res2)
        res4 = self.res4(res3)
        res5 = self.res5(res4)
        return [res3, res4, res5]

    @staticmethod
    def from_config(variant: str | int):
        arch_configs = {18: [2, 2, 2, 2], 34: [3, 4, 6, 3], 50: [3, 4, 6, 3], 101: [3, 4, 23, 3], 152: [3, 8, 36, 3]}
        variant = int(variant)
        KORNIA_CHECK(variant in arch_configs, "Only variant 18, 34, 50, 101, and 152 are supported")
        return ResNetD(arch_configs[variant])
