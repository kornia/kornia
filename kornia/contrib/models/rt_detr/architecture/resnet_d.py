"""ResNet-D as described in https://arxiv.org/abs/1812.01187.

Based on the code from https://github.com/pytorch/vision/blob/v0.15.1/torchvision/models/resnet.py
https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.6/ppdet/modeling/backbones/resnet.py
"""
from __future__ import annotations

from torch import nn

from kornia.contrib.models.common import ConvNormAct
from kornia.core import Module, Tensor
from kornia.core.check import KORNIA_CHECK


def _make_shortcut(in_channels: int, out_channels: int, stride: int) -> Module:
    return (
        nn.Sequential(nn.AvgPool2d(2, 2), ConvNormAct(in_channels, out_channels, 1, act="none"))
        if stride == 2
        else ConvNormAct(in_channels, out_channels, 1, act="none")
    )


class BasicBlockD(Module):
    expansion = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int, shortcut: bool) -> None:
        KORNIA_CHECK(stride in {1, 2})
        super().__init__()
        self.convs = nn.Sequential(
            ConvNormAct(in_channels, out_channels, 3, stride=stride),
            ConvNormAct(out_channels, out_channels, 3, act="none"),
        )
        self.shortcut = nn.Identity() if shortcut else _make_shortcut(in_channels, out_channels, stride)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        return self.relu(self.convs(x) + self.shortcut(x))


class BottleneckD(Module):
    expansion = 4

    def __init__(self, in_channels: int, out_channels: int, stride: int, shortcut: bool) -> None:
        KORNIA_CHECK(stride in {1, 2})
        super().__init__()
        expanded_out_channels = out_channels * self.expansion
        self.convs = nn.Sequential(
            ConvNormAct(in_channels, out_channels, 1),
            ConvNormAct(out_channels, out_channels, 3, stride=stride),
            ConvNormAct(out_channels, expanded_out_channels, 1, act="none"),
        )
        self.shortcut = nn.Identity() if shortcut else _make_shortcut(in_channels, out_channels, stride)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        return self.relu(self.convs(x) + self.shortcut(x))


class ResNetD(Module):
    def __init__(self, n_blocks: list[int], block: type[BasicBlockD | BottleneckD]) -> None:
        KORNIA_CHECK(len(n_blocks) == 4)
        super().__init__()
        in_channels = 64
        self.conv1 = nn.Sequential(
            ConvNormAct(3, in_channels // 2, 3, stride=2),
            ConvNormAct(in_channels // 2, in_channels // 2, 3),
            ConvNormAct(in_channels // 2, in_channels, 3),
            nn.MaxPool2d(3, stride=2, padding=1),
        )

        self.res2, in_channels = self.make_stage(in_channels, 64, 1, n_blocks[0], block)
        self.res3, in_channels = self.make_stage(in_channels, 128, 2, n_blocks[1], block)
        self.res4, in_channels = self.make_stage(in_channels, 256, 2, n_blocks[2], block)
        self.res5, in_channels = self.make_stage(in_channels, 512, 2, n_blocks[3], block)

        self.out_channels = [ch * block.expansion for ch in [128, 256, 512]]

    @staticmethod
    def make_stage(
        in_channels: int, out_channels: int, stride: int, n_blocks: int, block: type[BasicBlockD | BottleneckD]
    ) -> tuple[Module, int]:
        stage = nn.Sequential(
            block(in_channels, out_channels, stride, False),
            *[block(out_channels * block.expansion, out_channels, 1, True) for _ in range(n_blocks - 1)],
        )
        return stage, out_channels * block.expansion

    def forward(self, x: Tensor) -> list[Tensor]:
        x = self.conv1(x)
        res2 = self.res2(x)
        res3 = self.res3(res2)
        res4 = self.res4(res3)
        res5 = self.res5(res4)
        return [res3, res4, res5]

    @staticmethod
    def from_config(variant: str | int) -> ResNetD:
        variant = str(variant)
        if variant == "18":
            return ResNetD([2, 2, 2, 2], BasicBlockD)
        elif variant == "34":
            return ResNetD([3, 4, 6, 3], BasicBlockD)
        elif variant == "50":
            return ResNetD([3, 4, 6, 3], BottleneckD)
        elif variant == "101":
            return ResNetD([3, 4, 23, 3], BottleneckD)
        elif variant == "152":
            return ResNetD([3, 8, 36, 3], BottleneckD)
        else:
            raise ValueError("Only variant 18, 34, 50, 101, and 152 are supported")
