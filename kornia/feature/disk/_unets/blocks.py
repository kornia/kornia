from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from kornia.core import Module, Tensor


class TrivialUpsample(Module):
    def forward(self, x: Tensor) -> Tensor:
        return F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)


class TrivialDownsample(Module):
    def forward(self, x: Tensor) -> Tensor:
        return F.avg_pool2d(x, 2)


class Conv(nn.Sequential):
    def __init__(self, in_: int, out_: int, size: int, skip_norm_and_gate: bool = False) -> None:
        norm: Module
        nonl: Module

        if skip_norm_and_gate:
            norm = nn.Sequential()
            nonl = nn.Sequential()
        else:
            norm = nn.InstanceNorm2d(in_)
            nonl = nn.PReLU(in_)

        dropout = nn.Sequential()
        conv = nn.Conv2d(in_, out_, size, padding="same", bias=True)

        super().__init__(norm, nonl, dropout, conv)


class ThinUnetDownBlock(nn.Sequential):
    def __init__(self, in_: int, out_: int, size: int = 5, is_first: bool = False, setup: Any = None) -> None:
        self.in_ = in_
        self.out_ = out_

        downsample: Module
        if is_first:
            downsample = nn.Sequential()
            conv = Conv(in_, out_, size, skip_norm_and_gate=True)
        else:
            downsample = TrivialDownsample()
            conv = Conv(in_, out_, size)

        super().__init__(downsample, conv)


class ThinUnetUpBlock(Module):
    def __init__(self, bottom_: int, horizontal_: int, out_: int, size: int = 5, setup: Any = None) -> None:
        super().__init__()

        self.bottom_ = bottom_
        self.horizontal_ = horizontal_
        self.cat_ = bottom_ + horizontal_
        self.out_ = out_

        self.upsample = TrivialUpsample()
        self.conv = Conv(self.cat_, self.out_, size)

    def forward(self, bot: Tensor, hor: Tensor) -> Tensor:
        bot_big = self.upsample(bot)
        combined = torch.cat([bot_big, hor], dim=1)

        return self.conv(combined)
