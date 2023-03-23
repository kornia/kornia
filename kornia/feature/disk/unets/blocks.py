import torch
import torch.nn as nn
import torch.nn.functional as F

from kornia.core import Tensor

from .utils import cut_to_match, size_is_pow2


class TrivialUpsample(nn.Module):
    def forward(self, x):
        r = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        return r


class TrivialDownsample(nn.Module):
    def forward(self, x):
        if not size_is_pow2(x):
            msg = f"Trying to downsample feature map of size {x.size()}"
            raise RuntimeError(msg)

        return F.avg_pool2d(x, 2)


class Conv(nn.Sequential):
    def __init__(self, in_, out_, size, skip_norm_and_gate=False):
        norm: nn.Module
        nonl: nn.Module

        if skip_norm_and_gate:
            norm = nn.Sequential()
            nonl = nn.Sequential()
        else:
            norm = nn.InstanceNorm2d(in_)
            nonl = nn.PReLU(in_)

        dropout = nn.Sequential()
        conv = nn.Conv2d(in_, out_, size, padding='same', bias=True)

        super().__init__(norm, nonl, dropout, conv)


class ThinUnetDownBlock(nn.Sequential):
    def __init__(self, in_, out_, size=5, is_first=False, setup=None):
        self.in_ = in_
        self.out_ = out_

        downsample: nn.Module
        if is_first:
            downsample = nn.Sequential()
            conv = Conv(in_, out_, size, skip_norm_and_gate=True)
        else:
            downsample = TrivialDownsample()
            conv = Conv(in_, out_, size)

        super().__init__(downsample, conv)


class ThinUnetUpBlock(nn.Module):
    def __init__(self, bottom_, horizontal_, out_, size=5, setup=None):
        super().__init__()

        self.bottom_ = bottom_
        self.horizontal_ = horizontal_
        self.cat_ = bottom_ + horizontal_
        self.out_ = out_

        self.upsample = TrivialUpsample()
        self.conv = Conv(self.cat_, self.out_, size)

    def forward(self, bot: Tensor, hor: Tensor) -> Tensor:
        bot_big = self.upsample(bot)
        hor = cut_to_match(bot_big, hor, n_pref=2)
        combined = torch.cat([bot_big, hor], dim=1)

        return self.conv(combined)
