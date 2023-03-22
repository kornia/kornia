import torch
import torch.nn as nn

from kornia.core import Tensor

from .ops import NoOp
from .utils import cut_to_match


class Conv(nn.Sequential):
    def __init__(self, in_, out_, size, setup=None):
        norm = setup['norm'](in_)
        nonl = setup['gate'](in_)
        dropout = setup['dropout']()

        if setup['padding']:
            padding = size // 2
        else:
            padding = 0

        if 'bias' in setup:
            bias = setup['bias']
        else:
            bias = True

        conv = nn.Conv2d(in_, out_, size, padding=padding, bias=bias)

        super().__init__(norm, nonl, dropout, conv)


class ThinUnetDownBlock(nn.Sequential):
    def __init__(self, in_, out_, size=5, name=None, is_first=False, setup=None):
        self.name = name
        self.in_ = in_
        self.out_ = out_

        if is_first:
            downsample = NoOp()
            conv = Conv(in_, out_, size, setup={**setup, 'gate': NoOp, 'norm': NoOp})
        else:
            downsample = setup['downsample'](in_, size, setup=setup)
            conv = Conv(in_, out_, size, setup=setup)

        super().__init__(downsample, conv)


class ThinUnetUpBlock(nn.Module):
    def __init__(self, bottom_, horizontal_, out_, size=5, name=None, setup=None):
        super().__init__()

        self.name = name
        self.bottom_ = bottom_
        self.horizontal_ = horizontal_
        self.cat_ = bottom_ + horizontal_
        self.out_ = out_

        self.upsample = setup['upsample'](bottom_, size, setup=setup)
        self.conv = Conv(self.cat_, self.out_, size, setup=setup)

    def forward(self, bot: Tensor, hor: Tensor) -> Tensor:
        bot_big = self.upsample(bot)
        hor = cut_to_match(bot_big, hor, n_pref=2)
        combined = torch.cat([bot_big, hor], dim=1)

        return self.conv(combined)
