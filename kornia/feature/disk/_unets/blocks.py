# LICENSE HEADER MANAGED BY add-license-header
#
# Copyright 2018 Kornia Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class TrivialUpsample(nn.Module):
    """Bilinear upsampling layer used inside the thin DISK U-Net.

    The layer doubles spatial height and width while keeping batch size and channel count unchanged.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run this DISK component forward.

        Feature maps use `(B, C, H, W)`, where `B` is batch size, `C` channels, and `H`/`W` are spatial dimensions.

        Args:
            x: Input tensor processed by this module. For image-like features this usually follows the `(B, C, H, W)`
                layout, where `B` is batch size, `C` is channels, and `H`/`W` are height and width.

        Returns:
            Output tensor or dictionary produced by the module while preserving the shape contract documented by the
            surrounding class.
        """
        return F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)


class TrivialDownsample(nn.Module):
    """Average-pooling downsampling layer used inside the thin DISK U-Net.

    The layer halves spatial height and width with a `2x2` average-pooling window.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run this DISK component forward.

        Feature maps use `(B, C, H, W)`, where `B` is batch size, `C` channels, and `H`/`W` are spatial dimensions.

        Args:
            x: Input tensor processed by this module. For image-like features this usually follows the `(B, C, H, W)`
                layout, where `B` is batch size, `C` is channels, and `H`/`W` are height and width.

        Returns:
            Output tensor or dictionary produced by the module while preserving the shape contract documented by the
            surrounding class.
        """
        return F.avg_pool2d(x, 2)


class Conv(nn.Sequential):
    """Convolution block used by the thin DISK U-Net.

    The block optionally applies instance normalization and a PReLU gate before a same-padded convolution.
    """

    def __init__(self, in_: int, out_: int, size: int, skip_norm_and_gate: bool = False) -> None:
        norm: nn.Module
        nonl: nn.Module

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
    """Downsampling block for the thin DISK U-Net encoder path.

    The block optionally downsamples the feature map and then applies the configured convolution block.
    """

    def __init__(self, in_: int, out_: int, size: int = 5, is_first: bool = False) -> None:
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
    """Upsampling block for the thin DISK U-Net decoder path.

    The block upsamples the bottom feature map, concatenates it with the horizontal skip feature map, and applies a
    convolution block.
    """

    def __init__(self, bottom_: int, horizontal_: int, out_: int, size: int = 5) -> None:
        super().__init__()

        self.bottom_ = bottom_
        self.horizontal_ = horizontal_
        self.cat_ = bottom_ + horizontal_
        self.out_ = out_

        self.upsample = TrivialUpsample()
        self.conv = Conv(self.cat_, self.out_, size)

    def forward(self, bot: torch.Tensor, hor: torch.Tensor) -> torch.Tensor:
        """Run this DISK component forward.

        Feature maps use `(B, C, H, W)`, where `B` is batch size, `C` channels, and `H`/`W` are spatial dimensions.

        Args:
            bot: Input value used by this method.
            hor: Input value used by this method.

        Returns:
            Output tensor or dictionary produced by the module while preserving the shape contract documented by the
            surrounding class.
        """
        bot_big = self.upsample(bot)
        combined = torch.cat([bot_big, hor], dim=1)

        return self.conv(combined)
