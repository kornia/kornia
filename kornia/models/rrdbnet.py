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

"""Residual-in-Residual Dense Block (RRDB) generator used by ESRGAN and Real-ESRGAN.

Adapted from BasicSR (https://github.com/XPixelGroup/BasicSR), Copyright 2018-2022 BasicSR Authors,
Apache-2.0. Modified: type hints and docstrings, and the vendored ``pixel_unshuffle`` helper is
expressed as :func:`torch.nn.functional.pixel_unshuffle`, which returns a byte-identical result.

The module and parameter names are kept identical to upstream so that the published Real-ESRGAN
checkpoints load with ``strict=True``.

References:
    Wang et al., "ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks", ECCVW 2018.
    https://arxiv.org/abs/1809.00219
"""

from __future__ import annotations

from typing import Iterable, Union

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import init
from torch.nn.modules.batchnorm import _BatchNorm

__all__ = ["RRDBNet"]


@torch.no_grad()
def _default_init_weights(
    module_list: Union[Iterable[nn.Module], nn.Module], scale: float = 1.0, bias_fill: float = 0.0
) -> None:
    """Initialize network weights.

    Args:
        module_list: Modules to be initialized.
        scale: Scale initialized weights, especially for residual blocks.
        bias_fill: The value to fill the bias with.

    """
    modules = [module_list] if isinstance(module_list, nn.Module) else list(module_list)
    for module in modules:
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)
            elif isinstance(m, _BatchNorm):
                init.constant_(m.weight, 1)
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)


class ResidualDenseBlock(nn.Module):
    """Residual Dense Block, used inside the :class:`RRDB` block of ESRGAN.

    Args:
        num_feat: Channel number of intermediate features.
        num_grow_ch: Channels for each growth.

    """

    def __init__(self, num_feat: int = 64, num_grow_ch: int = 32) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # initialization
        _default_init_weights([self.conv1, self.conv2, self.conv3, self.conv4, self.conv5], 0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the dense block on ``x`` of shape :math:`(B, C, H, W)`."""
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        # Empirically, we use 0.2 to scale the residual for better performance
        return x5 * 0.2 + x


class RRDB(nn.Module):
    """Residual in Residual Dense Block, used in RRDB-Net of ESRGAN.

    Args:
        num_feat: Channel number of intermediate features.
        num_grow_ch: Channels for each growth.

    """

    def __init__(self, num_feat: int, num_grow_ch: int = 32) -> None:
        super().__init__()
        self.rdb1 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb2 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb3 = ResidualDenseBlock(num_feat, num_grow_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the three stacked dense blocks on ``x`` of shape :math:`(B, C, H, W)`."""
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        # Empirically, we use 0.2 to scale the residual for better performance
        return out * 0.2 + x


class RRDBNet(nn.Module):
    r"""Network consisting of Residual in Residual Dense Blocks, as used in ESRGAN and Real-ESRGAN.

    ESRGAN is extended here for scale x2 and scale x1. For those scales the input is first
    pixel-unshuffled -- the inverse of a pixel shuffle -- to reduce the spatial size and enlarge the
    channel size before it is fed into the main ESRGAN architecture, so the network always upsamples
    by a factor of 4 internally.

    Args:
        num_in_ch: Channel number of inputs.
        num_out_ch: Channel number of outputs.
        scale: Upsampling factor. One of ``1``, ``2`` or ``4``.
        num_feat: Channel number of intermediate features.
        num_block: Block number in the trunk network.
        num_grow_ch: Channels for each growth.

    Shape:
        - Input: :math:`(B, C_{in}, H, W)`. For ``scale=2`` both spatial sizes must be divisible by
          2, and for ``scale=1`` by 4.
        - Output: :math:`(B, C_{out}, H \cdot scale, W \cdot scale)`.

    Example:
        >>> import torch
        >>> model = RRDBNet(num_in_ch=3, num_out_ch=3, scale=4, num_feat=8, num_block=1, num_grow_ch=4)
        >>> model(torch.rand(1, 3, 8, 8)).shape
        torch.Size([1, 3, 32, 32])

    """

    def __init__(
        self,
        num_in_ch: int,
        num_out_ch: int,
        scale: int = 4,
        num_feat: int = 64,
        num_block: int = 23,
        num_grow_ch: int = 32,
    ) -> None:
        super().__init__()
        self.scale = scale
        if scale == 2:
            num_in_ch = num_in_ch * 4
        elif scale == 1:
            num_in_ch = num_in_ch * 16
        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body = nn.Sequential(*[RRDB(num_feat=num_feat, num_grow_ch=num_grow_ch) for _ in range(num_block)])
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        # upsample
        self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Super-resolve ``x`` of shape :math:`(B, C_{in}, H, W)` by ``self.scale``."""
        if self.scale == 2:
            feat = F.pixel_unshuffle(x, downscale_factor=2)
        elif self.scale == 1:
            feat = F.pixel_unshuffle(x, downscale_factor=4)
        else:
            feat = x
        feat = self.conv_first(feat)
        body_feat = self.conv_body(self.body(feat))
        feat = feat + body_feat
        # upsample
        feat = self.lrelu(self.conv_up1(F.interpolate(feat, scale_factor=2, mode="nearest")))
        feat = self.lrelu(self.conv_up2(F.interpolate(feat, scale_factor=2, mode="nearest")))
        return self.conv_last(self.lrelu(self.conv_hr(feat)))
