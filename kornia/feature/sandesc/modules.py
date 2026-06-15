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

"""Reusable network building blocks: UNet blocks and CBAM attention."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

# --------------------------------------------------------
#                   HELPERS
# --------------------------------------------------------


def get_norm(norm: str, ch_in: int) -> nn.Module:
    """Returns a normalization layer given a string."""
    if norm not in ("batch", "instance", "group", None):
        raise ValueError(f"Norm type {norm} not recognized")

    norms = {
        "batch": nn.BatchNorm2d(ch_in),
        "instance": nn.InstanceNorm2d(ch_in, affine=True),
        "group": nn.GroupNorm(num_groups=ch_in // 16, num_channels=ch_in),
        None: nn.Identity(),
    }

    return norms[norm]


def get_activation(activation: str) -> nn.Module:
    """Return an activation layer given a string."""
    if activation not in ("relu", "gelu", None):
        raise ValueError(f"Activation type {activation} not recognized")
    activations = {
        "relu": nn.ReLU(inplace=False),
        "gelu": nn.GELU(),
        None: nn.Identity(),
    }
    return activations[activation]


# --------------------------------------------------------
#                   UNET MODULES
# --------------------------------------------------------
class UNetBlock(nn.Module):
    """Pre-activation block for the UNet, similar to DISK.

    Why pre-activation? See https://arxiv.org/abs/1603.05027.
    """

    def __init__(
        self,
        ch_in: int,
        ch_out: int,
        kernel_size: int = 5,
        norm: str = "batch",
        activation: str = "relu",
    ) -> None:
        """Build a norm -> activation -> conv pre-activation block."""
        super().__init__()

        self.conv = nn.Conv2d(ch_in, ch_out, kernel_size, stride=1, padding=kernel_size // 2)
        self.norm = get_norm(norm, ch_in)
        self.activation = get_activation(activation)

    def forward(self, x: Tensor) -> Tensor:
        """Apply norm, activation and convolution to x."""
        x = self.norm(x)
        x = self.activation(x)
        return self.conv(x)


class UNetDownBlock(nn.Module):
    """Downsampling UNet block with optional skip and attention."""

    def __init__(
        self,
        ch_in: int,
        ch_out: int,
        kernel_size: int = 5,
        activation: str | None = "relu",
        norm: str = "batch",
        third_block: bool = False,
        skip_connection: bool = False,
        spatial_attention: bool = False,
    ) -> None:
        """Build a downsampling block.

        Args:
            ch_in: Number of input channels.
            ch_out: Number of output channels.
            kernel_size: Kernel size for the convolutions.
            activation: Activation function.
            norm: Normalization layer.
            third_block: If True, add a third UNet block.
            skip_connection: If True, add a skip connection. If False, the same
                unet as in DISK and S-TReK.
            spatial_attention: If True, add a spatial attention module. Works
                only if skip_connection is True.
        """
        super().__init__()
        self.skip_connection = skip_connection

        self.block1 = UNetBlock(ch_in, ch_out, kernel_size, norm, activation)
        if skip_connection:
            self.align = nn.Conv2d(ch_in, ch_out, kernel_size=1, padding=0, bias=False)
            self.block2 = UNetBlock(ch_out, ch_out, kernel_size, norm, activation)
            self.block3 = UNetBlock(ch_out, ch_out, kernel_size, norm, activation) if third_block else nn.Identity()

        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        # Spatial attention
        self.cbam = CBAM(gate_channels=ch_out) if spatial_attention else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        """Downsample x and apply the block(s)."""
        x_ = self.pool(x)
        x = self.block1(x_)
        if self.skip_connection:
            x = self.block2(x)  # second block
            x = self.block3(x)  # third block
            x = self.cbam(x)  # spatial attention
            x = self.align(x_) + x  # skip connection
        return x


class UNetUpBlock(nn.Module):
    """Upsampling UNet block with optional skip and attention."""

    def __init__(
        self,
        ch_in: int,
        ch_out: int,
        kernel_size: int = 5,
        activation: str | None = None,
        norm: str = "batch",
        third_block: bool = False,
        skip_connection: bool = False,
        spatial_attention: bool = False,
    ) -> None:
        """Build an upsampling block mirroring UNetDownBlock."""
        super().__init__()
        self.skip_connection = skip_connection
        self.block1 = UNetBlock(ch_in, ch_out, kernel_size, norm, activation)
        if skip_connection:
            self.align = nn.Conv2d(ch_out, ch_out, kernel_size=1, padding=0, bias=False)
            self.block2 = UNetBlock(ch_out, ch_out, kernel_size, norm, activation)
            self.block3 = UNetBlock(ch_out, ch_out, kernel_size, norm, activation) if third_block else nn.Identity()

        self.upsample_2x = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        # Spatial attention
        self.cbam = CBAM(gate_channels=ch_out) if spatial_attention else nn.Identity()

    def forward(self, x: Tensor, x_from_past: Tensor | None = None) -> Tensor:
        """Upsample x, concatenate the skip tensor and apply the block(s)."""
        x_ = self.upsample_2x(x)  # c -> c
        x = torch.cat([x_, x_from_past], dim=1)
        x = self.block1(x)
        if self.skip_connection:
            x = self.block2(x)  # second block
            x = self.block3(x)  # third block
            x = self.cbam(x)  # spatial attention
            x = self.align(x_) + x  # skip connection
        return x


# --------------------------------------------------------
#                   Spatial Attention
# --------------------------------------------------------
# from: https://github.com/Jongchan/attention-module/blob/master/MODELS/cbam.py


class BasicConv(nn.Module):
    """Convolution followed by optional batch norm and activation."""

    def __init__(
        self,
        in_planes: int,
        out_planes: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        relu: bool = True,
        bn: bool = True,
        bias: bool = False,
    ) -> None:
        """Build the conv (+ optional batch norm and ReLU) block."""
        super().__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.GELU() if relu else None

    def forward(self, x: Tensor) -> Tensor:
        """Apply conv, then optional norm and activation."""
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class Flatten(nn.Module):
    """Flatten all dimensions except the batch dimension."""

    def forward(self, x: Tensor) -> Tensor:
        """Flatten x to shape [batch, -1]."""
        return x.view(x.size(0), -1)


class ChannelGate(nn.Module):
    """Channel attention gate of CBAM."""

    def __init__(
        self,
        gate_channels: int,
        reduction_ratio: int = 16,
        pool_types: list[str] | None = None,
    ) -> None:
        """Build the channel attention MLP over the given pooling types."""
        super().__init__()
        if pool_types is None:
            pool_types = ["avg"]
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.GELU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels),
        )
        self.pool_types = pool_types

    def forward(self, x: Tensor) -> Tensor:
        """Reweight channels of x by pooled channel attention."""
        channel_att_sum = None
        b, c = x.size()[0], x.size()[1]
        for pool_type in self.pool_types:
            if pool_type == "avg":
                pool = F.adaptive_avg_pool2d(x, 1)
            elif pool_type == "max":
                pool = F.adaptive_max_pool2d(x, 1)
            else:
                continue
            channel_att_raw = self.mlp(pool)
            channel_att_sum = channel_att_raw if channel_att_sum is None else channel_att_sum + channel_att_raw

        scale = torch.sigmoid(channel_att_sum).view(b, c, 1, 1)
        return x * scale


class ChannelPool(nn.Module):
    """Pool channels into max and mean feature maps."""

    def forward(self, x: Tensor) -> Tensor:
        """Concatenate channel-wise max and mean of x."""
        return torch.cat(
            (torch.max(x, dim=1, keepdim=True)[0], torch.mean(x, dim=1, keepdim=True)),
            dim=1,
        )


class SpatialGate(nn.Module):
    """Spatial attention gate of CBAM."""

    def __init__(self) -> None:
        """Build the spatial attention convolution."""
        super().__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(
            2,
            1,
            kernel_size=kernel_size,
            stride=1,
            padding=(kernel_size - 1) // 2,
            relu=False,
            bn=True,
        )

    def forward(self, x: Tensor) -> Tensor:
        """Reweight spatial locations of x by spatial attention."""
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = torch.sigmoid(x_out)
        return x * scale


class CBAM(nn.Module):
    """Convolutional Block Attention Module (channel + spatial attention)."""

    def __init__(
        self,
        gate_channels: int,
        reduction_ratio: int = 16,
        pool_types: list[str] | None = None,
        use_spatial: bool = True,
    ) -> None:
        """Build the channel and (optional) spatial attention gates."""
        super().__init__()
        if pool_types is None:
            pool_types = ["avg", "max"]

        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
        self.SpatialGate = SpatialGate() if use_spatial else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        """Apply channel then spatial attention to x."""
        x_out = self.ChannelGate(x)
        return self.SpatialGate(x_out)
