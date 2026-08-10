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


def get_norm(norm: str | None, ch_in: int) -> nn.Module:
    """Returns a normalization layer given a string."""
    if norm == "batch":
        return nn.BatchNorm2d(ch_in)
    if norm == "instance":
        return nn.InstanceNorm2d(ch_in, affine=True)
    if norm == "group":
        return nn.GroupNorm(num_groups=max(ch_in // 16, 1), num_channels=ch_in)
    if norm is None:
        return nn.Identity()
    raise ValueError(f"Norm type {norm} not recognized")


def get_activation(activation: str | None) -> nn.Module:
    """Return an activation layer given a string."""
    if activation == "relu":
        return nn.ReLU(inplace=False)
    if activation == "gelu":
        return nn.GELU()
    if activation is None:
        return nn.Identity()
    raise ValueError(f"Activation type {activation} not recognized")


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
        norm: str | None = "batch",
        activation: str | None = "relu",
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
        norm: str | None = "batch",
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
            self.cbam = CBAM(gate_channels=ch_out) if spatial_attention else nn.Identity()

        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

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
        ch_up: int,
        ch_skip: int,
        ch_out: int,
        kernel_size: int = 5,
        activation: str | None = None,
        norm: str | None = "batch",
        third_block: bool = False,
        skip_connection: bool = False,
        spatial_attention: bool = False,
    ) -> None:
        """Build an upsampling block mirroring UNetDownBlock.

        Args:
            ch_up: Number of channels of the tensor coming from the previous (coarser) block.
            ch_skip: Number of channels of the encoder tensor concatenated to it.
            ch_out: Number of output channels.
            kernel_size: Kernel size for the convolutions.
            activation: Activation function.
            norm: Normalization layer.
            third_block: If True, add a third UNet block.
            skip_connection: If True, add a skip connection.
            spatial_attention: If True, add a spatial attention module. Works
                only if skip_connection is True.
        """
        super().__init__()
        self.skip_connection = skip_connection
        self.block1 = UNetBlock(ch_up + ch_skip, ch_out, kernel_size, norm, activation)
        if skip_connection:
            # The residual path starts from the upsampled tensor, so it carries ch_up channels.
            self.align = nn.Conv2d(ch_up, ch_out, kernel_size=1, padding=0, bias=False)
            self.block2 = UNetBlock(ch_out, ch_out, kernel_size, norm, activation)
            self.block3 = UNetBlock(ch_out, ch_out, kernel_size, norm, activation) if third_block else nn.Identity()
            self.cbam = CBAM(gate_channels=ch_out) if spatial_attention else nn.Identity()

        self.upsample_2x = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)

    def forward(self, x: Tensor, x_from_past: Tensor) -> Tensor:
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
        relu: bool = True,
        bn: bool = True,
    ) -> None:
        """Build the conv (+ optional batch norm and activation) block."""
        super().__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
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
        unknown = [p for p in pool_types if p not in ("avg", "max")]
        if unknown:
            raise ValueError(f"Pool types {unknown} not recognized. Available: ['avg', 'max']")
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.GELU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels),
        )
        self.pool_types = pool_types

    def forward(self, x: Tensor) -> Tensor:
        """Reweight channels of x by pooled channel attention."""
        b, c = x.size()[0], x.size()[1]
        channel_att_sum = x.new_zeros(b, c)
        for pool_type in self.pool_types:
            pool = F.adaptive_avg_pool2d(x, 1) if pool_type == "avg" else F.adaptive_max_pool2d(x, 1)
            channel_att_sum = channel_att_sum + self.mlp(pool)

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
    ) -> None:
        """Build the channel and spatial attention gates."""
        super().__init__()
        if pool_types is None:
            pool_types = ["avg", "max"]

        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
        self.SpatialGate = SpatialGate()

    def forward(self, x: Tensor) -> Tensor:
        """Apply channel then spatial attention to x."""
        x_out = self.ChannelGate(x)
        return self.SpatialGate(x_out)
