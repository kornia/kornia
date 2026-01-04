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

# EfficientViT: Multi-Scale Linear Attention for High-Resolution Dense Prediction
# Han Cai, Junyan Li, Muyan Hu, Chuang Gan, Song Han
# International Conference on Computer Vision (ICCV), 2023

# type: ignore
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from kornia.models.efficient_vit.nn.act import build_act
from kornia.models.efficient_vit.nn.norm import build_norm
from kornia.models.efficient_vit.utils import get_same_padding, val2tuple
from kornia.utils._compat import autocast

__all__ = [
    "ConvLayer",
    "DSConv",
    "EfficientViTBlock",
    "FusedMBConv",
    "IdentityLayer",
    "MBConv",
    "OpSequential",
    "ResBlock",
    "ResidualBlock",
]

#################################################################################
#                             Basic Layers                                      #
#################################################################################


class ConvLayer(nn.Module):
    """Implement a standard convolutional layer with Batch Normalization and Activation.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        kernel_size: Size of the convolving kernel.
        stride: Stride of the convolution. Default: 1.
        groups: Number of blocked connections from input to output. Default: 1.
        use_bias: Whether to include a bias term in the convolution. Default: False.
        dropout: Dropout rate to apply before convolution. Default: 0 (no dropout).
        norm: Normalization layer type. Default: :class:`bn2d` (BatchNorm2d).
        act_func: Activation layer type. Default: :class:`relu` (ReLU).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size=3,
        stride=1,
        dilation=1,
        groups=1,
        use_bias=False,
        dropout=0,
        norm="bn2d",
        act_func="relu",
    ):
        super().__init__()

        padding = get_same_padding(kernel_size)
        padding *= dilation

        self.dropout = nn.Dropout2d(dropout, inplace=False) if dropout > 0 else None
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, kernel_size),
            stride=(stride, stride),
            padding=padding,
            dilation=(dilation, dilation),
            groups=groups,
            bias=use_bias,
        )
        self.norm = build_norm(norm, num_features=out_channels)
        self.act = build_act(act_func)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.dropout is not None:
            x = self.dropout(x)
        x = self.conv(x)
        if self.norm:
            x = self.norm(x)
        if self.act:
            x = self.act(x)
        return x


class IdentityLayer(nn.Module):
    """Implement a placeholder layer that returns the input as-is."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


#################################################################################
#                             Basic Blocks                                      #
#################################################################################


class DSConv(nn.Module):
    """Implement Depthwise Separable Convolution.

    This layer splits a standard convolution into a depthwise convolution
    followed by a pointwise convolution to reduce parameters and FLOPs.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        kernel_size: Size of the convolving kernel. Default: 3.
        stride: Stride of the convolution. Default: 1.
        use_bias: Whether to use bias in the convolutional layers. Default: False.
        norm: Normalization layers for the depthwise and pointwise stages. Default: ("bn2d", "bn2d").
        act_func: Activation functions for the depthwise and pointwise stages. Default: ("relu6", None).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size=3,
        stride=1,
        use_bias=False,
        norm=("bn2d", "bn2d"),
        act_func=("relu6", None),
    ):
        super().__init__()

        use_bias = val2tuple(use_bias, 2)
        norm = val2tuple(norm, 2)
        act_func = val2tuple(act_func, 2)

        self.depth_conv = ConvLayer(
            in_channels,
            in_channels,
            kernel_size,
            stride,
            groups=in_channels,
            norm=norm[0],
            act_func=act_func[0],
            use_bias=use_bias[0],
        )
        self.point_conv = ConvLayer(
            in_channels, out_channels, 1, norm=norm[1], act_func=act_func[1], use_bias=use_bias[1]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depth_conv(x)
        x = self.point_conv(x)
        return x


class MBConv(nn.Module):
    """Implement the Inverted Residual Block (Mobile Inverted Bottleneck).

    This block follows the MobileNetV2 design: a wide intermediate layer
    (expansion) between two narrow layers, using depthwise convolutions
    to maintain efficiency.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        kernel_size: Size of the depthwise convolution kernel. Default: 3.
        stride: Stride of the depthwise convolution. Default: 1.
        mid_channels: Number of intermediate expansion channels. If None, calculated
            using expand_ratio. Default: None.
        expand_ratio: Expansion factor for the intermediate channels relative to
            in_channels. Default: 6.
        use_bias: Whether to use bias in the convolutional layers. Default: False.
        norm: Normalization layers for the expansion, depthwise, and pointwise stages.
            Default: ("bn2d", "bn2d", "bn2d").
        act_func: Activation functions for the expansion, depthwise, and pointwise stages.
            Default: ("relu6", "relu6", None).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size=3,
        stride=1,
        mid_channels=None,
        expand_ratio=6,
        use_bias=False,
        norm=("bn2d", "bn2d", "bn2d"),
        act_func=("relu6", "relu6", None),
    ):
        super().__init__()

        use_bias = val2tuple(use_bias, 3)
        norm = val2tuple(norm, 3)
        act_func = val2tuple(act_func, 3)
        mid_channels = mid_channels or round(in_channels * expand_ratio)

        self.inverted_conv = ConvLayer(
            in_channels, mid_channels, 1, stride=1, norm=norm[0], act_func=act_func[0], use_bias=use_bias[0]
        )
        self.depth_conv = ConvLayer(
            mid_channels,
            mid_channels,
            kernel_size,
            stride=stride,
            groups=mid_channels,
            norm=norm[1],
            act_func=act_func[1],
            use_bias=use_bias[1],
        )
        self.point_conv = ConvLayer(
            mid_channels, out_channels, 1, norm=norm[2], act_func=act_func[2], use_bias=use_bias[2]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.inverted_conv(x)
        x = self.depth_conv(x)
        x = self.point_conv(x)
        return x


class FusedMBConv(nn.Module):
   """Implement a fused version of the Inverted Residual Block for efficiency.

    This block improves throughput on certain hardware by replacing the
    separate expansion and depthwise convolutions with a single regular
    convolution, as seen in EfficientNet-V2.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        kernel_size: Size of the main convolution kernel. Default: 3.
        stride: Stride of the convolution. Default: 1.
        mid_channels: Number of intermediate expansion channels. If None,
            calculated using expand_ratio. Default: None.
        expand_ratio: Expansion factor for the intermediate channels relative
            to in_channels. Default: 6.
        groups: Number of groups for the main convolution. Default: 1.
        use_bias: Whether to use bias in the convolutional layers. Default: False.
        norm: Normalization layers for the fused and pointwise stages.
            Default: ("bn2d", "bn2d").
        act_func: Activation functions for the fused and pointwise stages.
            Default: ("relu6", None).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size=3,
        stride=1,
        mid_channels=None,
        expand_ratio=6,
        groups=1,
        use_bias=False,
        norm=("bn2d", "bn2d"),
        act_func=("relu6", None),
    ):
        super().__init__()
        use_bias = val2tuple(use_bias, 2)
        norm = val2tuple(norm, 2)
        act_func = val2tuple(act_func, 2)

        mid_channels = mid_channels or round(in_channels * expand_ratio)

        self.spatial_conv = ConvLayer(
            in_channels,
            mid_channels,
            kernel_size,
            stride,
            groups=groups,
            use_bias=use_bias[0],
            norm=norm[0],
            act_func=act_func[0],
        )
        self.point_conv = ConvLayer(
            mid_channels, out_channels, 1, use_bias=use_bias[1], norm=norm[1], act_func=act_func[1]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.spatial_conv(x)
        x = self.point_conv(x)
        return x


class ResBlock(nn.Module):
    """Implement a standard residual block for EfficientViT.

    This block applies a series of convolutions and adds the original input
    back to the output via a skip connection, helping to mitigate the
    vanishing gradient problem in deep networks.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        kernel_size: Size of the convolving kernel. Default: 3.
        stride: Stride of the convolution. Default: 1.
        mid_channels: Number of intermediate channels. If None, calculated
            using expand_ratio. Default: None.
        expand_ratio: Expansion factor for the intermediate channels. Default: 1.
        use_bias: Whether to use bias in the convolutional layers. Default: False.
        norm: Normalization layers for the main and projection stages.
            Default: ("bn2d", "bn2d").
        act_func: Activation functions for the main and projection stages.
            Default: ("relu6", None).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size=3,
        stride=1,
        mid_channels=None,
        expand_ratio=1,
        use_bias=False,
        norm=("bn2d", "bn2d"),
        act_func=("relu6", None),
    ):
        super().__init__()
        use_bias = val2tuple(use_bias, 2)
        norm = val2tuple(norm, 2)
        act_func = val2tuple(act_func, 2)

        mid_channels = mid_channels or round(in_channels * expand_ratio)

        self.conv1 = ConvLayer(
            in_channels, mid_channels, kernel_size, stride, use_bias=use_bias[0], norm=norm[0], act_func=act_func[0]
        )
        self.conv2 = ConvLayer(
            mid_channels, out_channels, kernel_size, 1, use_bias=use_bias[1], norm=norm[1], act_func=act_func[1]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class LiteMLA(nn.Module):
    r"""Lightweight multi-scale linear attention.

    This module implements a linear attention mechanism that avoids the quadratic
    complexity of standard self-attention. It incorporates multi-scale depthwise
    convolutions (scales) to aggregate local information effectively.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        heads: Number of attention heads. If None, calculated using heads_ratio.
            Default: None.
        heads_ratio: Ratio to determine the number of heads relative to channels.
            Default: 1.0.
        dim: Dimension of each attention head. Default: 8.
        use_bias: Whether to use bias in the linear projections. Default: False.
        norm: Normalization layers for the query/key/value and output projections.
            Default: (None, "bn2d").
        act_func: Activation functions for the internal and output stages.
            Default: (None, None).
        kernel_func: The kernel function to use for linear attention (e.g., "relu").
            Default: "relu".
        scales: Kernel sizes for the multi-scale depthwise convolutions.
            Default: (5,).
        eps: A small value to ensure numerical stability during division.
            Default: 1.0e-15.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        heads: int or None = None,
        heads_ratio: float = 1.0,
        dim=8,
        use_bias=False,
        norm=(None, "bn2d"),
        act_func=(None, None),
        kernel_func="relu",
        scales: tuple[int, ...] = (5,),
        eps=1.0e-15,
    ):
        super().__init__()
        self.eps = eps
        heads = heads or int(in_channels // dim * heads_ratio)

        total_dim = heads * dim

        use_bias = val2tuple(use_bias, 2)
        norm = val2tuple(norm, 2)
        act_func = val2tuple(act_func, 2)

        self.dim = dim
        self.qkv = ConvLayer(in_channels, 3 * total_dim, 1, use_bias=use_bias[0], norm=norm[0], act_func=act_func[0])
        self.aggreg = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        3 * total_dim,
                        3 * total_dim,
                        scale,
                        padding=get_same_padding(scale),
                        groups=3 * total_dim,
                        bias=use_bias[0],
                    ),
                    nn.Conv2d(3 * total_dim, 3 * total_dim, 1, groups=3 * heads, bias=use_bias[0]),
                )
                for scale in scales
            ]
        )
        self.kernel_func = build_act(kernel_func, inplace=False)

        self.proj = ConvLayer(
            total_dim * (1 + len(scales)), out_channels, 1, use_bias=use_bias[1], norm=norm[1], act_func=act_func[1]
        )

    @autocast(enabled=False)
    def relu_linear_att(self, qkv: torch.Tensor) -> torch.Tensor:
        B, _, H, W = list(qkv.size())

        if qkv.dtype == torch.float16:
            qkv = qkv.float()

        qkv = torch.reshape(qkv, (B, -1, 3 * self.dim, H * W))
        qkv = torch.transpose(qkv, -1, -2)
        q, k, v = (qkv[..., 0 : self.dim], qkv[..., self.dim : 2 * self.dim], qkv[..., 2 * self.dim :])

        # lightweight linear attention
        q = self.kernel_func(q)
        k = self.kernel_func(k)

        # linear matmul
        trans_k = k.transpose(-1, -2)

        v = F.pad(v, (0, 1), mode="constant", value=1)
        kv = torch.matmul(trans_k, v)
        out = torch.matmul(q, kv)
        out = out[..., :-1] / (out[..., -1:] + self.eps)

        out = torch.transpose(out, -1, -2)
        out = torch.reshape(out, (B, -1, H, W))
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # generate multi-scale q, k, v
        qkv = self.qkv(x)
        multi_scale_qkv = [qkv]
        for op in self.aggreg:
            multi_scale_qkv.append(op(qkv))
        multi_scale_qkv = torch.cat(multi_scale_qkv, dim=1)

        out = self.relu_linear_att(multi_scale_qkv)
        out = self.proj(out)

        return out


class EfficientViTBlock(nn.Module):
    """Implement a single EfficientViT building block.

    This block consists of two main components: a Lightweight Multi-scale Linear
    Attention (LiteMLA) layer and a Feed-Forward Network (FFN). It follows the
    standard transformer-style residual architecture optimized for efficiency.

    Args:
        in_channels: Number of input channels.
        heads_ratio: Ratio to determine the number of attention heads in LiteMLA.
            Default: 1.0.
        dim: The dimension for the attention head and FFN layers. Default: 32.
        expand_ratio: Expansion factor for the FFN (Feed-Forward Network).
            Default: 4.
        norm: Normalization layer type to use (e.g., "bn2d"). Default: "bn2d".
        act_func: Activation function type to use (e.g., "hswish"). Default: "hswish".
    """

    def __init__(
        self,
        in_channels: int,
        heads_ratio: float = 1.0,
        dim=32,
        expand_ratio: float = 4,
        norm="bn2d",
        act_func="hswish",
    ):
        super().__init__()
        self.context_module = ResidualBlock(
            LiteMLA(
                in_channels=in_channels, out_channels=in_channels, heads_ratio=heads_ratio, dim=dim, norm=(None, norm)
            ),
            IdentityLayer(),
        )
        local_module = MBConv(
            in_channels=in_channels,
            out_channels=in_channels,
            expand_ratio=expand_ratio,
            use_bias=(True, True, False),
            norm=(None, None, norm),
            act_func=(act_func, act_func, None),
        )
        self.local_module = ResidualBlock(local_module, IdentityLayer())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.context_module(x)
        x = self.local_module(x)
        return x


#################################################################################
#                             Functional Blocks                                 #
#################################################################################


class ResidualBlock(nn.Module):
    """Provide a flexible residual wrapper for a main branch and a shortcut.

    Args:
        main: The primary neural network branch.
        shortcut: The identity or projection shortcut branch.
        post_act: Activation to apply after the summation.
        pre_norm: Normalization to apply before the branches.
    """

    def __init__(
        self, main: nn.Module | None, shortcut: nn.Module | None, post_act=None, pre_norm: nn.Module | None = None
    ):
        super().__init__()

        self.pre_norm = pre_norm
        self.main = main
        self.shortcut = shortcut
        self.post_act = build_act(post_act)

    def forward_main(self, x: torch.Tensor) -> torch.Tensor:
        if self.pre_norm is None:
            return self.main(x)
        else:
            return self.main(self.pre_norm(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.main is None:
            res = x
        elif self.shortcut is None:
            res = self.forward_main(x)
        else:
            res = self.forward_main(x) + self.shortcut(x)
            if self.post_act:
                res = self.post_act(res)
        return res


class OpSequential(nn.Module):
    """A container for sequential execution that handles optional or None modules.

    This class behaves similarly to :class:`nn.Sequential` but explicitly
    allows for ``None`` elements within the operation list, which are
    ignored during the forward pass.

    Args:
        op_list: A list of modules to be executed sequentially.
            Elements can be ``None``, in which case they act as an identity.
    """

    def __init__(self, op_list: list[nn.Module | None]):
        super().__init__()
        valid_op_list = []
        for op in op_list:
            if op is not None:
                valid_op_list.append(op)
        self.op_list = nn.ModuleList(valid_op_list)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for op in self.op_list:
            x = op(x)
        return x
