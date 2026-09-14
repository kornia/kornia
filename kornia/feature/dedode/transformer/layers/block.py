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

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# References:
#   https://github.com/facebookresearch/dino/blob/master/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/layers/patch_embed.py

from typing import Callable

import torch
from torch import Tensor, nn

from .attention import Attention
from .drop_path import DropPath
from .layer_scale import LayerScale
from .mlp import Mlp


class Block(nn.Module):
    """Implement a transformer block with attention and feed-forward sublayers.

    Args:
        dim: Embedding dimension of the input and output features.
        num_heads: Number of attention heads.
        mlp_ratio: Expansion ratio used to compute the hidden dimension of the feed-forward network
            as ``int(dim * mlp_ratio)``.
        qkv_bias: If True, add a learnable bias to the query, key and value projections.
        proj_bias: If True, add a learnable bias to the output projection of the attention layer.
        ffn_bias: If True, add a learnable bias to the linear layers in the feed-forward network.
        drop: Dropout probability applied after attention projection and inside the feed-forward network.
        attn_drop: Dropout probability applied to the attention weights.
        init_values: Initial value for the :class:`LayerScale` modules. If falsy, LayerScale is disabled
            and an identity mapping is used instead.
        drop_path: Stochastic depth probability for dropping the residual branch.
        act_layer: Callable that constructs the activation layer used in the feed-forward network.
        norm_layer: Callable that constructs the normalization layers applied before attention and
            feed-forward sublayers.
        attn_class: Callable that constructs the attention module.
        ffn_layer: Callable that constructs the feed-forward network module.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        ffn_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        init_values=None,
        drop_path: float = 0.0,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
        attn_class: Callable[..., nn.Module] = Attention,
        ffn_layer: Callable[..., nn.Module] = Mlp,
    ) -> None:
        super().__init__()
        # print(f"biases: qkv: {qkv_bias}, proj: {proj_bias}, ffn: {ffn_bias}")
        self.norm1 = norm_layer(dim)
        self.attn = attn_class(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = ffn_layer(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
            bias=ffn_bias,
        )
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.sample_drop_ratio = drop_path

    def forward(self, x: Tensor) -> Tensor:
        """Run this DeDoDe module forward.

        Args:
            x: Input token sequence with shape :math:`(B, N, C)`, where ``B`` is batch size,
               ``N`` is token count, and ``C`` is embedding dimension.

        Returns:
            Output token sequence with the same shape :math:`(B, N, C)`.
        """

        def attn_residual_func(x: Tensor) -> Tensor:
            return self.ls1(self.attn(self.norm1(x)))

        def ffn_residual_func(x: Tensor) -> Tensor:
            return self.ls2(self.mlp(self.norm2(x)))

        if self.training and self.sample_drop_ratio > 0.1:
            # the overhead is compensated only for a drop path rate larger than 0.1
            x = drop_add_residual_stochastic_depth(
                x,
                residual_func=attn_residual_func,
                sample_drop_ratio=self.sample_drop_ratio,
            )
            x = drop_add_residual_stochastic_depth(
                x,
                residual_func=ffn_residual_func,
                sample_drop_ratio=self.sample_drop_ratio,
            )
        elif self.training and self.sample_drop_ratio > 0.0:
            x = x + self.drop_path1(attn_residual_func(x))
            x = x + self.drop_path2(ffn_residual_func(x))
        else:
            x = x + attn_residual_func(x)
            x = x + ffn_residual_func(x)
        return x


def drop_add_residual_stochastic_depth(
    x: Tensor,
    residual_func: Callable[[Tensor], Tensor],
    sample_drop_ratio: float = 0.0,
) -> Tensor:
    """Add residual connection."""
    # 1) extract subset using permutation
    b, _n, _d = x.shape
    sample_subset_size = max(int(b * (1 - sample_drop_ratio)), 1)
    brange = (torch.randperm(b, device=x.device))[:sample_subset_size]
    x_subset = x[brange]

    # 2) apply residual_func to get residual
    residual = residual_func(x_subset)

    x_flat = x.flatten(1)
    residual = residual.flatten(1)

    residual_scale_factor = b / sample_subset_size

    # 3) add the residual
    x_plus_residual = torch.index_add(x_flat, 0, brange, residual.to(dtype=x.dtype), alpha=residual_scale_factor)
    return x_plus_residual.view_as(x)


class NestedTensorBlock(Block):
    """Implement a transformer block over a single token tensor.

    The name is kept because the vendored DINOv2 model builders reference it. The nested-tensor
    list path this class used to offer needed an optional third-party dependency kornia never
    declared, so it was never reachable and has been removed.
    """

    def forward(self, x_or_x_list):
        """Run this DeDoDe module forward.

        Args:
            x_or_x_list: A single token tensor with shape :math:`(B, N, C)`.

        Returns:
            Output token tensor with the same shape as the input.

        Raises:
            TypeError: If the input is not a :class:`torch.Tensor`.
        """
        if isinstance(x_or_x_list, Tensor):
            return super().forward(x_or_x_list)
        raise TypeError(
            "NestedTensorBlock only accepts a Tensor; the nested-tensor list path was removed, "
            f"got {type(x_or_x_list).__name__}."
        )
