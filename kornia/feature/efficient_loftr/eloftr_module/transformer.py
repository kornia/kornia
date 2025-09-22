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

import copy
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import nn

from kornia.core import Module, Tensor
from kornia.feature.efficient_loftr.utils.position_encoding import RoPEPositionEncodingSine

from .linear_attention import Attention


class AG_RoPE_EncoderLayer(Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        agg_size0: int = 4,
        agg_size1: int = 4,
        no_flash: bool = False,
        rope: int = False,
        npe: Optional[List[int]] = None,
        fp32: bool = False,
    ) -> None:
        super().__init__()

        self.dim = d_model // nhead
        self.nhead = nhead
        self.agg_size0, self.agg_size1 = agg_size0, agg_size1
        self.rope = rope

        # aggregate and position encoding
        self.aggregate = (
            nn.Conv2d(d_model, d_model, kernel_size=agg_size0, padding=0, stride=agg_size0, bias=False, groups=d_model)
            if self.agg_size0 != 1
            else nn.Identity()
        )
        self.max_pool = (
            torch.nn.MaxPool2d(kernel_size=self.agg_size1, stride=self.agg_size1)
            if self.agg_size1 != 1
            else nn.Identity()
        )
        if self.rope:
            self.rope_pos_enc = RoPEPositionEncodingSine(d_model, max_shape=(256, 256), npe=npe, ropefp16=True)

        # multi-head attention
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.attention = Attention(no_flash, self.nhead, self.dim, fp32)
        self.merge = nn.Linear(d_model, d_model, bias=False)

        # feed-forward network
        self.mlp = nn.Sequential(
            nn.Linear(d_model * 2, d_model * 2, bias=False),
            nn.LeakyReLU(inplace=True),
            nn.Linear(d_model * 2, d_model, bias=False),
        )

        # norm
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: Tensor, source: Tensor, x_mask: Tensor = None, source_mask: Tensor = None) -> Tensor:
        """Positional encoding forwrd loop.

        Args:
            x (torch.Tensor): [N, C, H0, W0]
            source (torch.Tensor): [N, C, H1, W1]
            x_mask (torch.Tensor): [N, H0, W0] (optional) (L = H0*W0)
            source_mask (torch.Tensor): [N, H1, W1] (optional) (S = H1*W1)
        """
        bs, C, H0, W0 = x.size()
        # H1, W1 = source.size(-2), source.size(-1)

        # Aggregate feature
        query, source = (
            self.norm1(self.aggregate(x).permute(0, 2, 3, 1)),
            self.norm1(self.max_pool(source).permute(0, 2, 3, 1)),
        )  # [N, H, W, C]
        if x_mask is not None:
            x_mask, source_mask = (self.max_pool(x.float()).bool() for x in [x_mask, source_mask])
        query, key, value = self.q_proj(query), self.k_proj(source), self.v_proj(source)

        # Positional encoding
        if self.rope:
            query = self.rope_pos_enc(query)
            key = self.rope_pos_enc(key)

        # multi-head attention handle padding mask
        m = self.attention(query, key, value, q_mask=x_mask, kv_mask=source_mask)
        m = self.merge(m.reshape(bs, -1, self.nhead * self.dim))  # [N, L, C]

        # Upsample feature
        # m = rearrange(m, 'b (h w) c -> b c h w', h=H0 // self.agg_size0, w=W0 // self.agg_size0) # [N, C, H0, W0]
        b0, hw0, c0 = m.shape
        m = m.reshape(b0, -1, W0 // self.agg_size0, c0).permute(0, 3, 1, 2)

        if self.agg_size0 != 1:
            m = torch.nn.functional.interpolate(
                m, scale_factor=self.agg_size0, mode="bilinear", align_corners=False
            )  # [N, C, H0, W0]

        # feed-forward network
        m = self.mlp(torch.cat([x, m], dim=1).permute(0, 2, 3, 1))  # [N, H0, W0, C]
        m = self.norm2(m).permute(0, 3, 1, 2)  # [N, C, H0, W0]

        return x + m


class LocalFeatureTransformer(Module):
    """A Local Feature Transformer (LoFTR) module."""

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__()

        self.full_config = config
        self.fp32 = not (config["mp"] or config["half"])
        config = config["coarse"]
        self.d_model = config["d_model"]
        self.nhead = config["nhead"]
        self.layer_names = config["layer_names"]
        self.agg_size0, self.agg_size1 = config["agg_size0"], config["agg_size1"]
        self.rope = config["rope"]

        self_layer = AG_RoPE_EncoderLayer(
            config["d_model"],
            config["nhead"],
            config["agg_size0"],
            config["agg_size1"],
            config["no_flash"],
            config["rope"],
            config["npe"],
            self.fp32,
        )
        cross_layer = AG_RoPE_EncoderLayer(
            config["d_model"],
            config["nhead"],
            config["agg_size0"],
            config["agg_size1"],
            config["no_flash"],
            False,
            config["npe"],
            self.fp32,
        )
        self.layers = nn.ModuleList(
            [copy.deepcopy(self_layer) if _ == "self" else copy.deepcopy(cross_layer) for _ in self.layer_names]
        )
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self, feat0: Tensor, feat1: Tensor, mask0: Tensor = None, mask1: Tensor = None, data: Any = None
    ) -> Tuple[Tensor]:
        """Forward run.

        Args:
            feat0 (torch.Tensor): [N, C, H, W]
            feat1 (torch.Tensor): [N, C, H, W]
            mask0 (torch.Tensor): [N, L] (optional)
            mask1 (torch.Tensor): [N, S] (optional)
            data : (optional)
        """
        # H0, W0, H1, W1 = feat0.size(-2), feat0.size(-1), feat1.size(-2), feat1.size(-1)
        bs = feat0.shape[0]

        feature_cropped = False
        if bs == 1 and mask0 is not None and mask1 is not None:
            mask_H0, mask_W0, mask_H1, mask_W1 = mask0.size(-2), mask0.size(-1), mask1.size(-2), mask1.size(-1)
            mask_h0, mask_w0, mask_h1, mask_w1 = (
                mask0[0].sum(-2)[0],
                mask0[0].sum(-1)[0],
                mask1[0].sum(-2)[0],
                mask1[0].sum(-1)[0],
            )
            mask_h0, mask_w0, mask_h1, mask_w1 = (
                mask_h0 // self.agg_size0 * self.agg_size0,
                mask_w0 // self.agg_size0 * self.agg_size0,
                mask_h1 // self.agg_size1 * self.agg_size1,
                mask_w1 // self.agg_size1 * self.agg_size1,
            )
            feat0 = feat0[:, :, :mask_h0, :mask_w0]
            feat1 = feat1[:, :, :mask_h1, :mask_w1]
            feature_cropped = True

        for _i, (layer, name) in enumerate(zip(self.layers, self.layer_names)):
            if feature_cropped:
                mask0, mask1 = None, None
            if name == "self":
                feat0 = layer(feat0, feat0, mask0, mask0)
                feat1 = layer(feat1, feat1, mask1, mask1)
            elif name == "cross":
                feat0 = layer(feat0, feat1, mask0, mask1)
                feat1 = layer(feat1, feat0, mask1, mask0)
            else:
                raise KeyError

        if feature_cropped:
            # padding feature
            bs, c, mask_h0, mask_w0 = feat0.size()
            if mask_h0 != mask_H0:
                feat0 = torch.cat(
                    [feat0, torch.zeros(bs, c, mask_H0 - mask_h0, mask_W0, device=feat0.device, dtype=feat0.dtype)],
                    dim=-2,
                )
            elif mask_w0 != mask_W0:
                feat0 = torch.cat(
                    [feat0, torch.zeros(bs, c, mask_H0, mask_W0 - mask_w0, device=feat0.device, dtype=feat0.dtype)],
                    dim=-1,
                )

            bs, c, mask_h1, mask_w1 = feat1.size()
            if mask_h1 != mask_H1:
                feat1 = torch.cat(
                    [feat1, torch.zeros(bs, c, mask_H1 - mask_h1, mask_W1, device=feat1.device, dtype=feat1.dtype)],
                    dim=-2,
                )
            elif mask_w1 != mask_W1:
                feat1 = torch.cat(
                    [feat1, torch.zeros(bs, c, mask_H1, mask_W1 - mask_w1, device=feat1.device, dtype=feat1.dtype)],
                    dim=-1,
                )

        return feat0, feat1
