from __future__ import annotations

import copy
from typing import Any, Literal, Optional

import torch
from torch import nn

from kornia.core import Module, Tensor

from .linear_attention import FullAttention, LinearAttention


class LoFTREncoderLayer(Module):
    def __init__(self, d_model: int, nhead: int, attention: Optional[Literal["linear"]] = "linear") -> None:
        super().__init__()

        self.dim = d_model // nhead
        self.nhead = nhead

        # multi-head attention
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.attention = LinearAttention() if attention == "linear" else FullAttention()
        self.merge = nn.Linear(d_model, d_model, bias=False)

        # feed-forward network
        self.mlp = nn.Sequential(
            nn.Linear(d_model * 2, d_model * 2, bias=False), nn.ReLU(True), nn.Linear(d_model * 2, d_model, bias=False)
        )

        # norm and dropout
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(
        self, x: Tensor, source: Tensor, x_mask: Optional[Tensor] = None, source_mask: Optional[Tensor] = None
    ) -> Tensor:
        """
        Args:
            x: [N, L, C]
            source: [N, S, C]
            x_mask: [N, L] (optional)
            source_mask: [N, S] (optional)
        """
        bs = x.size(0)
        query, key, value = x, source, source

        # multi-head attention
        query = self.q_proj(query).view(bs, -1, self.nhead, self.dim)  # [N, L, (H, D)]
        key = self.k_proj(key).view(bs, -1, self.nhead, self.dim)  # [N, S, (H, D)]
        value = self.v_proj(value).view(bs, -1, self.nhead, self.dim)
        message = self.attention(query, key, value, q_mask=x_mask, kv_mask=source_mask)  # [N, L, (H, D)]
        message = self.merge(message.view(bs, -1, self.nhead * self.dim))  # [N, L, C]
        message = self.norm1(message)

        # feed-forward network
        message = self.mlp(torch.cat([x, message], dim=2))
        message = self.norm2(message)

        return x + message


class LocalFeatureTransformer(Module):
    """A Local Feature Transformer (LoFTR) module."""

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()

        self.config = config
        self.d_model = config["d_model"]
        self.nhead = config["nhead"]
        self.layer_names = config["layer_names"]
        encoder_layer = LoFTREncoderLayer(config["d_model"], config["nhead"], config["attention"])
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(len(self.layer_names))])
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self, feat0: Tensor, feat1: Tensor, mask0: None | Tensor = None, mask1: None | Tensor = None
    ) -> tuple[Tensor, Tensor]:
        """
        Args:
            feat0: [N, L, C]
            feat1: [N, S, C]
            mask0: [N, L] (optional)
            mask1: [N, S] (optional)
        """
        if self.d_model != feat0.size(2):
            msg = "the feature number of src and transformer must be equal"
            raise ValueError(msg)

        for layer, name in zip(self.layers, self.layer_names):
            if name == "self":
                feat0 = layer(feat0, feat0, mask0, mask0)
                feat1 = layer(feat1, feat1, mask1, mask1)
            elif name == "cross":
                feat0 = layer(feat0, feat1, mask0, mask1)
                feat1 = layer(feat1, feat0, mask1, mask0)
            else:
                raise KeyError

        return feat0, feat1
