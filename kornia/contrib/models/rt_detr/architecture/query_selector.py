"""
https://github.com/PaddlePaddle/PaddleDetection/blob/5d1f888362241790000950e2b63115dc8d1c6019/ppdet/modeling/transformers/rtdetr_transformer.py
"""

from __future__ import annotations

from torch import nn

from kornia.core import Module, Tensor

from .common import MLP, conv_norm_act


class QuerySelector(Module):
    def __init__(
        self, num_classes: int, num_queries: int, in_channels: list[int], hidden_dim: int, num_decoder_layers: int
    ):
        super().__init__()
        self.projs = nn.ModuleList()
        for ch_in in in_channels:
            self.projs.append(conv_norm_act(ch_in, hidden_dim, act="none"))

        decoder_layer = nn.TransformerDecoderLayer(hidden_dim, 8, hidden_dim * 4, 0, "relu")
        self.decoder = nn.TransformerDecoder(decoder_layer, 6)

        self.class_embed = nn.Embedding(num_classes, hidden_dim)

        self.tgt_embed = nn.Embedding(num_queries, hidden_dim)
        self.query_pos_head = MLP(4, hidden_dim * 2, hidden_dim, 2)

        self.enc_output = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim))
        self.enc_score_head = nn.Linear(hidden_dim, num_classes)
        self.enc_bbox_head = MLP(hidden_dim, hidden_dim, 4, 3)

        self.dec_score_head = nn.ModuleList()
        self.dec_bbox_head = nn.ModuleList()
        for _ in range(num_decoder_layers):
            self.dec_score_head.append(nn.Linear(hidden_dim, num_classes))
            self.dec_bbox_head.append(MLP(hidden_dim, hidden_dim, 4, 3))

    def forward(self, fmaps: list[Tensor]):
        pass
