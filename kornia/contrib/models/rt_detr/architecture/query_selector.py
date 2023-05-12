"""
https://github.com/PaddlePaddle/PaddleDetection/blob/5d1f888362241790000950e2b63115dc8d1c6019/ppdet/modeling/transformers/rtdetr_transformer.py
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from kornia.core import Module, Tensor, concatenate

from .common import MLP, conv_norm_act


class DeformableAttention(Module):
    def __init__(self, embed_dim: int = 256, num_heads: int = 8, num_levels: int = 4, num_points: int = 4):
        super().__init__()
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.num_points = num_points
        self.total_points = num_heads * num_levels * num_points

        self.sampling_offsets = nn.Linear(embed_dim, self.total_points * 2)
        self.attention_weights = nn.Linear(embed_dim, self.total_points)
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        self.output_proj = nn.Linear(embed_dim, embed_dim)

    def forward(
        self, query: Tensor, ref_points: Tensor, value: Tensor, value_spatial_shapes: list[tuple[int, int]]
    ) -> Tensor:
        """
        Args:
            query: shape (N, Lq, C)
            ref_points: shape (N, Lq, n_levels, 4)
            value: shape (N, Lv, C)
            value_spatial_shapes: [(H0, W0), (H1, W1), ...]
        """
        N, Lq, C = query.shape
        Lv = value.shape[1]

        value = self.value_proj(value).view(N, Lv, self.num_heads, -1)

        sampling_offsets = self.sampling_offsets(query).view(N, Lq, self.num_heads, self.num_levels, self.num_points, 2)
        attention_weights = self.attention_weights(query).view(N, Lq, self.num_heads, self.num_levels * self.num_points)
        attention_weights = attention_weights.softmax(-1).view(N, Lq, self.num_heads, self.num_levels, self.num_points)

        # (N, Lq, num_heads, num_levels, num_points, 2)
        ref_points_xy, ref_points_wh = ref_points.view(N, Lq, 1, -1, 1, 4).chunk(2, -1)
        sampling_locations = ref_points_xy + sampling_offsets / self.num_points * ref_points_wh * 0.5

        split_size = [h * w for h, w in value_spatial_shapes]
        value_list = value.split(split_size, 1)
        sampling_grids = 2 * sampling_locations - 1

        sampling_value_list = []
        for level, (H, W) in enumerate(value_spatial_shapes):
            # (N, H*W, C) -> (N * num_heads, head_dim, H, W)
            value_l_ = value_list[level].permute(0, 2, 1).reshape(N * self.num_heads, -1, H, W)
            sampling_grid_l_ = sampling_grids[:, :, :, level].permute(0, 2, 1, 3, 4).flatten(0, 1)
            sampling_value_list.append(F.grid_sample(value_l_, sampling_grid_l_))

        attention_weights = attention_weights.permute(0, 2, 1, 3, 4)
        attention_weights = attention_weights.reshape(N * self.num_heads, 1, Lq, self.num_levels * self.num_points)
        out = (torch.stack(sampling_value_list, -2).flatten(-2) * attention_weights).sum(-1)
        return out.view(N, C, Lq).permute(0, 2, 1)


class QuerySelector(Module):
    def __init__(
        self, num_classes: int, num_queries: int, in_channels: list[int], hidden_dim: int, num_decoder_layers: int
    ):
        super().__init__()
        self.num_queries = num_queries
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
        N = fmaps[0].shape[0]
        fmaps = [proj(fmap) for proj, fmap in zip(self.projs, fmaps)]

        feats, shapes, prefix = [], [], [0]
        for fmap in fmaps:
            H, W = fmaps.shape[2:]
            feats.append(fmap.flatten(2).permute(0, 2, 1))  # (N, C, H, W) -> (N, H*W, C)
            shapes.append([H, W])
            prefix.append(prefix[-1] + H * W)

        feats = concatenate(feats, 1)  # (N, H*W, C)
        prefix.pop()

        anchors, valid_mask = self.generate_anchors(shapes)
        feats = torch.where(valid_mask, feats, 0)
        feats = self.enc_output(feats)

        output_logits = self.enc_score_head(feats)
        output_bboxes = anchors + self.enc_bbox_head(feats)

        # only consider class with highest score at each spatial location
        topk_logits, topk_indices = output_logits.max(-1)[0].topk(self.num_queries, 1)  # (N, num_queries)
        batch_indices = torch.arange(N, dtype=topk_indices.dtype, device=topk_indices.device)

        # alternative
        # topk_bboxes = output_bboxes.gather(1, topk_indices.unsqueeze(-1).expand(-1, -1, 4))
        topk_bboxes = output_bboxes[batch_indices.unsqueeze(1), topk_indices]
        topk_bboxes = torch.sigmoid(topk_bboxes)

        target = self.tgt_embed.weight.unsqueeze(0).expand(N, -1, -1)

        ref_points = topk_bboxes
        for layer in self.decoder.layers:
            query_pos_embed = self.query_pos_head(ref_points)
            target = target + query_pos_embed

            # out = layer(target, ref_points.unsqueeze(2), feats, )

    def generate_anchors(self, shapes, grid_size=0.05, eps=0.01):
        anchors = []
        for i, (h, w) in enumerate(shapes):
            xs = torch.arange(w)
            ys = torch.arange(h)
            grid_x, grid_y = torch.meshgrid(xs, ys)
            grid_xy = torch.stack([grid_x, grid_y], -1)

            valid_wh = torch.tensor([h, w])  # wrong order?
            grid_xy = grid_xy.unsqueeze(0).add(0.5).div(valid_wh)
            wh = torch.ones_like(grid_xy) * grid_size * 2**i
            anchors.append(concatenate([grid_xy, wh], -1).reshape(-1, h * w, 4))

        anchors = concatenate(anchors, 1)
        valid_mask = ((anchors > eps) & (anchors < 1 - eps)).all(-1, keepdim=True)
        anchors = anchors.div(1 - anchors).log()
        anchors = torch.where(valid_mask, anchors, torch.inf)
        return anchors, valid_mask
