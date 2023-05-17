"""Based on code from
https://github.com/PaddlePaddle/PaddleDetection/blob/ec37e66685f3bc5a38cd13f60685acea175922e1/
ppdet/modeling/transformers/rtdetr_transformer.py."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from kornia.contrib.models.common import MLP, ConvNormAct
from kornia.core import Module, Tensor, concatenate
from kornia.utils import create_meshgrid


class DeformableAttention(Module):
    """Deformable Attention used in Deformable DETR, as described in https://arxiv.org/abs/2010.04159."""

    def __init__(self, embed_dim: int, num_heads: int, num_levels: int, num_points: int):
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
        sampling_offsets = self.sampling_offsets(query).view(N, Lq, self.num_heads, self.num_levels, self.num_points, 2)
        attention_weights = self.attention_weights(query).view(N, Lq, self.num_heads, self.num_levels * self.num_points)
        attention_weights = attention_weights.softmax(-1).view(N, Lq, self.num_heads, self.num_levels, self.num_points)

        # (N, Lq, num_heads, num_levels, num_points, 2)
        ref_points_cxcy = ref_points[:, :, None, :, None, :2]
        ref_points_wh = ref_points[:, :, None, :, None, 2:]
        sampling_locations = ref_points_cxcy + sampling_offsets / self.num_points * ref_points_wh * 0.5

        # https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.6/ppdet/modeling/transformers/utils.py#L71
        # https://github.com/fundamentalvision/Deformable-DETR/blob/main/models/ops/functions/ms_deform_attn_func.py#L41
        split_size = [H * W for H, W in value_spatial_shapes]
        value_list = self.value_proj(value).split(split_size, 1)
        sampling_grids = 2 * sampling_locations - 1

        # NOTE: this can be sped up with torch.vmap() ?
        sampling_value_list = []
        for level, (H, W) in enumerate(value_spatial_shapes):
            # (N * num_heads, head_dim, H, W)
            value_l_ = value_list[level].permute(0, 2, 1).reshape(N * self.num_heads, -1, H, W)
            # (N * num_heads, Lq, num_points, 2)
            sampling_grid_l_ = sampling_grids[:, :, :, level].permute(0, 2, 1, 3, 4).flatten(0, 1)
            # (N * num_heads, head_dim, Lq, num_points)
            sampling_value_list.append(F.grid_sample(value_l_, sampling_grid_l_, "bilinear", "zeros", False))

        # (N * num_heads, head_dim, Lq, num_levels * num_points)
        sampling_value = concatenate(sampling_value_list, -1)

        # NOTE: probably can be simplified and sped up with einsum
        attention_weights = attention_weights.permute(0, 2, 1, 3, 4)
        attention_weights = attention_weights.reshape(N * self.num_heads, 1, Lq, self.num_levels * self.num_points)
        out = (sampling_value * attention_weights).sum(-1)
        out = out.reshape(N, C, Lq).permute(0, 2, 1)
        return self.output_proj(out)


# this is similar to nn.TransformerDecoderLayer, but replace cross attention with deformable attention
# add use positional embeddings
# https://github.com/fundamentalvision/Deformable-DETR/blob/main/models/deformable_transformer.py#L261
class DeformableTransformerDecoderLayer(Module):
    """Deformable Transformer Decoder layer used in Deformable DETR, as described in
    https://arxiv.org/abs/2010.04159."""

    def __init__(self, embed_dim: int, num_levels: int, num_heads: int = 8, num_points: int = 4, dropout: float = 0.0):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(embed_dim)

        self.cross_attn = DeformableAttention(embed_dim, num_heads, num_levels, num_points)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(embed_dim)

        self.linear1 = nn.Linear(embed_dim, embed_dim * 4)
        self.act = nn.ReLU(inplace=True)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(embed_dim * 4, embed_dim)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(embed_dim)

    def forward(
        self,
        tgt: Tensor,
        ref_points: Tensor,
        memory: Tensor,
        memory_spatial_shapes: list[tuple[int, int]],
        query_pos_embed: Tensor,
    ) -> Tensor:
        q = k = tgt + query_pos_embed
        out, _ = self.self_attn(q, k, tgt, need_weights=False)
        tgt = self.norm1(tgt + self.dropout1(out))

        out = self.cross_attn(tgt + query_pos_embed, ref_points, memory, memory_spatial_shapes)
        tgt = self.norm2(tgt + self.dropout2(out))

        return self.norm3(tgt + self.dropout4(self.ffn(tgt)))

    def ffn(self, x: Tensor) -> Tensor:
        return self.linear2(self.dropout3(self.act(self.linear1(x))))


class RTDETRHead(Module):
    def __init__(
        self,
        num_classes: int,
        hidden_dim: int,
        num_queries: int,
        in_channels: list[int],
        num_heads: int = 8,
        num_decoder_points: int = 4,
        num_decoder_layers: int = 6,
    ):
        super().__init__()
        self.num_queries = num_queries
        self.input_proj = nn.ModuleList()
        for ch_in in in_channels:
            self.input_proj.append(ConvNormAct(ch_in, hidden_dim, 1, act="none"))

        self.enc_output = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim))
        self.enc_score_head = nn.Linear(hidden_dim, num_classes)
        self.enc_bbox_head = MLP(hidden_dim, hidden_dim, 4, 3)

        self.denoising_class_embed = nn.Embedding(num_classes, hidden_dim)  # not used in evaluation
        self.query_pos_head = MLP(4, hidden_dim * 2, hidden_dim, 2)

        self.decoder_layers = nn.ModuleList()
        self.dec_score_head = nn.ModuleList()
        self.dec_bbox_head = nn.ModuleList()
        for _ in range(num_decoder_layers):
            layer = DeformableTransformerDecoderLayer(hidden_dim, len(in_channels), num_heads, num_decoder_points)
            self.decoder_layers.append(layer)
            self.dec_score_head.append(nn.Linear(hidden_dim, num_classes))
            self.dec_bbox_head.append(MLP(hidden_dim, hidden_dim, 4, 3))

    def forward(self, fmaps: list[Tensor]) -> tuple[Tensor, Tensor]:
        N = fmaps[0].shape[0]
        fmaps = [proj(fmap) for proj, fmap in zip(self.input_proj, fmaps)]
        spatial_shapes = [(fmap.shape[2], fmap.shape[3]) for fmap in fmaps]

        feats = [fmap.flatten(2).permute(0, 2, 1) for fmap in fmaps]  # (N, C, H, W) -> (N, H*W, C)
        memory = concatenate(feats, 1)  # rename to match original impl

        # TODO: cache anchors and valid_mask as buffers
        anchors, valid_mask = self.generate_anchors(spatial_shapes, device=memory.device, dtype=memory.dtype)

        # NOTE: why not use the mask to gather the values here, instead of filling zeros with torch.where()?
        zero = torch.zeros([], device=memory.device, dtype=memory.dtype)
        out_memory = torch.where(valid_mask, memory, zero)
        out_memory = self.enc_output(out_memory)

        enc_out_logits = self.enc_score_head(out_memory)
        enc_out_bboxes = anchors + self.enc_bbox_head(out_memory)

        # only consider class with highest score at each spatial location
        _, topk_indices = enc_out_logits.max(-1)[0].topk(self.num_queries, 1)  # (N, num_queries)
        batch_indices = torch.arange(N, dtype=topk_indices.dtype, device=topk_indices.device).unsqueeze(1)
        ref_points = enc_out_bboxes[batch_indices, topk_indices]
        tgt = out_memory[batch_indices, topk_indices]

        # iterative box refinements
        for decoder_layer, bbox_head in zip(self.decoder_layers, self.dec_bbox_head):
            ref_points_sigmoid = ref_points.sigmoid()
            query_pos_embed = self.query_pos_head(ref_points_sigmoid)
            tgt = decoder_layer(tgt, ref_points_sigmoid.unsqueeze(2), memory, spatial_shapes, query_pos_embed)
            ref_points = ref_points + bbox_head(tgt)
        logits = self.dec_score_head[-1](tgt)  # in evaluation, only last score head is used

        return ref_points.sigmoid(), logits

    @staticmethod
    def generate_anchors(
        spatial_shapes: list[tuple[int, int]],
        grid_size: float = 0.05,
        eps: float = 0.01,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> tuple[Tensor, Tensor]:
        # TODO: might make this into a separate reusable function
        anchors_list = []
        for i, (h, w) in enumerate(spatial_shapes):
            grid_xy = create_meshgrid(h, w, normalized_coordinates=False, device=device, dtype=dtype)
            grid_xy = (grid_xy + 0.5) / torch.tensor([h, w], device=device, dtype=dtype)
            wh = torch.ones_like(grid_xy) * grid_size * 2**i
            anchors_list.append(concatenate([grid_xy, wh], -1).reshape(-1, h * w, 4))

        anchors = concatenate(anchors_list, 1)
        valid_mask = ((anchors > eps) & (anchors < 1 - eps)).all(-1, keepdim=True)
        anchors = torch.log(anchors / (1 - anchors))

        # anchors = torch.where(valid_mask, anchors, float("inf")) fails in PyTorch 1.9.1
        inf = torch.tensor(float('inf'), device=device, dtype=dtype)
        anchors = torch.where(valid_mask, anchors, inf)
        return anchors, valid_mask
