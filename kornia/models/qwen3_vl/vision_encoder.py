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

from __future__ import annotations

from typing import NamedTuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .config import Qwen3VLConfig, Qwen3VLVisionConfig

__all__ = [
    "Qwen3VLAttention",
    "Qwen3VLBlock",
    "Qwen3VLMLP",
    "Qwen3VLPatchEmbed",
    "Qwen3VLPatchMerger",
    "Qwen3VLRotaryEmbedding",
    "Qwen3VLVisionEncoderOutput",
    "Qwen3VLVisionModel",
    "apply_rotary_pos_emb_vision",
    "rotate_half",
]


def rotate_half(x: Tensor) -> Tensor:
    """Swap and negate the two halves of the last dimension."""
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def apply_rotary_pos_emb_vision(q: Tensor, k: Tensor, cos: Tensor, sin: Tensor) -> tuple[Tensor, Tensor]:
    """Apply 2D rotary positional embeddings shared between query and key tensors."""
    q_dtype, k_dtype = q.dtype, k.dtype
    compute_dtype = q.dtype if q.dtype in (torch.float32, torch.float64) else torch.float32
    q = q.to(compute_dtype)
    k = k.to(compute_dtype)
    cos = cos.unsqueeze(-2).to(compute_dtype)
    sin = sin.unsqueeze(-2).to(compute_dtype)
    q_out = (q * cos) + (rotate_half(q) * sin)
    k_out = (k * cos) + (rotate_half(k) * sin)
    return q_out.to(q_dtype), k_out.to(k_dtype)


class Qwen3VLPatchEmbed(nn.Module):
    """Conv3d patch embedding shared between image and video inputs.

    The encoder consumes flat patch tensors of shape
    ``(N, in_channels * temporal_patch_size * patch_size * patch_size)``
    produced by the image processor; this module reshapes them into a
    ``(N, C, T, P, P)`` mini-batch and applies the official 3D conv.
    """

    def __init__(self, config: Qwen3VLVisionConfig) -> None:
        super().__init__()
        self.in_channels = config.in_channels
        self.patch_size = config.patch_size
        self.temporal_patch_size = config.temporal_patch_size
        self.hidden_size = config.hidden_size
        kernel = (self.temporal_patch_size, self.patch_size, self.patch_size)
        self.proj = nn.Conv3d(self.in_channels, self.hidden_size, kernel_size=kernel, stride=kernel, bias=True)

    def forward(self, hidden_states: Tensor) -> Tensor:
        if hidden_states.dim() != 2:
            raise ValueError(f"Expected flat patch tensor of shape (N, C*T*P*P); got {tuple(hidden_states.shape)}.")
        target_dtype = self.proj.weight.dtype
        x = hidden_states.view(-1, self.in_channels, self.temporal_patch_size, self.patch_size, self.patch_size)
        x = self.proj(x.to(target_dtype))
        return x.view(-1, self.hidden_size)


class Qwen3VLRotaryEmbedding(nn.Module):
    """Half-dim rotary frequency table.

    Stores ``inv_freq`` of length ``dim // 2`` (where ``dim`` is half of the
    attention head dimension) and returns the outer-product frequency table
    used to build separate row / column position embeddings for each token.
    """

    inv_freq: Tensor

    def __init__(self, dim: int, theta: float = 10000.0) -> None:
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"rotary dim must be even, got {dim}.")
        self.dim = dim
        self.theta = theta
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seqlen: int) -> Tensor:
        seq = torch.arange(seqlen, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        return torch.outer(seq, self.inv_freq)


class Qwen3VLAttention(nn.Module):
    """Multi-head self-attention over a packed-sequence input.

    Operates on the flat ``(seq_len, hidden_size)`` token stream, splitting
    into per-image segments via ``cu_seqlens`` so each image attends only to
    its own patches via ``F.scaled_dot_product_attention``.
    """

    def __init__(self, config: Qwen3VLVisionConfig) -> None:
        super().__init__()
        if config.hidden_size % config.num_heads != 0:
            raise ValueError(f"hidden_size ({config.hidden_size}) must be divisible by num_heads ({config.num_heads}).")
        self.dim = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = self.dim // self.num_heads
        self.qkv = nn.Linear(self.dim, self.dim * 3, bias=True)
        self.proj = nn.Linear(self.dim, self.dim)

    def forward(self, hidden_states: Tensor, cu_seqlens: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
        seq_len = hidden_states.shape[0]
        qkv = self.qkv(hidden_states).reshape(seq_len, 3, self.num_heads, self.head_dim).permute(1, 0, 2, 3)
        q, k, v = qkv.unbind(0)
        q, k = apply_rotary_pos_emb_vision(q, k, cos, sin)

        q = q.transpose(0, 1).unsqueeze(0)
        k = k.transpose(0, 1).unsqueeze(0)
        v = v.transpose(0, 1).unsqueeze(0)

        lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
        if len(lengths) == 1:
            attn = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
        else:
            q_splits = torch.split(q, lengths, dim=2)
            k_splits = torch.split(k, lengths, dim=2)
            v_splits = torch.split(v, lengths, dim=2)
            attn_chunks = [
                F.scaled_dot_product_attention(qi, ki, vi, dropout_p=0.0)
                for qi, ki, vi in zip(q_splits, k_splits, v_splits)
            ]
            attn = torch.cat(attn_chunks, dim=2)

        attn = attn.squeeze(0).transpose(0, 1).contiguous().reshape(seq_len, self.dim)
        return self.proj(attn)


class Qwen3VLMLP(nn.Module):
    """Feed-forward block: ``linear_fc1 -> activation -> linear_fc2``."""

    def __init__(self, config: Qwen3VLVisionConfig) -> None:
        super().__init__()
        self.linear_fc1 = nn.Linear(config.hidden_size, config.intermediate_size, bias=True)
        self.linear_fc2 = nn.Linear(config.intermediate_size, config.hidden_size, bias=True)
        if config.hidden_act == "gelu_pytorch_tanh":
            self.act_fn: nn.Module = nn.GELU(approximate="tanh")
        elif config.hidden_act == "gelu":
            self.act_fn = nn.GELU()
        else:
            raise ValueError(f"Unsupported hidden_act={config.hidden_act!r}.")

    def forward(self, x: Tensor) -> Tensor:
        return self.linear_fc2(self.act_fn(self.linear_fc1(x)))


class Qwen3VLBlock(nn.Module):
    """Single pre-norm transformer block of the Qwen3-VL vision tower."""

    def __init__(self, config: Qwen3VLVisionConfig) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.attn = Qwen3VLAttention(config)
        self.mlp = Qwen3VLMLP(config)

    def forward(self, hidden_states: Tensor, cu_seqlens: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
        hidden_states = hidden_states + self.attn(self.norm1(hidden_states), cu_seqlens, cos, sin)
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
        return hidden_states


class Qwen3VLPatchMerger(nn.Module):
    """Patch merger that flattens ``spatial_merge_size**2`` patches into a single token.

    With ``use_postshuffle_norm=False`` the layer norm is applied at the per-patch
    ``hidden_size`` (matching the main projector). With ``use_postshuffle_norm=True``
    the norm runs on the post-shuffle dimension ``hidden_size * spatial_merge_size**2``
    (matching the DeepStack mergers).
    """

    def __init__(self, config: Qwen3VLVisionConfig, use_postshuffle_norm: bool = False) -> None:
        super().__init__()
        self.merged_size = config.hidden_size * (config.spatial_merge_size**2)
        self.use_postshuffle_norm = use_postshuffle_norm
        norm_dim = self.merged_size if use_postshuffle_norm else config.hidden_size
        self.norm = nn.LayerNorm(norm_dim, eps=config.layer_norm_eps)
        self.linear_fc1 = nn.Linear(self.merged_size, self.merged_size, bias=True)
        self.act_fn = nn.GELU()
        self.linear_fc2 = nn.Linear(self.merged_size, config.out_hidden_size, bias=True)

    def forward(self, x: Tensor) -> Tensor:
        if self.use_postshuffle_norm:
            x = self.norm(x.view(-1, self.merged_size))
        else:
            x = self.norm(x).view(-1, self.merged_size)
        return self.linear_fc2(self.act_fn(self.linear_fc1(x)))


class Qwen3VLVisionEncoderOutput(NamedTuple):
    """Output of :class:`Qwen3VLVisionModel`.

    ``last_hidden_state`` and ``deepstack_features`` are post-merger and live
    in ``out_hidden_size`` space, ready to be injected into the LLM. Both are
    flat ``(N_merged, out_hidden_size)`` tensors where
    ``N_merged = sum(grid_t * grid_h * grid_w) // spatial_merge_size**2``.
    """

    last_hidden_state: Tensor
    deepstack_features: tuple[Tensor, ...]
    grid_thw: Tensor


class Qwen3VLVisionModel(nn.Module):
    """Qwen3-VL vision tower with DeepStack feature fusion.

    Consumes flat patch tensors and a ``grid_thw`` descriptor produced by
    :class:`Qwen3VLImageProcessor`. The state-dict layout matches the
    ``model.visual.*`` keys of the official HuggingFace checkpoints, so
    weights loaded via :mod:`._hf_weights` slot in directly.
    """

    def __init__(self, config: Qwen3VLVisionConfig) -> None:
        super().__init__()
        self.config = config
        self.patch_size = config.patch_size
        self.spatial_merge_size = config.spatial_merge_size

        self.patch_embed = Qwen3VLPatchEmbed(config)
        self.pos_embed = nn.Embedding(config.num_position_embeddings, config.hidden_size)
        self.num_grid_per_side = round(config.num_position_embeddings**0.5)
        if self.num_grid_per_side * self.num_grid_per_side != config.num_position_embeddings:
            raise ValueError(f"num_position_embeddings={config.num_position_embeddings} must be a perfect square.")

        head_dim = config.hidden_size // config.num_heads
        self.rotary_pos_emb = Qwen3VLRotaryEmbedding(head_dim // 2, theta=config.rope_theta)

        self.blocks = nn.ModuleList([Qwen3VLBlock(config) for _ in range(config.depth)])
        self.merger = Qwen3VLPatchMerger(config, use_postshuffle_norm=False)

        for idx in config.deepstack_visual_indexes:
            if idx < 0 or idx >= config.depth:
                raise ValueError(f"deepstack_visual_indexes contain {idx}, out of range for depth={config.depth}.")
        self.deepstack_visual_indexes = tuple(config.deepstack_visual_indexes)
        self.deepstack_merger_list = nn.ModuleList(
            [Qwen3VLPatchMerger(config, use_postshuffle_norm=True) for _ in self.deepstack_visual_indexes]
        )

    @classmethod
    def from_size(cls, size: str) -> Qwen3VLVisionModel:
        vision_config = Qwen3VLConfig.from_size(size).vision_config or Qwen3VLVisionConfig()
        return cls(vision_config)

    def fast_pos_embed_interpolate(self, grid_thw: Tensor) -> Tensor:
        device = self.pos_embed.weight.device
        weight_dtype = self.pos_embed.weight.dtype
        merge_size = self.spatial_merge_size
        side = self.num_grid_per_side

        chunks: list[Tensor] = []
        for t, h, w in grid_thw.tolist():
            h_idxs = torch.linspace(0, side - 1, h, dtype=torch.float32, device=device)
            w_idxs = torch.linspace(0, side - 1, w, dtype=torch.float32, device=device)
            h_floor = h_idxs.to(torch.long)
            w_floor = w_idxs.to(torch.long)
            h_ceil = (h_floor + 1).clamp(max=side - 1)
            w_ceil = (w_floor + 1).clamp(max=side - 1)
            dh = h_idxs - h_floor.to(torch.float32)
            dw = w_idxs - w_floor.to(torch.float32)

            base_h = h_floor * side
            base_h_ceil = h_ceil * side

            indices = (
                (base_h.unsqueeze(1) + w_floor.unsqueeze(0)).reshape(-1),
                (base_h.unsqueeze(1) + w_ceil.unsqueeze(0)).reshape(-1),
                (base_h_ceil.unsqueeze(1) + w_floor.unsqueeze(0)).reshape(-1),
                (base_h_ceil.unsqueeze(1) + w_ceil.unsqueeze(0)).reshape(-1),
            )
            weights = (
                ((1 - dh).unsqueeze(1) * (1 - dw).unsqueeze(0)).reshape(-1).to(weight_dtype),
                ((1 - dh).unsqueeze(1) * dw.unsqueeze(0)).reshape(-1).to(weight_dtype),
                (dh.unsqueeze(1) * (1 - dw).unsqueeze(0)).reshape(-1).to(weight_dtype),
                (dh.unsqueeze(1) * dw.unsqueeze(0)).reshape(-1).to(weight_dtype),
            )
            embed = self.pos_embed(indices[0]) * weights[0].unsqueeze(-1)
            for idx, w_ij in zip(indices[1:], weights[1:]):
                embed = embed + self.pos_embed(idx) * w_ij.unsqueeze(-1)

            embed = embed.repeat(t, 1)
            embed = (
                embed.view(t, h // merge_size, merge_size, w // merge_size, merge_size, -1)
                .permute(0, 1, 3, 2, 4, 5)
                .flatten(0, 4)
            )
            chunks.append(embed)
        return torch.cat(chunks, dim=0)

    def rot_pos_emb(self, grid_thw: Tensor) -> Tensor:
        merge_size = self.spatial_merge_size
        max_hw = int(grid_thw[:, 1:].max().item())
        freq_table = self.rotary_pos_emb(max_hw)
        device = freq_table.device

        total_tokens = int(torch.prod(grid_thw, dim=1).sum().item())
        pos_ids = torch.empty((total_tokens, 2), dtype=torch.long, device=device)

        offset = 0
        for t, h, w in grid_thw.tolist():
            merged_h, merged_w = h // merge_size, w // merge_size
            block_rows = torch.arange(merged_h, device=device)
            block_cols = torch.arange(merged_w, device=device)
            intra_row = torch.arange(merge_size, device=device)
            intra_col = torch.arange(merge_size, device=device)
            row_idx = block_rows[:, None, None, None] * merge_size + intra_row[None, None, :, None]
            col_idx = block_cols[None, :, None, None] * merge_size + intra_col[None, None, None, :]
            row_idx = row_idx.expand(merged_h, merged_w, merge_size, merge_size).reshape(-1)
            col_idx = col_idx.expand(merged_h, merged_w, merge_size, merge_size).reshape(-1)
            coords = torch.stack((row_idx, col_idx), dim=-1)
            if t > 1:
                coords = coords.repeat(t, 1)
            n = coords.shape[0]
            pos_ids[offset : offset + n] = coords
            offset += n

        embeddings = freq_table[pos_ids].flatten(1)
        return embeddings

    def forward(self, hidden_states: Tensor, grid_thw: Tensor) -> Qwen3VLVisionEncoderOutput:
        if grid_thw.dim() != 2 or grid_thw.shape[1] != 3:
            raise ValueError(f"grid_thw must have shape (B, 3); got {tuple(grid_thw.shape)}.")
        if grid_thw.dtype not in (torch.int32, torch.int64):
            raise ValueError(f"grid_thw must be an integer tensor; got dtype {grid_thw.dtype}.")

        x = self.patch_embed(hidden_states)
        pos = self.fast_pos_embed_interpolate(grid_thw).to(x.dtype)
        x = x + pos

        rot = self.rot_pos_emb(grid_thw)
        emb = torch.cat((rot, rot), dim=-1)
        cos, sin = emb.cos(), emb.sin()

        token_counts = grid_thw[:, 0] * grid_thw[:, 1] * grid_thw[:, 2]
        cu_seqlens = F.pad(torch.cumsum(token_counts, dim=0, dtype=torch.int32), (1, 0))

        deepstack_set = set(self.deepstack_visual_indexes)
        deepstack_outputs: list[Tensor] = []
        for i, blk in enumerate(self.blocks):
            x = blk(x, cu_seqlens, cos, sin)
            if i in deepstack_set:
                merger_idx = self.deepstack_visual_indexes.index(i)
                deepstack_outputs.append(self.deepstack_merger_list[merger_idx](x))

        merged = self.merger(x)
        return Qwen3VLVisionEncoderOutput(
            last_hidden_state=merged,
            deepstack_features=tuple(deepstack_outputs),
            grid_thw=grid_thw,
        )
