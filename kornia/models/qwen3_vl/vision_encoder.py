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

from typing import NamedTuple, Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .config import Qwen3VLVisionConfig

__all__ = [
    "Qwen3VLAttention",
    "Qwen3VLEncoder",
    "Qwen3VLLayer",
    "Qwen3VLMLP",
    "Qwen3VLPatchEmbed",
    "Qwen3VLRotaryEmbedding",
    "Qwen3VLVisionEncoderOutput",
    "Qwen3VLVisionTransformer",
    "apply_rotary_pos_emb",
]


def apply_rotary_pos_emb(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """Apply rotary positional embeddings to a query or key tensor.

    Args:
        x: Tensor of shape ``(..., seq_len, head_dim)``.
        cos: Cosine table broadcastable against ``x``.
        sin: Sine table broadcastable against ``x``.

    Returns:
        Tensor of the same shape as ``x`` with RoPE applied along the last dim.
    """
    x1, x2 = x.chunk(2, dim=-1)
    x_rotated = torch.cat((-x2, x1), dim=-1)
    return (x * cos) + (x_rotated * sin)


class Qwen3VLPatchEmbed(nn.Module):
    """Spatial patch embedding for the Qwen3-VL vision tower.

    Splits an input image into non-overlapping patches via a strided 2D convolution
    and projects each patch into the model hidden dimension. Video / temporal
    handling is added in a follow-up PR; this module currently expects ``BCHW``.

    Args:
        config: Vision encoder configuration providing ``in_channels``,
            ``patch_size`` and ``hidden_size``.
    """

    def __init__(self, config: Qwen3VLVisionConfig) -> None:
        super().__init__()
        self.patch_size = config.patch_size
        self.hidden_size = config.hidden_size
        self.proj = nn.Conv2d(
            config.in_channels,
            config.hidden_size,
            kernel_size=config.patch_size,
            stride=config.patch_size,
            bias=False,
        )

    def forward(self, pixel_values: Tensor) -> tuple[Tensor, int, int]:
        """Embed patches and return the token sequence with the spatial grid size.

        Args:
            pixel_values: Tensor of shape ``(B, C, H, W)``.

        Returns:
            Tuple ``(tokens, h_patches, w_patches)`` where ``tokens`` has shape
            ``(B, h_patches * w_patches, hidden_size)``.
        """
        if pixel_values.dim() != 4:
            raise ValueError(f"Expected 4D input (B, C, H, W); got shape {tuple(pixel_values.shape)}.")
        if pixel_values.shape[-1] % self.patch_size != 0 or pixel_values.shape[-2] % self.patch_size != 0:
            raise ValueError(
                f"Input spatial size {tuple(pixel_values.shape[-2:])} must be divisible by patch_size "
                f"{self.patch_size}."
            )
        x = self.proj(pixel_values)
        h_patches, w_patches = x.shape[-2], x.shape[-1]
        x = x.flatten(2).transpose(1, 2)
        return x, h_patches, w_patches


class Qwen3VLRotaryEmbedding(nn.Module):
    """Two-dimensional rotary positional embedding.

    The head dimension is split in half: the first half rotates with the row
    coordinate, the second half with the column coordinate. This mirrors the
    convention used by Qwen2-VL and Kimi-VL.

    Args:
        head_dim: Per-head feature dimension.
        theta: Base frequency for the rotary embedding.
    """

    def __init__(self, head_dim: int, theta: float = 10000.0) -> None:
        super().__init__()
        if head_dim % 4 != 0:
            raise ValueError(f"head_dim must be divisible by 4 for 2D RoPE, got {head_dim}.")
        self.head_dim = head_dim
        self.theta = theta

    def forward(self, h: int, w: int, device: torch.device, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
        """Build cosine/sine tables for the (h, w) patch grid.

        Args:
            h: Number of patches along height.
            w: Number of patches along width.
            device: Device on which to allocate the tables.
            dtype: Dtype of the returned tables.

        Returns:
            ``(cos, sin)`` each of shape ``(h * w, head_dim)``.
        """
        half = self.head_dim // 2
        inv_freq = 1.0 / (self.theta ** (torch.arange(0, half, 2, device=device, dtype=torch.float32) / half))

        seq_h = torch.arange(h, device=device, dtype=torch.float32)
        seq_w = torch.arange(w, device=device, dtype=torch.float32)

        freqs_h = torch.outer(seq_h, inv_freq)
        freqs_w = torch.outer(seq_w, inv_freq)

        freqs_h = freqs_h.repeat_interleave(w, dim=0)
        freqs_w = freqs_w.repeat(h, 1)

        emb_h = torch.cat((freqs_h, freqs_h), dim=-1)
        emb_w = torch.cat((freqs_w, freqs_w), dim=-1)
        emb = torch.cat((emb_h, emb_w), dim=-1)

        return emb.cos().to(dtype), emb.sin().to(dtype)


class Qwen3VLAttention(nn.Module):
    """Multi-head self-attention block with 2D rotary positional embeddings.

    Args:
        config: Vision encoder configuration providing ``hidden_size`` and
            ``num_attention_heads``.
    """

    def __init__(self, config: Qwen3VLVisionConfig) -> None:
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                f"hidden_size ({config.hidden_size}) must be divisible by "
                f"num_attention_heads ({config.num_attention_heads})."
            )
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.hidden_size = config.hidden_size

        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size)

    def forward(
        self,
        hidden_states: Tensor,
        cos: Tensor,
        sin: Tensor,
        attention_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Apply attention with rotary embeddings.

        Args:
            hidden_states: ``(B, N, hidden_size)``.
            cos: Rotary cosine table ``(N, head_dim)``.
            sin: Rotary sine table ``(N, head_dim)``.
            attention_mask: Optional broadcastable attention mask.

        Returns:
            Tensor of shape ``(B, N, hidden_size)``.
        """
        batch_size, seq_len, _ = hidden_states.shape

        q = self.q_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        cos = cos.view(1, 1, seq_len, self.head_dim)
        sin = sin.view(1, 1, seq_len, self.head_dim)

        q = apply_rotary_pos_emb(q, cos, sin)
        k = apply_rotary_pos_emb(k, cos, sin)

        attn = F.scaled_dot_product_attention(q, k, v, attn_mask=attention_mask, dropout_p=0.0)
        attn = attn.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        return self.out_proj(attn)


class Qwen3VLMLP(nn.Module):
    """Feed-forward block used inside each Qwen3-VL vision layer.

    Args:
        config: Vision encoder configuration providing ``hidden_size`` and
            ``intermediate_size``.
    """

    def __init__(self, config: Qwen3VLVisionConfig) -> None:
        super().__init__()
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, x: Tensor) -> Tensor:
        return self.fc2(self.act(self.fc1(x)))


class Qwen3VLLayer(nn.Module):
    """A single pre-norm transformer layer of the Qwen3-VL vision tower.

    Applies layer normalization before self-attention and again before the MLP,
    with residual connections around each sublayer.

    Args:
        config: Vision encoder configuration.
    """

    def __init__(self, config: Qwen3VLVisionConfig) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.attn = Qwen3VLAttention(config)
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = Qwen3VLMLP(config)

    def forward(
        self,
        x: Tensor,
        cos: Tensor,
        sin: Tensor,
        attention_mask: Optional[Tensor] = None,
    ) -> Tensor:
        x = x + self.attn(self.norm1(x), cos, sin, attention_mask)
        x = x + self.mlp(self.norm2(x))
        return x


class Qwen3VLEncoder(nn.Module):
    """Stack of :class:`Qwen3VLLayer` blocks with optional DeepStack feature collection.

    When ``deepstack_layer_indices`` is non-empty the encoder additionally returns
    the hidden state immediately after each indexed layer, in the order of the
    indices supplied. These intermediate features feed the projector's DeepStack
    fusion in subsequent PRs.

    Args:
        config: Vision encoder configuration.
        deepstack_layer_indices: Layer indices whose outputs should be exposed
            as DeepStack features.
    """

    def __init__(self, config: Qwen3VLVisionConfig, deepstack_layer_indices: tuple[int, ...]) -> None:
        super().__init__()
        for idx in deepstack_layer_indices:
            if idx < 0 or idx >= config.num_hidden_layers:
                raise ValueError(
                    f"deepstack_layer_indices contain {idx}, which is out of range for "
                    f"num_hidden_layers={config.num_hidden_layers}."
                )
        self.deepstack_layer_indices = tuple(deepstack_layer_indices)
        self.layers = nn.ModuleList([Qwen3VLLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(
        self,
        x: Tensor,
        cos: Tensor,
        sin: Tensor,
        attention_mask: Optional[Tensor] = None,
    ) -> tuple[Tensor, tuple[Tensor, ...]]:
        """Run the layer stack and collect DeepStack features.

        Args:
            x: ``(B, N, hidden_size)``.
            cos: Rotary cosine table ``(N, head_dim)``.
            sin: Rotary sine table ``(N, head_dim)``.
            attention_mask: Optional broadcastable mask.

        Returns:
            Tuple ``(last_hidden_state, deepstack_features)`` where the second
            element is a tuple of intermediate hidden states.
        """
        index_set = set(self.deepstack_layer_indices)
        deepstack: list[Tensor] = []
        for idx, layer in enumerate(self.layers):
            x = layer(x, cos, sin, attention_mask)
            if idx in index_set:
                deepstack.append(x)
        # Preserve caller-specified order, even if a layer index is requested twice.
        ordered = tuple(deepstack[self.deepstack_layer_indices.index(i)] for i in self.deepstack_layer_indices)
        return x, ordered


class Qwen3VLVisionEncoderOutput(NamedTuple):
    """Container returned by :class:`Qwen3VLVisionTransformer`.

    Attributes:
        last_hidden_state: Final encoded tokens of shape ``(B, N, hidden_size)``.
        deepstack_features: Tuple of intermediate hidden states collected at
            the layer indices configured for DeepStack fusion.
        grid_hw: ``(h_patches, w_patches)`` describing the spatial grid that
            produced the token sequence; downstream modules (projector, video
            handler) need this to reshape tokens back into 2D.
    """

    last_hidden_state: Tensor
    deepstack_features: tuple[Tensor, ...]
    grid_hw: tuple[int, int]


def _default_deepstack_indices(num_hidden_layers: int) -> tuple[int, ...]:
    """Return the default DeepStack layer indices for a tower with ``num_hidden_layers`` blocks."""
    if num_hidden_layers < 3:
        return (num_hidden_layers - 1,)
    return (num_hidden_layers // 3, (2 * num_hidden_layers) // 3, num_hidden_layers - 1)


class Qwen3VLVisionTransformer(nn.Module):
    """Qwen3-VL vision tower with DeepStack feature fusion.

    The encoder is a pre-norm ViT with 2D rotary positional embeddings and no
    absolute position parameters, enabling dynamic-resolution inputs once the
    Qwen3-VL preprocessor lands. Beyond returning the final hidden state the
    forward pass also exposes intermediate features from a configurable set of
    transformer layers, matching the Qwen3-VL DeepStack fusion strategy.

    Args:
        config: Vision encoder configuration. The default ``deepstack_layer_indices``
            schedule selects three layers spaced roughly evenly through the tower.
    """

    def __init__(self, config: Qwen3VLVisionConfig) -> None:
        super().__init__()
        self.config = config
        self.patch_embed = Qwen3VLPatchEmbed(config)
        head_dim = config.hidden_size // config.num_attention_heads
        self.rope = Qwen3VLRotaryEmbedding(head_dim, theta=config.rope_theta)

        if config.deepstack_layer_indices is None:
            deepstack_indices = _default_deepstack_indices(config.num_hidden_layers)
        else:
            deepstack_indices = config.deepstack_layer_indices
        self.encoder = Qwen3VLEncoder(config, deepstack_indices)
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    @property
    def deepstack_layer_indices(self) -> tuple[int, ...]:
        """Layer indices whose outputs are returned as DeepStack features."""
        return self.encoder.deepstack_layer_indices

    def forward(
        self,
        pixel_values: Tensor,
        attention_mask: Optional[Tensor] = None,
    ) -> Qwen3VLVisionEncoderOutput:
        """Encode a batch of images.

        Args:
            pixel_values: ``(B, C, H, W)``. Spatial dims must be divisible by
                ``config.patch_size``.
            attention_mask: Optional broadcastable mask, e.g. ``(B, 1, N, N)``.

        Returns:
            :class:`Qwen3VLVisionEncoderOutput` containing the final hidden
            state, the DeepStack feature tuple, and the spatial grid size.
        """
        x, h_patches, w_patches = self.patch_embed(pixel_values)
        cos, sin = self.rope(h_patches, w_patches, x.device, x.dtype)
        last, deepstack = self.encoder(x, cos, sin, attention_mask=attention_mask)
        last = self.norm(last)
        return Qwen3VLVisionEncoderOutput(
            last_hidden_state=last,
            deepstack_features=deepstack,
            grid_hw=(h_patches, w_patches),
        )
