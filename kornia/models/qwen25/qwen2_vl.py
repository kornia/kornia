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

import warnings
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn import Module

from kornia.core.check import KORNIA_CHECK

from .weights_loader import Qwen25WeightLoader


class Qwen2VLPatchMerger(Module):
    """Merge image patches using 3D convolution for video/temporal support.

    This implementation uses Conv3d for the reference architecture,
    which supports both images and video frames by processing the temporal dimension.

    Args:
        dim: Output embedding dimension.
        context_window: Context window size.
        spatial_merge_size: Spatial merge size (patch size = 2 for temporal).
    """

    def __init__(
        self,
        dim: int,
        context_window: int = 224,
        spatial_merge_size: int = 2,
        in_channels: int = 3,
    ) -> None:
        super().__init__()
        self.spatial_merge_size = spatial_merge_size

        # Use Conv3d to handle temporal dimension
        # kernel_size: (temporal=2, height=14, width=14)
        self.conv = nn.Conv3d(
            in_channels=in_channels,
            out_channels=dim,
            kernel_size=(spatial_merge_size, 14, 14),
            stride=(spatial_merge_size, 14, 14),
            bias=False,  # No bias in this implementation
        )
        # Note: No LayerNorm in patch_embed for this architecture

    def forward(self, x: Tensor) -> Tensor:
        # Input: (B, C, H, W) for images
        # For Conv3d with kernel (2, 14, 14), need at least T=2
        if x.dim() == 4:
            # Pad temporal dimension by duplicating the frame
            # (B, C, H, W) -> (B, C, 2, H, W)
            x = x.unsqueeze(2).repeat(1, 1, self.spatial_merge_size, 1, 1)

        # Conv3d: (B, C, T, H, W) -> (B, dim, T', H', W')
        x = self.conv(x)

        # Flatten spatial and temporal: (B, dim, T', H', W') -> (B, dim, T'*H'*W')
        x = x.flatten(2)

        # Transpose for output: (B, seq_len, dim)
        x = x.transpose(1, 2)
        return x


class Qwen2VLMerger(Module):
    """Final merger layer with spatial compression and MLP.

    This implementation:
    1. Compresses visual tokens spatially (2x2 patch grouping)
    2. Applies LayerNorm
    3. Projects through 2-layer MLP

    The spatial compression groups adjacent 2x2 patches:
    (B, H*W, 1280) → (B, H*W/4, 1280*4) = (B, H*W/4, 5120)

    Args:
        embed_dim: Input embedding dimension (default: 1280).
        hidden_dim: Hidden layer dimension after compression (default: 5120).
        out_dim: Output dimension (default: 2048).
        spatial_merge_size: Size of spatial grouping (default: 2 for 2x2).
    """

    def __init__(
        self,
        embed_dim: int = 1280,
        hidden_dim: int = 5120,
        out_dim: int = 2048,
        spatial_merge_size: int = 2,
    ) -> None:
        super().__init__()

        self.embed_dim = embed_dim
        self.spatial_merge_size = spatial_merge_size

        # After spatial merging, dim becomes embed_dim * (spatial_merge_size^2)
        # For 2x2 merging: 1280 * 4 = 5120
        merged_dim = embed_dim * (spatial_merge_size**2)

        # LayerNorm operates on original embed_dim (before spatial merge)
        self.ln_q = nn.LayerNorm(embed_dim, eps=1e-6, bias=False)

        # 2-layer MLP operates on merged dimension
        # Note: HF uses indices 0 and 2 (mlp.0 and mlp.2) in state_dict
        self.mlp = nn.Sequential(
            nn.Linear(merged_dim, hidden_dim),  # mlp.0: 5120 → 5120
            nn.GELU(),  # mlp.1 (implicit)
            nn.Linear(hidden_dim, out_dim),  # mlp.2: 5120 → 2048
        )

    def forward(self, x: Tensor, grid_h: int, grid_w: int) -> Tensor:
        """Forward pass with spatial compression.

        Args:
            x: Input tensor of shape (B, seq_len, embed_dim)
               where seq_len = grid_h * grid_w from vision encoder
            grid_h: Height of the spatial grid
            grid_w: Width of the spatial grid

        Returns:
            Output tensor of shape (B, seq_len/4, out_dim)
        """
        # Apply LayerNorm before spatial merging
        x = self.ln_q(x)

        # Spatial compression: group 2x2 patches
        # Input: (B, H*W, C) where L=H*W is a perfect square
        B, L, C = x.shape

        # Reshape to spatial grid: (B, L, C) -> (B, H, W, C)
        x = x.reshape(B, grid_h, grid_w, C)

        # Group into 2x2 patches for spatial merging
        merge_size = self.spatial_merge_size  # =2
        H_merged = grid_h // merge_size
        W_merged = grid_w // merge_size

        # Reshape to separate the merge patches:
        # (B, H, W, C) -> ( B, H/2, 2, W/2, 2, C)
        x = x.reshape(B, H_merged, merge_size, W_merged, merge_size, C)

        # Rearrange dimensions to group the 2x2 patches together:
        # (B, H/2, 2, W/2, 2, C) -> (B, H/2, W/2, 2, 2, C)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()

        # Flatten the 2x2 patches: (B, H/2, W/2, 2*2*C)
        new_seq_len = H_merged * W_merged
        merged_dim = C * (merge_size**2)  # 1280 * 4 = 5120
        x = x.reshape(B, new_seq_len, merged_dim)

        # Apply MLP: (B, seq_len/4, 5120) → (B, seq_len/4, 2048)
        x = self.mlp(x)

        return x


class Qwen2VLVisionAttention(Module):
    """Multi-head self-attention module used in the Qwen2-VL vision encoder.

    Args:
        dim: Input feature dimension.
        num_heads: Number of attention heads.
    """

    def __init__(self, dim: int, num_heads: int = 16) -> None:
        super().__init__()

        # Validate that dimension is divisible by number of heads
        KORNIA_CHECK(dim % num_heads == 0, f"dim ({dim}) must be divisible by num_heads ({num_heads})")

        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim, bias=True)

    def forward(self, x: Tensor, cu_seqlens: Optional[Tensor] = None) -> Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        x = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x


class Qwen2VLMLP(Module):
    """Gated FeedForward MLP (SwiGLU) used in Qwen2-VL vision transformer blocks.

    This implements the gated MLP (SwiGLU) architecture:
    - gate_proj: Projects to hidden dimension, applies SiLU activation
    - up_proj: Projects to hidden dimension (no activation)
    - down_proj: Projects back to original dimension

    Output = down_proj(silu(gate_proj(x)) * up_proj(x))

    Args:
        dim: Input and output feature dimension.
        hidden_dim: Dimension of the hidden layer. If None, defaults to 4 * dim.
    """

    def __init__(self, dim: int, hidden_dim: Optional[int] = None) -> None:
        super().__init__()
        if hidden_dim is None:
            hidden_dim = 4 * dim

        # Gated MLP components
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=True)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=True)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=True)
        self.act = nn.SiLU()  # SwiGLU uses SiLU (Swish) activation

    def forward(self, x: Tensor) -> Tensor:
        # SwiGLU: down_proj(silu(gate_proj(x)) * up_proj(x))
        return self.down_proj(self.act(self.gate_proj(x)) * self.up_proj(x))


class Qwen2VLVisionBlock(Module):
    """Single transformer block for the Qwen2-VL vision encoder.

    Applies layer-normalized self-attention followed by an MLP, each with
    residual connections.

    Args:
        dim: Token embedding dimension.
        num_heads: Number of attention heads.
        intermediate_size: Dimension of the MLP hidden layer.
    """

    def __init__(self, dim: int, num_heads: int, intermediate_size: int = 3420) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6, bias=False)
        self.attn = Qwen2VLVisionAttention(dim, num_heads=num_heads)

        self.norm2 = nn.LayerNorm(dim, eps=1e-6, bias=False)
        self.mlp = Qwen2VLMLP(dim, intermediate_size)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class Qwen2VLVisionTransformer(Module):
    """PyTorch implementation of the Qwen2-VL vision encoder.

    A ViT-style backbone composed of a patch merger followed by stacked
    transformer blocks with rotary positional embeddings.

    Args:
        embed_dim: Embedding dimension.
        depth: Number of transformer blocks.
        num_heads: Number of attention heads.
        intermediate_size: Dimension of the MLP hidden layer.
        patch_size: Patch size.
        in_channels: Input channel count.
    """

    def __init__(
        self,
        embed_dim: int = 1280,
        depth: int = 32,
        num_heads: int = 16,
        intermediate_size: int = 3420,  # HF uses 3420, not 4*1280=5120
        in_channels: int = 3,
        out_hidden_size: int = 2048,
    ) -> None:
        super().__init__()
        self.patch_embed = Qwen2VLPatchMerger(embed_dim, context_window=224, spatial_merge_size=2)
        self.blocks = nn.ModuleList([Qwen2VLVisionBlock(embed_dim, num_heads, intermediate_size) for _ in range(depth)])

        # Final merger projection: embed_dim -> out_hidden_size
        # Uses 2-layer MLP with LayerNorm
        self.merger = Qwen2VLMerger(embed_dim=embed_dim, out_dim=out_hidden_size)

    @classmethod
    def from_pretrained(
        cls,
        model_id: str = "Qwen/Qwen2.5-VL-3B-Instruct",
        embed_dim: int = 1280,
        depth: int = 32,
        num_heads: int = 16,
        intermediate_size: int = 3420,
        out_hidden_size: int = 2048,
    ) -> Qwen2VLVisionTransformer:
        """Load pretrained Vision Transformer.

        Args:
            model_id: Pretrained model identifier.
            embed_dim: Embedding dimension (must match pretrained model).
            depth: Number of transformer blocks (must match pretrained model).
            num_heads: Number of attention heads (must match pretrained model).
            intermediate_size: MLP hidden dimension (must match pretrained model).
            out_hidden_size: Output hidden dimension of the final merger projection (must match pretrained model).

        Returns:
            Qwen2VLVisionTransformer with loaded pretrained weights.

        Example:
            >>> model = Qwen2VLVisionTransformer.from_pretrained()
            >>> image = torch.randn(1, 3, 448, 448)
            >>> output = model(image)

        Note:
            This method requires the transformers library.
            Install it with: pip install transformers
        """
        # Detect device (use GPU if available to reduce RAM usage)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Create model instance
        model = cls(
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            intermediate_size=intermediate_size,
            out_hidden_size=out_hidden_size,
        )
        model = model.to(device)

        # Load weights
        loader = Qwen25WeightLoader(model_id)
        state_dict = loader.load_weights("vision_encoder")

        # Load into model (strict=False to allow missing rotary embedding keys)
        result = model.load_state_dict(state_dict, strict=False)

        # Log any missing or unexpected keys (these indicate potential issues)
        if result.missing_keys:
            warnings.warn(f"Missing {len(result.missing_keys)} keys when loading pretrained weights")
        if result.unexpected_keys:
            warnings.warn(f"Unexpected {len(result.unexpected_keys)} keys in pretrained weights")

        return model

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through the vision transformer.

        Args:
            x: Input image tensor of shape (B, C, H, W)

        Returns:
            Visual features of shape (B, num_tokens, out_hidden_size)

        Note:
            Input dimensions are automatically padded to be divisible by 28.
            Supports any aspect ratio (square, 16:9, 9:16, 4:3, etc.)
        """
        B, C, H, W = x.shape

        # Pad to nearest multiple of 28 (14*2) to ensure grid is even
        # The merger requires grid_h and grid_w to be divisible by 2 for 2x2 grouping
        pad_h = (28 - H % 28) % 28
        pad_w = (28 - W % 28) % 28

        # Always pad (even if 0) to avoid dynamic control flow for ONNX export
        x = F.pad(x, (0, pad_w, 0, pad_h))
        H = H + pad_h
        W = W + pad_w

        # Calculate grid dimensions after padding
        grid_h = H // 14
        grid_w = W // 14

        # Patch embedding: (B, C, H, W) -> (B, L, embed_dim)
        x = self.patch_embed(x)

        # Transformer blocks
        for block in self.blocks:
            x = block(x)

        # Apply final projection (merger) with grid dimensions
        x = self.merger(x, grid_h, grid_w)

        return x
