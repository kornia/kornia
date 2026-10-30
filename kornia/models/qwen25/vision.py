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

"""Qwen2.5-VL vision tower.

A port of ``Qwen2_5_VisionTransformerPretrainedModel`` from Hugging Face ``transformers``
(``models/qwen2_5_vl/modeling_qwen2_5_vl.py``, v4.51.0), taking a batch of images instead of the
flattened patches and ``grid_thw`` its image processor produces.
"""

from __future__ import annotations

import json
import os
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from kornia.core.safetensors import load_safetensors

from .config import Qwen25VisionConfig

__all__ = ["Qwen25VisionTransformer"]

_WEIGHTS_PREFIX = "visual."


def _compute_dtype(dtype: torch.dtype) -> torch.dtype:
    """Return the dtype the normalizations and rotary embedding run in: float32, or float64 for float64 inputs."""
    return torch.float64 if dtype == torch.float64 else torch.float32


def _merge_order(grid: torch.Tensor, merge_size: int) -> torch.Tensor:
    """Flatten a ``(gh, gw)`` patch grid so each ``merge_size`` x ``merge_size`` square is contiguous.

    This is the token order of the reference image processor: merged cells in raster order, and the
    patches of one cell in raster order within it.
    """
    gh, gw = grid.shape
    return grid.reshape(gh // merge_size, merge_size, gw // merge_size, merge_size).permute(0, 2, 1, 3).flatten()


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


class Qwen25RMSNorm(nn.Module):
    """Root mean square normalization, computed in at least float32.

    Args:
        dim: Size of the normalized last dimension.
        eps: Value added to the mean square for numerical stability.
    """

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize ``x`` over its last dimension.

        Args:
            x: Tensor with shape ``(*, D)``.

        Returns:
            Tensor with the shape and dtype of ``x``.
        """
        h = x.to(_compute_dtype(x.dtype))
        h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * h.to(x.dtype)


class Qwen25VisionPatchEmbed(nn.Module):
    """Patch embedding with a temporal kernel, applied to a still image repeated over time.

    Args:
        config: Vision tower configuration.
    """

    def __init__(self, config: Qwen25VisionConfig) -> None:
        super().__init__()
        self.temporal_patch_size = config.temporal_patch_size
        self.merge_size = config.spatial_merge_size
        kernel = (config.temporal_patch_size, config.patch_size, config.patch_size)
        self.proj = nn.Conv3d(config.in_channels, config.hidden_size, kernel_size=kernel, stride=kernel, bias=False)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Embed every patch of ``images``.

        Args:
            images: Image tensor with shape ``(B, C, H, W)``.

        Returns:
            Patch tokens with shape ``(B, N, D)`` in merge order, where :math:`N` is the number of patches.
        """
        frames = images.unsqueeze(2).expand(-1, -1, self.temporal_patch_size, -1, -1)
        x = self.proj(frames).squeeze(2)
        b, d, gh, gw = x.shape
        m = self.merge_size
        x = x.reshape(b, d, gh // m, m, gw // m, m).permute(0, 2, 4, 3, 5, 1)
        return x.reshape(b, gh * gw, d)


class Qwen25VisionAttention(nn.Module):
    """Multi-head self-attention with a 2D rotary embedding on queries and keys.

    Args:
        dim: Token dimension.
        num_heads: Number of attention heads.
    """

    def __init__(self, dim: int, num_heads: int) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim, bias=True)

    def forward(
        self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, attn_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Attend over the tokens of each image.

        Args:
            x: Tokens with shape ``(B, N, D)``.
            cos: Rotary cosine table with shape ``(N, D / H)``.
            sin: Rotary sine table with shape ``(N, D / H)``.
            attn_mask: Optional boolean mask with shape ``(N, N)``, ``True`` where attention is allowed.

        Returns:
            Tokens with shape ``(B, N, D)``.
        """
        b, n, d = x.shape
        q, k, v = self.qkv(x).reshape(b, n, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4).unbind(0)
        compute_dtype = _compute_dtype(x.dtype)
        qf, kf = q.to(compute_dtype), k.to(compute_dtype)
        q = (qf * cos + _rotate_half(qf) * sin).to(x.dtype)
        k = (kf * cos + _rotate_half(kf) * sin).to(x.dtype)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        return self.proj(out.transpose(1, 2).reshape(b, n, d))


class Qwen25VisionMLP(nn.Module):
    """Gated feed-forward network, ``down_proj(silu(gate_proj(x)) * up_proj(x))``.

    Args:
        dim: Token dimension.
        hidden_dim: Hidden dimension.
    """

    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=True)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=True)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the gated MLP.

        Args:
            x: Tokens with shape ``(B, N, D)``.

        Returns:
            Tokens with shape ``(B, N, D)``.
        """
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Qwen25VisionBlock(nn.Module):
    """Pre-norm transformer block of the Qwen2.5-VL vision tower.

    Args:
        config: Vision tower configuration.
    """

    def __init__(self, config: Qwen25VisionConfig) -> None:
        super().__init__()
        self.norm1 = Qwen25RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.norm2 = Qwen25RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attn = Qwen25VisionAttention(config.hidden_size, config.num_heads)
        self.mlp = Qwen25VisionMLP(config.hidden_size, config.intermediate_size)

    def forward(
        self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, attn_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Run one block.

        Args:
            x: Tokens with shape ``(B, N, D)``.
            cos: Rotary cosine table with shape ``(N, D / H)``.
            sin: Rotary sine table with shape ``(N, D / H)``.
            attn_mask: Optional boolean attention mask with shape ``(N, N)``.

        Returns:
            Tokens with shape ``(B, N, D)``.
        """
        x = x + self.attn(self.norm1(x), cos, sin, attn_mask)
        return x + self.mlp(self.norm2(x))


class Qwen25VisionPatchMerger(nn.Module):
    """Concatenate each square of neighbouring tokens and project it to the output dimension.

    Args:
        config: Vision tower configuration.
    """

    def __init__(self, config: Qwen25VisionConfig) -> None:
        super().__init__()
        self.merged_dim = config.hidden_size * config.spatial_merge_size**2
        self.ln_q = Qwen25RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = nn.Sequential(
            nn.Linear(self.merged_dim, self.merged_dim),
            nn.GELU(),
            nn.Linear(self.merged_dim, config.out_hidden_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Merge tokens.

        Args:
            x: Tokens with shape ``(B, N, D)`` in merge order.

        Returns:
            Merged tokens with shape :math:`(B, N / s^2, D_{out})`, where :math:`s` is the spatial merge size.
        """
        b, n, d = x.shape
        return self.mlp(self.ln_q(x).reshape(b, n * d // self.merged_dim, self.merged_dim))


class Qwen25VisionTransformer(nn.Module):
    """Vision tower of Qwen2.5-VL.

    A ViT with a temporal patch embedding, a 2D rotary position embedding, RMS-normalized blocks with gated MLPs,
    attention restricted to windows except in ``config.fullatt_block_indexes``, and a merger that turns every
    2 x 2 square of patches into one output token. It follows ``Qwen2_5_VisionTransformerPretrainedModel`` in
    Hugging Face ``transformers`` v4.51.0, with two differences in interface and none in the result:

    - It takes images rather than flattened patches. Each image is repeated over ``temporal_patch_size`` frames,
      which is what the reference image processor does for a still image.
    - Windowed attention uses a mask over tokens kept in place, instead of reordering the tokens into windows and
      back. Attention with a rotary embedding is unchanged by permuting the tokens, so the output matches.

    The input must already be resized and normalized as the checkpoint's ``preprocessor_config.json`` describes:
    for the released checkpoints, values in :math:`[0, 1]` normalized with mean ``(0.48145466, 0.4578275,
    0.40821073)`` and std ``(0.26862954, 0.26130258, 0.27577711)``.

    Args:
        config: Vision tower configuration. Defaults to the Qwen2.5-VL-3B-Instruct values.

    Example:
        >>> config = Qwen25VisionConfig(depth=2, hidden_size=32, intermediate_size=48, num_heads=2,
        ...                             out_hidden_size=16, fullatt_block_indexes=(1,))
        >>> model = Qwen25VisionTransformer(config)
        >>> model(torch.rand(1, 3, 56, 84)).shape
        torch.Size([1, 6, 16])
    """

    def __init__(self, config: Optional[Qwen25VisionConfig] = None) -> None:
        super().__init__()
        self.config = config if config is not None else Qwen25VisionConfig()
        self.patch_embed = Qwen25VisionPatchEmbed(self.config)
        self.blocks = nn.ModuleList([Qwen25VisionBlock(self.config) for _ in range(self.config.depth)])
        self.merger = Qwen25VisionPatchMerger(self.config)

    def _rotary_tables(
        self, gh: int, gw: int, device: torch.device, dtype: torch.dtype
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        cfg = self.config
        rot_dim = cfg.hidden_size // cfg.num_heads // 2
        inv_freq = 1.0 / (cfg.rope_theta ** (torch.arange(0, rot_dim, 2, device=device, dtype=dtype) / rot_dim))
        rows = torch.arange(gh, device=device).unsqueeze(1).expand(gh, gw)
        cols = torch.arange(gw, device=device).unsqueeze(0).expand(gh, gw)
        rows = _merge_order(rows, cfg.spatial_merge_size).to(dtype)
        cols = _merge_order(cols, cfg.spatial_merge_size).to(dtype)
        angles = torch.cat((rows.unsqueeze(1) * inv_freq, cols.unsqueeze(1) * inv_freq), dim=-1)
        emb = torch.cat((angles, angles), dim=-1)
        return emb.cos(), emb.sin()

    def _window_mask(self, gh: int, gw: int, device: torch.device) -> torch.Tensor:
        cfg = self.config
        patches_per_window = cfg.window_size // cfg.patch_size
        windows_w = -(-gw // patches_per_window)
        row_window = torch.arange(gh, device=device) // patches_per_window
        col_window = torch.arange(gw, device=device) // patches_per_window
        window = _merge_order(row_window.unsqueeze(1) * windows_w + col_window.unsqueeze(0), cfg.spatial_merge_size)
        return window.unsqueeze(1) == window.unsqueeze(0)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Encode a batch of images into vision tokens.

        Args:
            images: Normalized images with shape ``(B, C, H, W)``. ``H`` and ``W`` must be multiples of
                ``patch_size * spatial_merge_size`` (28 for the released checkpoints).

        Returns:
            Tokens with shape :math:`(B, H W / (p s)^2, D_{out})` in raster order over the merged grid, where
            :math:`p` is the patch size and :math:`s` the spatial merge size.
        """
        cfg = self.config
        if images.ndim != 4 or images.shape[1] != cfg.in_channels:
            raise ValueError(f"Expected images of shape (B, {cfg.in_channels}, H, W). Got {tuple(images.shape)}.")
        unit = cfg.patch_size * cfg.spatial_merge_size
        h, w = images.shape[-2:]
        if h % unit != 0 or w % unit != 0:
            raise ValueError(f"Image height and width must be multiples of {unit}. Got {h} x {w}.")

        gh, gw = h // cfg.patch_size, w // cfg.patch_size
        x = self.patch_embed(images)
        cos, sin = self._rotary_tables(gh, gw, images.device, _compute_dtype(images.dtype))
        window_mask = self._window_mask(gh, gw, images.device)
        for i, block in enumerate(self.blocks):
            x = block(x, cos, sin, None if i in cfg.fullatt_block_indexes else window_mask)
        return self.merger(x)

    @classmethod
    def from_pretrained(
        cls, path: str | os.PathLike[str], config: Optional[Qwen25VisionConfig] = None
    ) -> Qwen25VisionTransformer:
        """Build the tower and load the ``visual.*`` weights of a Qwen2.5-VL checkpoint the user downloaded.

        kornia does not host these weights. Only the vision tensors are read, so the language model in the same
        shards is never copied into memory. The keys are mapped strictly: a missing or unexpected key raises.

        Args:
            path: A ``.safetensors`` file, or a checkpoint directory containing ``model.safetensors.index.json``
                or ``model.safetensors``.
            config: Configuration matching the checkpoint. Defaults to Qwen2.5-VL-3B-Instruct. The 7B checkpoint
                needs ``out_hidden_size=3584``; the 32B and 72B checkpoints need ``intermediate_size=3456`` and
                ``out_hidden_size`` 5120 and 8192.

        Returns:
            The tower with the checkpoint's weights, in float32 and in evaluation mode.
        """
        path = os.fspath(path)
        if os.path.isdir(path):
            index_path = os.path.join(path, "model.safetensors.index.json")
            if os.path.exists(index_path):
                with open(index_path) as f:
                    weight_map = json.load(f)["weight_map"]
                shards = sorted({shard for key, shard in weight_map.items() if key.startswith(_WEIGHTS_PREFIX)})
            else:
                shards = ["model.safetensors"]
            files = [os.path.join(path, shard) for shard in shards]
        else:
            files = [path]

        state_dict = {}
        for file in files:
            tensors = load_safetensors(file, include=lambda name: name.startswith(_WEIGHTS_PREFIX))
            state_dict.update({name[len(_WEIGHTS_PREFIX) :]: tensor for name, tensor in tensors.items()})

        model = cls(config)
        model.load_state_dict(state_dict, strict=True)
        return model.eval()
