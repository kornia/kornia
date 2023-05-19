"""Based on the code from
https://github.com/PaddlePaddle/PaddleDetection/blob/ec37e66685f3bc5a38cd13f60685acea175922e1/
ppdet/modeling/transformers/hybrid_encoder.py."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from kornia.contrib.models.common import ConvNormAct
from kornia.core import Device, Dtype, Module, Tensor, concatenate


# NOTE: conv2 can be fused into conv1
class RepVggBlock(Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = ConvNormAct(in_channels, out_channels, 3, act="none")
        self.conv2 = ConvNormAct(in_channels, out_channels, 1, act="none")
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        return self.act(self.conv1(x) + self.conv2(x))


class CSPRepLayer(Module):
    def __init__(self, in_channels: int, out_channels: int, num_blocks: int) -> None:
        super().__init__()
        self.conv1 = ConvNormAct(in_channels, out_channels, 1, act="silu")
        self.conv2 = ConvNormAct(in_channels, out_channels, 1, act="silu")
        self.bottlenecks = nn.Sequential(*[RepVggBlock(out_channels, out_channels) for _ in range(num_blocks)])

    def forward(self, x: Tensor) -> Tensor:
        return self.bottlenecks(self.conv1(x)) + self.conv2(x)


# almost identical to nn.TransformerEncoderLayer
# but add positional embeddings to q and k
class AIFI(Module):
    def __init__(self, embed_dim: int, num_heads: int, dim_feedforward: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout)  # NOTE: batch_first = False
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(embed_dim)

        self.linear1 = nn.Linear(embed_dim, dim_feedforward)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, embed_dim)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x: Tensor) -> Tensor:
        # using post-norm
        N, C, H, W = x.shape
        x = x.permute(2, 3, 0, 1).flatten(0, 1)  # (N, C, H, W) -> (H * W, N, C)

        # NOTE: cache build_2d_sincos_pos_emb to buffer, if input size is known?
        q = k = x + self.build_2d_sincos_pos_emb(W, H, C, device=x.device, dtype=x.dtype)
        x = self.norm1(x + self.dropout1(self.self_attn(q, k, x, need_weights=False)[0]))
        x = self.norm2(x + self.dropout2(self.ffn(x)))

        x = x.view(H, W, N, C).permute(2, 3, 0, 1)  # (H * W, N, C) -> (N, C, H, W)
        return x

    def ffn(self, x: Tensor) -> Tensor:
        return self.linear2(self.dropout(self.act(self.linear1(x))))

    # TODO: make this into a reusable function
    @staticmethod
    def build_2d_sincos_pos_emb(
        w: int, h: int, embed_dim: int, temp: float = 10_000.0, device: Device = None, dtype: Dtype = None
    ) -> Tensor:
        xs = torch.arange(w, device=device, dtype=dtype)
        ys = torch.arange(h, device=device, dtype=dtype)
        grid_x, grid_y = torch.meshgrid(xs, ys)

        pos_dim = embed_dim // 4
        omega = torch.arange(pos_dim, device=device, dtype=dtype) / pos_dim
        omega = 1.0 / (temp**omega)

        out_x = grid_x.reshape(-1, 1) * omega.view(1, -1)
        out_y = grid_y.reshape(-1, 1) * omega.view(1, -1)

        pos_emb = concatenate([out_x.sin(), out_x.cos(), out_y.sin(), out_y.cos()], 1)
        return pos_emb.unsqueeze(1)  # (H * W, 1, C)


class CCFM(Module):
    def __init__(self, num_fmaps: int, hidden_dim: int) -> None:
        super().__init__()
        self.lateral_convs = nn.ModuleList()
        self.fpn_blocks = nn.ModuleList()
        for _ in range(num_fmaps - 1):
            self.lateral_convs.append(ConvNormAct(hidden_dim, hidden_dim, 1, 1, "silu"))
            self.fpn_blocks.append(CSPRepLayer(hidden_dim * 2, hidden_dim, 3))

        self.downsample_convs = nn.ModuleList()
        self.pan_blocks = nn.ModuleList()
        for _ in range(num_fmaps - 1):
            self.downsample_convs.append(ConvNormAct(hidden_dim, hidden_dim, 3, 2, "silu"))
            self.pan_blocks.append(CSPRepLayer(hidden_dim * 2, hidden_dim, 3))

    def forward(self, fmaps: list[Tensor]) -> list[Tensor]:
        # fmaps is ordered from hi-res to low-res
        fmaps = list(fmaps)  # shallow clone

        # new_fmaps is ordered from low-res to hi-res
        new_fmaps = [fmaps.pop()]
        while fmaps:
            new_fmaps[-1] = self.lateral_convs[len(new_fmaps) - 1](new_fmaps[-1])
            up_lowres_fmap = F.interpolate(new_fmaps[-1], scale_factor=2.0, mode="nearest")
            hires_fmap = fmaps.pop()

            concat_fmap = concatenate([up_lowres_fmap, hires_fmap], 1)
            new_fmaps.append(self.fpn_blocks[len(new_fmaps) - 1](concat_fmap))

        fmaps = [new_fmaps.pop()]
        while new_fmaps:
            down_hires_fmap = self.downsample_convs[len(fmaps) - 1](fmaps[-1])
            lowres_fmap = new_fmaps.pop()

            concat_fmap = concatenate([down_hires_fmap, lowres_fmap], 1)
            fmaps.append(self.pan_blocks[len(fmaps) - 1](concat_fmap))

        return fmaps


class HybridEncoder(Module):
    def __init__(self, in_channels: list[int], hidden_dim: int, dim_feedforward: int) -> None:
        super().__init__()
        self.input_proj = nn.ModuleList([ConvNormAct(in_ch, hidden_dim, 1, act="none") for in_ch in in_channels])
        self.aifi = AIFI(hidden_dim, 8, dim_feedforward)
        self.ccfm = CCFM(len(in_channels), hidden_dim)

    def forward(self, fmaps: list[Tensor]) -> list[Tensor]:
        fmaps = [proj(fmap) for proj, fmap in zip(self.input_proj, fmaps)]
        fmaps[-1] = self.aifi(fmaps[-1])
        fmaps = self.ccfm(fmaps)
        return fmaps
