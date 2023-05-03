"""Based on the code from https://github.com/PaddlePaddle/PaddleDetection/blob/5d1f888362241790000950e2b63115dc8d1c
6019/ppdet/modeling/transformers/hybrid_encoder.py."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from kornia.core import Module, Tensor, concatenate

from .common import conv_norm_act


class CSPRepLayer(Module):
    def __init__(self, in_channels: int, out_channels: int, num_blocks: int):
        super().__init__()
        self.conv1 = conv_norm_act(in_channels, out_channels, 1, act="silu")
        self.conv2 = conv_norm_act(in_channels, out_channels, 1, act="silu")
        blocks = [conv_norm_act(out_channels, out_channels, 3, act="silu") for _ in range(num_blocks)]
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x: Tensor) -> Tensor:
        return self.blocks(self.conv1(x)) + self.conv2(x)


class AIFI(Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        # note: batch_first=False, norm_first=False
        self.encoder = nn.TransformerEncoderLayer(hidden_dim, 8, hidden_dim * 4, 0, "gelu")

    def forward(self, x: Tensor) -> Tensor:
        N, C, H, W = x.shape
        x = x.permute(2, 3, 0, 1).flatten(0, 1)  # (N, C, H, W) -> (H * W, N, C)
        out = self.encoder(x + self.build_2d_sincos_pos_emb(H, W, C))
        out = out.view(H, W, N, C).permute(2, 3, 0, 1)  # (H * W, N, C) -> (N, C, H, W)
        return out

    @staticmethod
    def build_2d_sincos_pos_emb(w: int, h: int, embed_dim: int, temp: float = 10_000.0) -> Tensor:
        xs = torch.arange(w)
        ys = torch.arange(h)
        grid_x, grid_y = torch.meshgrid(xs, ys)

        pos_dim = embed_dim // 4
        omega = torch.arange(pos_dim) / pos_dim
        omega = 1.0 / (temp**omega)

        out_x = grid_x.reshape(-1, 1) * omega.view(1, -1)
        out_y = grid_y.reshape(-1, 1) * omega.view(1, -1)

        pos_emb = concatenate([out_x.sin(), out_x.cos(), out_y.sin(), out_y.cos()], 1)
        return pos_emb.unsqueeze(1)  # (H * W, N, C)


class CCFM(Module):
    def __init__(self, num_fmaps: int, hidden_dim: int):
        super().__init__()
        self.lateral_convs = nn.ModuleList()
        self.fpn_blocks = nn.ModuleList()
        for _ in range(num_fmaps - 1):
            self.lateral_convs.append(conv_norm_act(hidden_dim, hidden_dim, 1, 1, "silu"))
            self.fpn_blocks.append(CSPRepLayer(hidden_dim * 2, hidden_dim, 3))

        self.downsample_convs = nn.ModuleList()
        self.pan_blocks = nn.ModuleList()
        for _ in range(num_fmaps - 1):
            self.downsample_convs.append(conv_norm_act(hidden_dim, hidden_dim, 3, 2, "silu"))
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
    def __init__(self, in_channels: list[int], hidden_dim: int = 256):
        super().__init__()
        self.projs = nn.ModuleList()
        for in_ch in in_channels:
            self.projs.append(nn.Sequential(nn.Conv2d(in_ch, hidden_dim, 1, bias=False), nn.BatchNorm2d(hidden_dim)))

        self.aifi = AIFI(hidden_dim)
        self.ccfm = CCFM(len(in_channels), hidden_dim)

    def forward(self, fmaps: list[Tensor]) -> list[Tensor]:
        fmaps = [proj(fmap) for proj, fmap in zip(self.projs, fmaps)]
        fmaps[-1] = self.aifi(fmaps[-1])
        fmaps = self.ccfm(fmaps)
        return fmaps
