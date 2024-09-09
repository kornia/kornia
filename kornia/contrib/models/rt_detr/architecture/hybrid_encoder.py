"""Based on the code from
https://github.com/PaddlePaddle/PaddleDetection/blob/ec37e66685f3bc5a38cd13f60685acea175922e1/
ppdet/modeling/transformers/hybrid_encoder.py."""

from __future__ import annotations

import copy
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.fusion import fuse_conv_bn_weights

from kornia.contrib.models.common import ConvNormAct
from kornia.core import Module, Tensor, concatenate, pad
from kornia.utils._compat import torch_meshgrid


class RepVggBlock(Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv1 = ConvNormAct(in_channels, out_channels, 3, act="none")
        self.conv2 = ConvNormAct(in_channels, out_channels, 1, act="none")
        self.act = nn.SiLU(inplace=True)
        self.conv: Optional[nn.Conv2d] = None

    def forward(self, x: Tensor) -> Tensor:
        if self.conv is not None:
            out = self.act(self.conv(x))
        else:
            out = self.act(self.conv1(x) + self.conv2(x))
        return out

    @torch.no_grad()
    def optimize_for_deployment(self) -> None:
        def _fuse_conv_bn_weights(m: ConvNormAct) -> tuple[nn.Parameter, nn.Parameter]:
            if m.norm.running_mean is None or m.norm.running_var is None:
                raise ValueError

            return fuse_conv_bn_weights(
                m.conv.weight,
                m.conv.bias,
                m.norm.running_mean,
                m.norm.running_var,
                m.norm.eps,
                m.norm.weight,
                m.norm.bias,
            )

        kernel3x3, bias3x3 = _fuse_conv_bn_weights(self.conv1)
        kernel1x1, bias1x1 = _fuse_conv_bn_weights(self.conv2)
        kernel3x3.add_(pad(kernel1x1, [1, 1, 1, 1]))
        bias3x3.add_(bias1x1)

        self.conv = nn.Conv2d(kernel3x3.shape[1], kernel3x3.shape[0], 3, 1, 1)
        self.conv.weight = kernel3x3
        self.conv.bias = bias3x3


class CSPRepLayer(Module):
    def __init__(self, in_channels: int, out_channels: int, num_blocks: int, expansion: float = 1.0) -> None:
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        self.conv1 = ConvNormAct(in_channels, hidden_channels, 1, act="silu")
        self.conv2 = ConvNormAct(in_channels, hidden_channels, 1, act="silu")
        self.bottlenecks = nn.Sequential(*[RepVggBlock(hidden_channels, hidden_channels) for _ in range(num_blocks)])
        self.conv3 = (
            ConvNormAct(hidden_channels, out_channels, 1, act="silu")
            if hidden_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.conv3(self.bottlenecks(self.conv1(x)) + self.conv2(x))


# almost identical to nn.TransformerEncoderLayer
# but add positional embeddings to q and k
class AIFI(Module):
    def __init__(self, embed_dim: int, num_heads: int, dim_feedforward: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout)  # NOTE: batch_first = False

        self.linear1 = nn.Linear(embed_dim, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, embed_dim)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.act = nn.GELU()

    def forward(self, x: Tensor) -> Tensor:
        # using post-norm
        N, C, H, W = x.shape
        x = x.permute(2, 3, 0, 1).flatten(0, 1)  # (N, C, H, W) -> (H * W, N, C)

        # NOTE: cache pos_emb as buffer
        pos_emb = self.build_2d_sincos_pos_emb(W, H, C, device=x.device, dtype=x.dtype)
        q = k = x + pos_emb

        attn, _ = self.self_attn(q, k, x, need_weights=True)
        x = self.norm1(x + self.dropout1(attn))
        x = self.norm2(x + self.dropout2(self.ffn(x)))

        x = x.view(H, W, N, C).permute(2, 3, 0, 1)  # (H * W, N, C) -> (N, C, H, W)
        return x

    def ffn(self, x: Tensor) -> Tensor:
        return self.linear2(self.dropout(self.act(self.linear1(x))))

    # TODO: make this into a reusable function
    # https://github.com/facebookresearch/moco-v3/blob/main/vits.py#L53
    # https://github.com/PaddlePaddle/PaddleDetection/blob/79267419e1743157f376a7cb251e01caa3338ce0/ppdet/modeling/transformers/hybrid_encoder.py#L217
    @staticmethod
    def build_2d_sincos_pos_emb(
        w: int,
        h: int,
        embed_dim: int,
        temp: float = 10_000.0,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> Tensor:
        """Construct 2D sin-cos positional embeddings.

        Args:
            w: width of the image or feature map
            h: height of the image or feature map
            embed_dim: embedding dimension
            temp: temperature coefficient
            device: device to place the positional embeddings
            dtype: data type of the positional embeddings

        Returns:
            positional embeddings, shape :math:`(H * W, 1, C)`
        """
        xs = torch.arange(w, device=device, dtype=dtype)
        ys = torch.arange(h, device=device, dtype=dtype)
        grid_x, grid_y = torch_meshgrid([xs, ys], indexing="ij")

        pos_dim = embed_dim // 4
        omega = torch.arange(pos_dim, device=device, dtype=dtype) / pos_dim
        omega = 1.0 / (temp**omega)

        out_x = grid_x.reshape(-1, 1) * omega.view(1, -1)
        out_y = grid_y.reshape(-1, 1) * omega.view(1, -1)

        pos_emb = concatenate([out_x.sin(), out_x.cos(), out_y.sin(), out_y.cos()], 1)
        return pos_emb.unsqueeze(1)  # (H * W, 1, C)


class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer: nn.Module, num_layers: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])
        self.num_layers = num_layers

    def forward(self, src: Tensor) -> Tensor:  # NOTE: Missing src_mask: Tensor = None, pos_embed: Tensor = None
        output = src
        for layer in self.layers:
            output = layer(output)

        return output


class CCFM(Module):
    def __init__(self, num_fmaps: int, hidden_dim: int, expansion: float = 1.0) -> None:
        super().__init__()
        self.lateral_convs = nn.ModuleList()
        self.fpn_blocks = nn.ModuleList()
        for _ in range(num_fmaps - 1):
            self.lateral_convs.append(ConvNormAct(hidden_dim, hidden_dim, 1, 1, "silu"))
            self.fpn_blocks.append(CSPRepLayer(hidden_dim * 2, hidden_dim, 3, expansion))

        self.downsample_convs = nn.ModuleList()
        self.pan_blocks = nn.ModuleList()
        for _ in range(num_fmaps - 1):
            self.downsample_convs.append(ConvNormAct(hidden_dim, hidden_dim, 3, 2, "silu"))
            self.pan_blocks.append(CSPRepLayer(hidden_dim * 2, hidden_dim, 3, expansion))

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
    def __init__(self, in_channels: list[int], hidden_dim: int, dim_feedforward: int, expansion: float = 1.0) -> None:
        super().__init__()
        self.input_proj = nn.ModuleList(
            [
                ConvNormAct(  # To align the naming strategy for the official weights
                    in_ch, hidden_dim, 1, act="none", conv_naming="0", norm_naming="1", act_naming="2"
                )
                for in_ch in in_channels
            ]
        )
        encoder_layer = AIFI(hidden_dim, 8, dim_feedforward)
        self.encoder = nn.Sequential(TransformerEncoder(encoder_layer, 1))
        self.ccfm = CCFM(len(in_channels), hidden_dim, expansion)

    def forward(self, fmaps: list[Tensor]) -> list[Tensor]:
        projected_maps = [proj(fmap) for proj, fmap in zip(self.input_proj, fmaps)]
        projected_maps[-1] = self.encoder(projected_maps[-1])
        new_fmaps = self.ccfm(projected_maps)
        return new_fmaps
