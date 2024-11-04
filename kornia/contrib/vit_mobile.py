from typing import Any, Dict, Tuple

import torch
from torch import nn

from kornia.core import Module, Tensor


def conv_1x1_bn(inp: int, oup: int) -> Module:
    return nn.Sequential(nn.Conv2d(inp, oup, 1, 1, 0, bias=False), nn.BatchNorm2d(oup), nn.SiLU())


def conv_nxn_bn(inp: int, oup: int, kernal_size: int = 3, stride: int = 1) -> Module:
    return nn.Sequential(nn.Conv2d(inp, oup, kernal_size, stride, 1, bias=False), nn.BatchNorm2d(oup), nn.SiLU())


class PreNorm(Module):
    def __init__(self, dim: int, fn: Module) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x: Tensor, **kwargs: Dict[str, Any]) -> Tensor:
        return self.fn(self.norm(x), **kwargs)


class FeedForward(Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim), nn.SiLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, dim), nn.Dropout(dropout)
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class Attention(Module):
    def __init__(self, dim: int, heads: int = 8, dim_head: int = 64, dropout: float = 0.0) -> None:
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head**-0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout)) if project_out else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        qkv = self.to_qkv(x).chunk(3, dim=-1)

        b, p, n, hd = qkv[0].shape
        q, k, v = (t.reshape(b, p, n, self.heads, hd // self.heads).transpose(2, 3) for t in qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        out = torch.matmul(attn, v)
        out = out.transpose(2, 3).reshape(b, p, n, hd)
        return self.to_out(out)


class Transformer(Module):
    """Transformer block described in ViT.

    Paper: https://arxiv.org/abs/2010.11929
    Based on: https://github.com/lucidrains/vit-pytorch

    Args:
        dim: input dimension.
        depth: depth for transformer block.
        heads: number of heads in multi-head attention layer.
        dim_head: head size.
        mlp_dim: dimension of the FeedForward layer.
        dropout: dropout ratio, defaults to 0.
    """

    def __init__(self, dim: int, depth: int, heads: int, dim_head: int, mlp_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        PreNorm(dim, Attention(dim, heads, dim_head, dropout)),
                        PreNorm(dim, FeedForward(dim, mlp_dim, dropout)),
                    ]
                )
            )

    def forward(self, x: Tensor) -> Tensor:
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class MV2Block(Module):
    """MV2 block described in MobileNetV2.

    Paper: https://arxiv.org/pdf/1801.04381
    Based on: https://github.com/tonylins/pytorch-mobilenet-v2

    Args:
        inp: input channel.
        oup: output channel.
        stride: stride for convolution, defaults to 1, set to 2 if down-sample.
        expansion: expansion ratio for hidden dimension, defaults to 4.
    """

    def __init__(self, inp: int, oup: int, stride: int = 1, expansion: int = 4) -> None:
        super().__init__()
        self.stride = stride

        hidden_dim = int(inp * expansion)
        self.use_res_connect = self.stride == 1 and inp == oup

        if expansion == 1:
            self.conv = nn.Sequential(
                # depthwise
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.SiLU(),
                # pointwise
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pointwise
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.SiLU(),
                # depthwise
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.SiLU(),
                # pointwise
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x: Tensor) -> Tensor:
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileViTBlock(Module):
    """MobileViT block mentioned in MobileViT.

    Args:
        dim: input dimension of Transformer.
        depth: depth of Transformer.
        channel: input channel.
        kernel_size: kernel size.
        patch_size: patch size for folding and unfloding.
        mlp_dim: dimension of the FeedForward layer in Transformer.
        dropout: dropout ratio, defaults to 0.
    """

    def __init__(
        self,
        dim: int,
        depth: int,
        channel: int,
        kernel_size: int,
        patch_size: Tuple[int, int],
        mlp_dim: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.ph, self.pw = patch_size

        self.conv1 = conv_nxn_bn(channel, channel, kernel_size)
        self.conv2 = conv_1x1_bn(channel, dim)

        self.transformer = Transformer(dim, depth, 4, 8, mlp_dim, dropout)

        self.conv3 = conv_1x1_bn(dim, channel)
        self.conv4 = conv_nxn_bn(2 * channel, channel, kernel_size)

    def forward(self, x: Tensor) -> Tensor:
        y = x.clone()

        # Local representations
        x = self.conv1(x)
        x = self.conv2(x)

        b, d, h, w = x.shape
        nh, nw = h // self.ph, w // self.pw

        # Global representations
        # [b, d, h, w] -> [b * d * nh, nw, ph, pw]
        x = x.reshape(b * d * nh, self.ph, nw, self.pw).transpose(1, 2)
        # [b * d * nh, nw, ph, pw] -> [b, (ph pw), (nh nw), d]
        x = x.reshape(b, d, nh * nw, self.ph * self.pw).transpose(1, 3)

        x = self.transformer(x)

        # [b, (ph pw), (nh nw), d] -> [b * d * nh, nw, ph, pw]
        x = x.transpose(1, 3).reshape(b * d * nh, nw, self.ph, self.pw)
        # [b * d * nh, nw, ph, pw] -> [b, d, h, w]
        x = x.transpose(1, 2).reshape(b, d, h, w)

        # Fusion
        x = self.conv3(x)
        x = torch.cat((x, y), 1)
        x = self.conv4(x)
        return x


class MobileViT(Module):
    """Module MobileViT. Default arguments is for MobileViT XXS.

    Paper: https://arxiv.org/abs/2110.02178
    Based on: https://github.com/chinhsuanwu/mobilevit-pytorch

    Args:
        mode: 'xxs', 'xs' or 's', defaults to 'xxs'.
        in_channels: the number of channels for the input image.
        patch_size: image_size must be divisible by patch_size.
        dropout: dropout ratio in Transformer.

    Example:
        >>> img = torch.rand(1, 3, 256, 256)
        >>> mvit = MobileViT(mode='xxs')
        >>> mvit(img).shape
        torch.Size([1, 320, 8, 8])
    """

    def __init__(
        self, mode: str = "xxs", in_channels: int = 3, patch_size: Tuple[int, int] = (2, 2), dropout: float = 0.0
    ) -> None:
        super().__init__()
        if mode == "xxs":
            expansion = 2
            dims = [64, 80, 96]
            channels = [16, 16, 24, 24, 48, 48, 64, 64, 80, 80, 320]
        elif mode == "xs":
            expansion = 4
            dims = [96, 120, 144]
            channels = [16, 32, 48, 48, 64, 64, 80, 80, 96, 96, 384]
        elif mode == "s":
            expansion = 4
            dims = [144, 192, 240]
            channels = [16, 32, 64, 64, 96, 96, 128, 128, 160, 160, 640]

        kernel_size = 3
        depth = [2, 4, 3]

        self.conv1 = conv_nxn_bn(in_channels, channels[0], stride=2)

        self.mv2 = nn.ModuleList([])
        self.mv2.append(MV2Block(channels[0], channels[1], 1, expansion))
        self.mv2.append(MV2Block(channels[1], channels[2], 2, expansion))
        self.mv2.append(MV2Block(channels[2], channels[3], 1, expansion))
        self.mv2.append(MV2Block(channels[2], channels[3], 1, expansion))  # Repeat
        self.mv2.append(MV2Block(channels[3], channels[4], 2, expansion))
        self.mv2.append(MV2Block(channels[5], channels[6], 2, expansion))
        self.mv2.append(MV2Block(channels[7], channels[8], 2, expansion))

        self.mvit = nn.ModuleList([])
        self.mvit.append(
            MobileViTBlock(dims[0], depth[0], channels[5], kernel_size, patch_size, int(dims[0] * 2), dropout=dropout)
        )
        self.mvit.append(
            MobileViTBlock(dims[1], depth[1], channels[7], kernel_size, patch_size, int(dims[1] * 4), dropout=dropout)
        )
        self.mvit.append(
            MobileViTBlock(dims[2], depth[2], channels[9], kernel_size, patch_size, int(dims[2] * 4), dropout=dropout)
        )

        self.conv2 = conv_1x1_bn(channels[-2], channels[-1])

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.mv2[0](x)

        x = self.mv2[1](x)
        x = self.mv2[2](x)
        x = self.mv2[3](x)  # Repeat

        x = self.mv2[4](x)
        x = self.mvit[0](x)

        x = self.mv2[5](x)
        x = self.mvit[1](x)

        x = self.mv2[6](x)
        x = self.mvit[2](x)
        x = self.conv2(x)
        return x
