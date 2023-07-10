# https://github.com/microsoft/Cream/blob/main/TinyViT/models/tiny_vit.py

import itertools
from typing import Any

import torch
from torch import nn
from torch.utils import checkpoint

from kornia.contrib.models.common import DropPath
from kornia.contrib.models.sam.architecture.image_encoder import window_partition, window_unpartition
from kornia.core import Module, Tensor


class ConvBN(nn.Sequential):
    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, groups: int = 1
    ) -> None:
        super().__init__()
        self.c = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)


class PatchEmbed(nn.Sequential):
    def __init__(self, in_channels: int, embed_dim: int, activation: type[Module] = nn.GELU) -> None:
        super().__init__()
        self.seq = nn.Sequential(
            ConvBN(in_channels, embed_dim // 2, 3, 2, 1), activation(), ConvBN(embed_dim // 2, embed_dim, 3, 2, 1)
        )


class MBConv(Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        expansion_ratio: float,
        activation: type[Module] = nn.GELU,
        drop_path: float = 0.0,
    ) -> None:
        super().__init__()
        hidden_channels = int(in_channels * expansion_ratio)
        self.conv1 = ConvBN(in_channels, hidden_channels, 1)  # point-wise
        self.act1 = activation()
        self.conv2 = ConvBN(hidden_channels, hidden_channels, 3, 1, 1, hidden_channels)  # depth-wise
        self.act2 = activation()
        self.conv3 = ConvBN(hidden_channels, out_channels, 1)
        self.drop_path = DropPath(drop_path)
        self.act3 = activation()

    def forward(self, x: Tensor) -> Tensor:
        out = self.act1(self.conv1(x))
        out = self.act2(self.conv2(out))
        out = self.drop_path(self.conv3(out))
        out = self.act3(x + out)
        return out


class PatchMerging(Module):
    def __init__(
        self, input_resolution: tuple[int, int], dim: int, out_dim: int, activation: type[Module] = nn.GELU
    ) -> None:
        super().__init__()
        self.input_resolution = input_resolution
        self.conv1 = ConvBN(dim, out_dim, 1)
        self.act1 = activation()
        self.conv2 = ConvBN(out_dim, out_dim, 3, 2, 1, groups=out_dim)
        self.act2 = activation()
        self.conv3 = ConvBN(out_dim, out_dim, 1)

    def forward(self, x: Tensor) -> Tensor:
        # (B, H * W, C) -> (B, C, H, W)
        if x.ndim == 3:
            x = x.transpose(1, 2).unflatten(2, self.input_resolution)

        x = self.act1(self.conv1(x))
        x = self.act2(self.conv2(x))
        x = self.conv3(x)

        # (B, C, H, W) -> (B, H * W, C)
        x = x.flatten(2).transpose(1, 2)
        return x


class ConvLayer(Module):
    def __init__(
        self,
        dim: int,
        input_resolution: tuple[int, int],
        depth: int,
        activation: type[Module] = nn.GELU,
        drop_path: float | list[float] = 0.0,
        downsample: type[PatchMerging] | None = None,
        use_checkpoint: bool = False,
        out_dim: int | None = None,
        conv_expand_ratio: float = 4.0,
    ) -> None:
        super().__init__()
        self.use_checkpoint = use_checkpoint

        # build blocks
        if not isinstance(drop_path, list):
            drop_path = [drop_path] * depth
        self.blocks = nn.ModuleList(
            [MBConv(dim, dim, conv_expand_ratio, activation, drop_path[i]) for i in range(depth)]
        )

        # patch merging layer
        self.downsample: Module
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, out_dim=out_dim, activation=activation)
        else:
            self.downsample = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        for blk in self.blocks:
            x = checkpoint.checkpoint(blk, x) if self.use_checkpoint else blk(x)
        x = self.downsample(x)
        return x


class MLP(nn.Sequential):
    def __init__(
        self,
        in_features: int,
        hidden_features: int | None = None,
        out_features: int | None = None,
        activation: type[Module] = nn.GELU,
        drop: float = 0.0,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.norm = nn.LayerNorm(in_features)
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act1 = activation()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop)


class Attention(Module):
    def __init__(
        self,
        dim: int,
        key_dim: int,
        num_heads: int = 8,
        attn_ratio: float = 4.0,
        resolution: tuple[int, int] = (14, 14),
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.scale = key_dim**-0.5
        self.key_dim = key_dim
        self.nh_kd = key_dim * num_heads
        self.d = int(attn_ratio * key_dim)
        self.dh = int(attn_ratio * key_dim) * num_heads
        self.attn_ratio = attn_ratio
        h = self.dh + self.nh_kd * 2

        self.norm = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, h)
        self.proj = nn.Linear(self.dh, dim)

        points = list(itertools.product(range(resolution[0]), range(resolution[1])))
        N = len(points)
        attention_offsets: dict[tuple[int, int], int] = {}
        idxs = []
        for p1 in points:
            for p2 in points:
                offset = (abs(p1[0] - p2[0]), abs(p1[1] - p2[1]))
                if offset not in attention_offsets:
                    attention_offsets[offset] = len(attention_offsets)
                idxs.append(attention_offsets[offset])

        self.attention_biases = nn.Parameter(torch.zeros(num_heads, len(attention_offsets)))
        self.register_buffer('attention_bias_idxs', torch.LongTensor(idxs).view(N, N), persistent=False)
        self.ab: Tensor | None = None

    @torch.no_grad()
    def train(self, mode=True):
        super().train(mode)
        self.ab = None if (mode and self.ab is not None) else self.attention_biases[:, self.attention_bias_idxs]

    def forward(self, x: Tensor) -> Tensor:
        B, N, _ = x.shape
        x = self.norm(x)
        qkv = self.qkv(x)
        qkv = qkv.view(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
        q, k, v = qkv.split([self.key_dim, self.key_dim, self.d], dim=3)

        bias = self.attention_biases[:, self.attention_bias_idxs] if self.training else self.ab
        attn = (q @ k.transpose(-2, -1)) * self.scale + bias

        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, self.dh)
        x = self.proj(x)
        return x


class TinyViTBlock(Module):
    r"""TinyViT Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int, int]): Input resolution.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        drop (float, optional): Dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        local_conv_size (int): the kernel size of the convolution between
                               Attention and MLP. Default: 3
        activation: the activation function. Default: nn.GELU
    """

    def __init__(
        self,
        dim: int,
        input_resolution: tuple[int, int],
        num_heads: int,
        window_size: int = 7,
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
        drop_path: float = 0.0,
        local_conv_size: int = 3,
        activation: type[Module] = nn.GELU,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.mlp_ratio = mlp_ratio

        self.drop_path = DropPath(drop_path)

        # assert dim % num_heads == 0, 'dim must be divisible by num_heads'
        head_dim = dim // num_heads

        window_resolution = (window_size, window_size)
        self.attn = Attention(dim, head_dim, num_heads, 1.0, window_resolution)
        self.mlp = MLP(dim, int(dim * mlp_ratio), activation=activation, drop=drop)
        self.local_conv = ConvBN(dim, dim, local_conv_size, 1, local_conv_size // 2, dim)

    def forward(self, x: Tensor) -> Tensor:
        H, W = self.input_resolution
        B, L, C = x.shape
        # assert L == H * W, "input feature has wrong size"
        res_x = x
        if H == self.window_size and W == self.window_size:
            x = self.attn(x)
        else:
            x = x.view(B, H, W, C)
            x, (Hp, Wp) = window_partition(x, self.window_size)
            x = self.attn(x)
            x = window_unpartition(x, self.window_size, (Hp, Wp), (H, W))
            x = x.view(B, L, C)

        x = res_x + self.drop_path(x)

        x = x.transpose(1, 2).reshape(B, C, H, W)
        x = self.local_conv(x)
        x = x.view(B, C, L).transpose(1, 2)

        x = x + self.drop_path(self.mlp(x))
        return x

    def extra_repr(self) -> str:
        return (
            f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, "
            f"window_size={self.window_size}, mlp_ratio={self.mlp_ratio}"
        )


class BasicLayer(Module):
    """A basic TinyViT layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        drop (float, optional): Dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
        local_conv_size: the kernel size of the depthwise convolution between attention and MLP. Default: 3
        activation: the activation function. Default: nn.GELU
        out_dim: the output dimension of the layer. Default: dim
    """

    def __init__(
        self,
        dim: int,
        input_resolution: tuple[int, int],
        depth: int,
        num_heads: int,
        window_size: int,
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
        drop_path: float | list[float] = 0.0,
        downsample: type[PatchMerging] | None = None,
        use_checkpoint: bool = False,
        local_conv_size: int = 3,
        activation: type[Module] = nn.GELU,
        out_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList(
            [
                TinyViTBlock(
                    dim,
                    input_resolution,
                    num_heads,
                    window_size,
                    mlp_ratio,
                    drop,
                    drop_path[i] if isinstance(drop_path, list) else drop_path,
                    local_conv_size,
                    activation,
                )
                for i in range(depth)
            ]
        )

        # patch merging layer
        self.downsample: Module
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, out_dim=out_dim, activation=activation)
        else:
            self.downsample = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        for blk in self.blocks:
            x = checkpoint.checkpoint(blk, x) if self.use_checkpoint else blk(x)
        x = self.downsample(x)
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"


class TinyViT(Module):
    def __init__(
        self,
        img_size: int = 224,
        in_chans: int = 3,
        # num_classes=1000,
        embed_dims: list[int] = [96, 192, 384, 768],
        depths: list[int] = [2, 2, 6, 2],
        num_heads: list[int] = [3, 6, 12, 24],
        window_sizes: list[int] = [7, 7, 14, 7],
        mlp_ratio: float = 4.0,
        drop_rate: float = 0.0,
        drop_path_rate: float = 0.1,
        use_checkpoint: bool = False,
        mbconv_expand_ratio: float = 4.0,
        local_conv_size: int = 3,
        # layer_lr_decay: float = 1.0,
        activation: type[Module] = nn.GELU,
    ) -> None:
        super().__init__()

        # self.num_classes = num_classes
        self.depths = depths
        self.num_layers = len(depths)
        self.mlp_ratio = mlp_ratio

        self.patch_embed = PatchEmbed(in_chans, embed_dims[0], activation)
        self.patches_resolution = (img_size // 4, img_size // 4)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.Sequential()
        for i_layer in range(self.num_layers):
            kwargs: dict[str, Any] = {
                'dim': embed_dims[i_layer],
                'input_resolution': (
                    self.patches_resolution[0] // (2**i_layer),
                    self.patches_resolution[1] // (2**i_layer),
                ),
                'depth': depths[i_layer],
                'drop_path': dpr[sum(depths[:i_layer]) : sum(depths[: i_layer + 1])],
                'downsample': PatchMerging if (i_layer < self.num_layers - 1) else None,
                'use_checkpoint': use_checkpoint,
                'out_dim': embed_dims[min(i_layer + 1, len(embed_dims) - 1)],
                'activation': activation,
            }
            layer: ConvLayer | BasicLayer
            if i_layer == 0:
                layer = ConvLayer(conv_expand_ratio=mbconv_expand_ratio, **kwargs)
            else:
                layer = BasicLayer(
                    num_heads=num_heads[i_layer],
                    window_size=window_sizes[i_layer],
                    mlp_ratio=self.mlp_ratio,
                    drop=drop_rate,
                    local_conv_size=local_conv_size,
                    **kwargs,
                )
            self.layers.append(layer)

        # Classifier head
        # self.norm_head = nn.LayerNorm(embed_dims[-1])
        # self.head = nn.Linear(embed_dims[-1], num_classes) if num_classes > 0 else torch.nn.Identity()

        # init weights
        self.apply(self._init_weights)
        # self.set_layer_lr_decay(layer_lr_decay)

    # def set_layer_lr_decay(self, layer_lr_decay):
    #     decay_rate = layer_lr_decay

    #     # layers -> blocks (depth)
    #     depth = sum(self.depths)
    #     lr_scales = [decay_rate ** (depth - i - 1) for i in range(depth)]

    #     def _set_lr_scale(m, scale):
    #         for p in m.parameters():
    #             p.lr_scale = scale

    #     self.patch_embed.apply(lambda x: _set_lr_scale(x, lr_scales[0]))
    #     i = 0
    #     for layer in self.layers:
    #         for block in layer.blocks:
    #             block.apply(lambda x: _set_lr_scale(x, lr_scales[i]))
    #             i += 1
    #         if not isinstance(layer.downsample, nn.Identity):
    #             layer.downsample.apply(lambda x: _set_lr_scale(x, lr_scales[i - 1]))
    #     assert i == depth
    #     for m in [self.norm_head, self.head]:
    #         m.apply(lambda x: _set_lr_scale(x, lr_scales[-1]))

    #     for k, p in self.named_parameters():
    #         p.param_name = k

    #     def _check_lr_scale(m):
    #         for p in m.parameters():
    #             assert hasattr(p, 'lr_scale'), p.param_name

    #     self.apply(_check_lr_scale)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        # NOTE: how about batch norm and conv2d?

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'attention_biases'}

    def forward(self, x: Tensor) -> Tensor:
        x = self.patch_embed(x)
        x = self.layers(x)
        x = x.mean(1)
        # x = self.norm_head(x)
        # x = self.head(x)
        return x
