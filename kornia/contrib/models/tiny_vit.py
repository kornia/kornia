# https://github.com/microsoft/Cream/blob/8dc38822b99fff8c262c585a32a4f09ac504d693/TinyViT/models/tiny_vit.py
# https://github.com/ChaoningZhang/MobileSAM/blob/01ea8d0f5590082f0c1ceb0a3e2272593f20154b/mobile_sam/modeling/tiny_vit_sam.py

from __future__ import annotations

import itertools
import warnings
from typing import Any, Optional

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils import checkpoint

from kornia.contrib.models.common import DropPath, LayerNorm2d, window_partition, window_unpartition
from kornia.core import Module, Tensor
from kornia.core.check import KORNIA_CHECK


def _make_pair(x: int | tuple[int, int]) -> tuple[int, int]:
    return (x, x) if isinstance(x, int) else x


class ConvBN(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        groups: int = 1,
        activation: type[Module] = nn.Identity,
    ) -> None:
        super().__init__()
        self.c = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = activation()


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
        self.conv1 = ConvBN(in_channels, hidden_channels, 1, activation=activation)  # point-wise
        self.conv2 = ConvBN(hidden_channels, hidden_channels, 3, 1, 1, hidden_channels, activation)  # depth-wise
        self.conv3 = ConvBN(hidden_channels, out_channels, 1)
        self.drop_path = DropPath(drop_path)
        self.act = activation()

    def forward(self, x: Tensor) -> Tensor:
        return self.act(x + self.drop_path(self.conv3(self.conv2(self.conv1(x)))))


class PatchMerging(Module):
    def __init__(
        self,
        input_resolution: int | tuple[int, int],
        dim: int,
        out_dim: int,
        stride: int,
        activation: type[Module] = nn.GELU,
    ) -> None:
        KORNIA_CHECK(stride in (1, 2), "stride must be either 1 or 2")
        super().__init__()
        self.input_resolution = _make_pair(input_resolution)
        self.conv1 = ConvBN(dim, out_dim, 1, activation=activation)
        self.conv2 = ConvBN(out_dim, out_dim, 3, stride, 1, groups=out_dim, activation=activation)
        self.conv3 = ConvBN(out_dim, out_dim, 1)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim == 3:
            x = x.transpose(1, 2).unflatten(2, self.input_resolution)  # (B, H * W, C) -> (B, C, H, W)
        x = self.conv3(self.conv2(self.conv1(x)))
        x = x.flatten(2).transpose(1, 2)  # (B, C, H, W) -> (B, H * W, C)
        return x


class ConvLayer(Module):
    def __init__(
        self,
        dim: int,
        depth: int,
        activation: type[Module] = nn.GELU,
        drop_path: float | list[float] = 0.0,
        downsample: Optional[Module] = None,
        use_checkpoint: bool = False,
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
        self.downsample = downsample

    def forward(self, x: Tensor) -> Tensor:
        for blk in self.blocks:
            x = checkpoint.checkpoint(blk, x) if self.use_checkpoint else blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x


class MLP(nn.Sequential):
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int,
        activation: type[Module] = nn.GELU,
        drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(in_features)
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act1 = activation()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop)


# NOTE: differences from image_encoder.Attention:
# - different relative position encoding mechanism (separable/decomposed vs joint)
# - this impl supports attn_ratio (increase output size for value), though it is not used
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

        indices, attn_offset_size = self.build_attention_bias(resolution)
        self.attention_biases = nn.Parameter(torch.zeros(num_heads, attn_offset_size))
        self.register_buffer("attention_bias_idxs", indices, persistent=False)
        self.attention_bias_idxs: Tensor
        self.ab: Optional[Tensor] = None

    @staticmethod
    def build_attention_bias(resolution: tuple[int, int]) -> tuple[Tensor, int]:
        points = list(itertools.product(range(resolution[0]), range(resolution[1])))
        attention_offsets: dict[tuple[int, int], int] = {}
        idxs: list[int] = []
        for p1 in points:
            for p2 in points:
                offset = (abs(p1[0] - p2[0]), abs(p1[1] - p2[1]))
                if offset not in attention_offsets:
                    attention_offsets[offset] = len(attention_offsets)
                idxs.append(attention_offsets[offset])

        N = len(points)
        indices = torch.LongTensor(idxs).view(N, N)
        attn_offset_size = len(attention_offsets)
        return indices, attn_offset_size

    # is this really necessary?
    @torch.no_grad()
    def train(self, mode: bool = True) -> Attention:
        super().train(mode)
        self.ab = None if (mode and self.ab is not None) else self.attention_biases[:, self.attention_bias_idxs]
        return self

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
    def __init__(
        self,
        dim: int,
        input_resolution: int | tuple[int, int],
        num_heads: int,
        window_size: int = 7,
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
        drop_path: float = 0.0,
        local_conv_size: int = 3,
        activation: type[Module] = nn.GELU,
    ) -> None:
        KORNIA_CHECK(dim % num_heads == 0, "dim must be divislbe by num_heads")
        super().__init__()
        self.input_resolution = _make_pair(input_resolution)
        self.window_size = window_size
        head_dim = dim // num_heads

        self.attn = Attention(dim, head_dim, num_heads, 1.0, (window_size, window_size))
        self.drop_path1 = DropPath(drop_path)
        self.local_conv = ConvBN(dim, dim, local_conv_size, 1, local_conv_size // 2, dim)
        self.mlp = MLP(dim, int(dim * mlp_ratio), dim, activation, drop)
        self.drop_path2 = DropPath(drop_path)

    def forward(self, x: Tensor) -> Tensor:
        H, W = self.input_resolution
        B, L, C = x.shape
        res_x = x

        x = x.view(B, H, W, C)
        x, pad_hw = window_partition(x, self.window_size)  # (B * num_windows, window_size, window_size, C)
        x = self.attn(x.flatten(1, 2))
        x = window_unpartition(x, self.window_size, pad_hw, (H, W))
        x = x.view(B, L, C)

        x = res_x + self.drop_path1(x)

        x = x.transpose(1, 2).reshape(B, C, H, W)
        x = self.local_conv(x)
        x = x.view(B, C, L).transpose(1, 2)

        x = x + self.drop_path2(self.mlp(x))
        return x


class BasicLayer(Module):
    def __init__(
        self,
        dim: int,
        input_resolution: int | tuple[int, int],
        depth: int,
        num_heads: int,
        window_size: int,
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
        drop_path: float | list[float] = 0.0,
        downsample: Optional[Module] = None,
        use_checkpoint: bool = False,
        local_conv_size: int = 3,
        activation: type[Module] = nn.GELU,
    ) -> None:
        super().__init__()
        self.use_checkpoint = use_checkpoint

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
        self.downsample = downsample

    def forward(self, x: Tensor) -> Tensor:
        for blk in self.blocks:
            x = checkpoint.checkpoint(blk, x) if self.use_checkpoint else blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x


class TinyViT(Module):
    """TinyViT model, as described in https://arxiv.org/abs/2207.10666

    Args:
        img_size: Size of input image.
        in_chans: Number of input image's channels.
        num_classes: Number of output classes.
        embed_dims: List of embedding dimensions.
        depths: List of block count for each downsampling stage
        num_heads: List of attention heads used in self-attention for each downsampling stage.
        window_sizes: List of self-attention's window size for each downsampling stage.
        mlp_ratio: Ratio of MLP dimension to embedding dimension in self-attention.
        drop_rate: Dropout rate.
        drop_path_rate: Stochastic depth rate.
        use_checkpoint: Whether to use activation checkpointing to trade compute for memory.
        mbconv_expand_ratio: Expansion ratio used in MBConv block.
        local_conv_size: Kernel size of convolution used in TinyViTBlock
        activation: activation function.
        mobile_same: Whether to use modifications for MobileSAM.
    """

    def __init__(
        self,
        img_size: int = 224,
        in_chans: int = 3,
        num_classes: int = 1000,
        embed_dims: list[int] = [96, 192, 384, 768],
        depths: list[int] = [2, 2, 6, 2],
        num_heads: list[int] = [3, 6, 12, 24],
        window_sizes: list[int] = [7, 7, 14, 7],
        mlp_ratio: float = 4.0,
        drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        use_checkpoint: bool = False,
        mbconv_expand_ratio: float = 4.0,
        local_conv_size: int = 3,
        # layer_lr_decay: float = 1.0,
        activation: type[Module] = nn.GELU,
        mobile_sam: bool = False,
    ) -> None:
        super().__init__()
        self.img_size = img_size
        self.mobile_sam = mobile_sam
        self.neck: Optional[Module]
        if mobile_sam:
            # MobileSAM adjusts the stride to match the total stride of other ViT backbones
            # used in the original SAM (stride 16)
            strides = [2, 2, 1, 1]
            self.neck = nn.Sequential(
                nn.Conv2d(embed_dims[-1], 256, 1, bias=False),
                LayerNorm2d(256),
                nn.Conv2d(256, 256, 3, 1, 1, bias=False),
                LayerNorm2d(256),
            )
        else:
            strides = [2, 2, 2, 1]
            self.neck = None

        self.patch_embed = PatchEmbed(in_chans, embed_dims[0], activation)
        input_resolution = img_size // 4

        # NOTE: if we don't support training, this might be unimportant
        # stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        # build layers
        n_layers = len(depths)
        layers = []
        for i_layer, (embed_dim, depth, num_heads_i, window_size, stride) in enumerate(
            zip(embed_dims, depths, num_heads, window_sizes, strides)
        ):
            out_dim = embed_dims[min(i_layer + 1, len(embed_dims) - 1)]
            downsample = (
                PatchMerging(input_resolution, embed_dim, out_dim, stride, activation)
                if (i_layer < n_layers - 1)
                else None
            )
            kwargs: dict[str, Any] = {
                "dim": embed_dim,
                "depth": depth,
                "drop_path": dpr[sum(depths[:i_layer]) : sum(depths[: i_layer + 1])],
                "downsample": downsample,
                "use_checkpoint": use_checkpoint,
                "activation": activation,
            }
            layer: ConvLayer | BasicLayer
            if i_layer == 0:
                layer = ConvLayer(conv_expand_ratio=mbconv_expand_ratio, **kwargs)
            else:
                layer = BasicLayer(
                    input_resolution=input_resolution,
                    num_heads=num_heads_i,
                    window_size=window_size,
                    mlp_ratio=mlp_ratio,
                    drop=drop_rate,
                    local_conv_size=local_conv_size,
                    **kwargs,
                )
            layers.append(layer)
            input_resolution //= stride
        self.layers = nn.Sequential(*layers)
        self.feat_size = input_resolution  # final feature map size

        # Classifier head
        # NOTE: this is redundant for MobileSAM, but we still need it
        # to load pre-trained weights with strict=True
        # TODO: enable strict=False, or host our own weights
        self.norm_head = nn.LayerNorm(embed_dims[-1])
        self.head = nn.Linear(embed_dims[-1], num_classes)

    def forward(self, x: Tensor) -> Tensor:
        """Classify images if ``mobile_sam=False``, produce feature maps if ``mobile_sam=True``."""
        x = self.patch_embed(x)
        x = self.layers(x)

        if self.mobile_sam:
            # MobileSAM
            x = x.unflatten(1, (self.feat_size, self.feat_size)).permute(0, 3, 1, 2)
            x = self.neck(x)  # type: ignore
        else:
            # classification
            x = x.mean(1)
            x = self.head(self.norm_head(x))
        return x

    @staticmethod
    def from_config(variant: str, pretrained: bool | str = False, **kwargs: Any) -> TinyViT:
        """Create a TinyViT model from pre-defined variants.

        Args:
            variant: TinyViT variant. Possible values: ``'5m'``, ``'11m'``, ``'21m'``.
            pretrained: whether to use pre-trained weights. Possible values: ``False``, ``True``, ``'in22k'``,
                ``'in1k'``. For TinyViT-21M (``variant='21m'``), ``'in1k_384'``, ``'in1k_512'`` are also available.
            **kwargs: other keyword arguments that will be passed to :class:`TinyViT`.

        .. note::
            When ``img_size`` is different from the pre-trained size, bicubic interpolation will be performed on
            attention biases. When using ``pretrained=True``, ImageNet-1k checkpoint (``'in1k'``) is used.
            For feature extraction or fine-tuning, ImageNet-22k checkpoint (``'in22k'``) is preferred.
        """
        KORNIA_CHECK(variant in ("5m", "11m", "21m"), "Only variant 5m, 11m, and 21m are supported")
        return {"5m": _tiny_vit_5m, "11m": _tiny_vit_11m, "21m": _tiny_vit_21m}[variant](pretrained, **kwargs)


def _load_pretrained(model: TinyViT, url: str) -> TinyViT:
    model_state_dict = model.state_dict()
    state_dict = torch.hub.load_state_dict_from_url(url)

    # official checkpoint has "model" key
    if "model" in state_dict:
        state_dict = state_dict["model"]

    # https://github.com/microsoft/Cream/blob/8dc38822b99fff8c262c585a32a4f09ac504d693/TinyViT/utils.py#L163
    # bicubic interpolate attention biases
    ab_keys = [k for k in state_dict.keys() if "attention_biases" in k]
    for k in ab_keys:
        n_heads1, L1 = state_dict[k].shape
        n_heads2, L2 = model_state_dict[k].shape
        KORNIA_CHECK(n_heads1 == n_heads2, f"Fail to load {k}. Pre-trained checkpoint should have num_heads={n_heads1}")

        if L1 != L2:
            S1 = int(L1**0.5)
            S2 = int(L2**0.5)
            attention_biases = state_dict[k].view(1, n_heads1, S1, S1)
            attention_biases = F.interpolate(attention_biases, size=(S2, S2), mode="bicubic")
            state_dict[k] = attention_biases.view(n_heads2, L2)

    if state_dict["head.weight"].shape[0] != model.head.out_features:
        msg = "Number of classes does not match pre-trained checkpoint's. Resetting classification head to zeros"
        warnings.warn(msg)
        state_dict["head.weight"] = torch.zeros_like(model.head.weight)
        state_dict["head.bias"] = torch.zeros_like(model.head.bias)

    model.load_state_dict(state_dict)
    return model


def _tiny_vit_5m(pretrained: bool | str = False, **kwargs: Any) -> TinyViT:
    model = TinyViT(
        embed_dims=[64, 128, 160, 320],
        depths=[2, 2, 6, 2],
        num_heads=[2, 4, 5, 10],
        window_sizes=[7, 7, 14, 7],
        drop_path_rate=0.0,
        **kwargs,
    )

    if pretrained:
        if pretrained is True:
            pretrained = "in1k"

        url = {
            "in22k": (
                "https://github.com/wkcn/TinyViT-model-zoo/releases/download/checkpoints/tiny_vit_5m_22k_distill.pth"
            ),
            "in1k": "https://github.com/wkcn/TinyViT-model-zoo/releases/download/checkpoints/tiny_vit_5m_22kto1k_distill.pth",
        }[pretrained]
        model = _load_pretrained(model, url)

    return model


def _tiny_vit_11m(pretrained: bool | str = False, **kwargs: Any) -> TinyViT:
    model = TinyViT(
        embed_dims=[64, 128, 256, 448],
        depths=[2, 2, 6, 2],
        num_heads=[2, 4, 8, 14],
        window_sizes=[7, 7, 14, 7],
        drop_path_rate=0.1,
        **kwargs,
    )

    if pretrained:
        if pretrained is True:
            pretrained = "in1k"

        url = {
            "in22k": (
                "https://github.com/wkcn/TinyViT-model-zoo/releases/download/checkpoints/tiny_vit_11m_22k_distill.pth"
            ),
            "in1k": "https://github.com/wkcn/TinyViT-model-zoo/releases/download/checkpoints/tiny_vit_11m_22kto1k_distill.pth",
        }[pretrained]
        model = _load_pretrained(model, url)

    return model


def _tiny_vit_21m(pretrained: bool | str = False, **kwargs: Any) -> TinyViT:
    model = TinyViT(
        embed_dims=[96, 192, 384, 576],
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 18],
        window_sizes=[7, 7, 14, 7],
        drop_path_rate=0.2,
        **kwargs,
    )

    if pretrained:
        if pretrained is True:
            pretrained = "in1k"
            img_size = kwargs.get("img_size", 224)
            if img_size >= 384:
                pretrained = "in1k_384"
            if img_size >= 512:
                pretrained = "in1k_512"

        url = {
            "in22k": (
                "https://github.com/wkcn/TinyViT-model-zoo/releases/download/checkpoints/tiny_vit_21m_22k_distill.pth"
            ),
            "in1k": "https://github.com/wkcn/TinyViT-model-zoo/releases/download/checkpoints/tiny_vit_21m_22kto1k_distill.pth",
            "in1k_384": "https://github.com/wkcn/TinyViT-model-zoo/releases/download/checkpoints/tiny_vit_21m_22kto1k_384_distill.pth",
            "in1k_512": "https://github.com/wkcn/TinyViT-model-zoo/releases/download/checkpoints/tiny_vit_21m_22kto1k_512_distill.pth",
        }[pretrained]
        model = _load_pretrained(model, url)

    return model
