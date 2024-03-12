"""Module that implement Vision Transformer (ViT).

Paper: https://paperswithcode.com/paper/an-image-is-worth-16x16-words-transformers-1

Based on: `https://towardsdatascience.com/implementing-visualttransformer-in-pytorch-184f9f16f632`

Added some tricks from: `https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py`
"""

from __future__ import annotations

from typing import Any, Callable

import torch
from torch import nn

from kornia.core import Module, Tensor, concatenate
from kornia.core.check import KORNIA_CHECK

__all__ = ["VisionTransformer"]


class ResidualAdd(Module):
    def __init__(self, fn: Callable[..., Tensor]) -> None:
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor, **kwargs: Any) -> Tensor:
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x


class FeedForward(nn.Sequential):
    def __init__(self, in_features: int, hidden_features: int, out_features: int, dropout_rate: float = 0.0) -> None:
        super().__init__(
            nn.Linear(in_features, hidden_features),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_features, out_features),
            nn.Dropout(dropout_rate),  # added one extra as in timm
        )


class MultiHeadAttention(Module):
    def __init__(self, emb_size: int, num_heads: int, att_drop: float, proj_drop: float) -> None:
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        head_size = emb_size // num_heads  # from timm
        self.scale = head_size**-0.5  # from timm

        if self.emb_size % self.num_heads:
            raise ValueError(
                f"Size of embedding inside the transformer decoder must be visible by number of heads"
                f"for correct multi-head attention "
                f"Got: {self.emb_size} embedding size and {self.num_heads} numbers of heads"
            )

        # fuse the queries, keys and values in one matrix
        self.qkv = nn.Linear(emb_size, emb_size * 3)
        self.att_drop = nn.Dropout(att_drop)
        self.projection = nn.Linear(emb_size, emb_size)
        self.projection_drop = nn.Dropout(proj_drop)  # added timm trick

    def forward(self, x: Tensor) -> Tensor:
        B, N, C = x.shape
        # split keys, queries and values in num_heads
        # NOTE: the line below differs from timm
        # timm: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # sum up over the last axis
        att = torch.einsum("bhqd, bhkd -> bhqk", q, k) * self.scale
        att = att.softmax(dim=-1)
        att = self.att_drop(att)

        # sum up over the third axis
        out = torch.einsum("bhal, bhlv -> bhav ", att, v)
        out = out.permute(0, 2, 1, 3).contiguous().view(B, N, -1)
        out = self.projection(out)
        out = self.projection_drop(out)
        return out


class TransformerEncoderBlock(nn.Sequential):
    def __init__(self, embed_dim: int, num_heads: int, dropout_rate: float, dropout_attn: float) -> None:
        super().__init__(
            ResidualAdd(
                nn.Sequential(
                    nn.LayerNorm(embed_dim, 1e-6),
                    MultiHeadAttention(embed_dim, num_heads, dropout_attn, dropout_rate),
                    nn.Dropout(dropout_rate),
                )
            ),
            ResidualAdd(
                nn.Sequential(
                    nn.LayerNorm(embed_dim, 1e-6),
                    FeedForward(embed_dim, embed_dim * 4, embed_dim, dropout_rate=dropout_rate),
                    nn.Dropout(dropout_rate),
                )
            ),
        )


class TransformerEncoder(Module):
    def __init__(
        self,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        dropout_rate: float = 0.0,
        dropout_attn: float = 0.0,
    ) -> None:
        super().__init__()
        self.blocks = nn.Sequential(
            *(TransformerEncoderBlock(embed_dim, num_heads, dropout_rate, dropout_attn) for _ in range(depth))
        )
        self.results: list[Tensor] = []

    def forward(self, x: Tensor) -> Tensor:
        self.results = []
        out = x
        for m in self.blocks.children():
            out = m(out)
            self.results.append(out)
        return out


class PatchEmbedding(Module):
    """Compute the 2d image patch embedding ready to pass to transformer encoder."""

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 768,
        patch_size: int = 16,
        image_size: int = 224,
        backbone: Module | None = None,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.patch_size = patch_size

        # logic needed in case a backbone is passed
        self.backbone = backbone or nn.Conv2d(in_channels, out_channels, kernel_size=patch_size, stride=patch_size)
        if backbone is not None:
            out_channels, feat_size = self._compute_feats_dims((in_channels, image_size, image_size))
            self.out_channels = out_channels
        else:
            feat_size = (image_size // patch_size) ** 2

        self.cls_token = nn.Parameter(torch.randn(1, 1, out_channels))
        self.positions = nn.Parameter(torch.randn(feat_size + 1, out_channels))

    def _compute_feats_dims(self, image_size: tuple[int, int, int]) -> tuple[int, int]:
        out = self.backbone(torch.zeros(1, *image_size)).detach()
        return out.shape[-3], out.shape[-2] * out.shape[-1]

    def forward(self, x: Tensor) -> Tensor:
        x = self.backbone(x)
        B, N, _, _ = x.shape
        x = x.view(B, N, -1).permute(0, 2, 1)  # BxNxE
        cls_tokens = self.cls_token.repeat(B, 1, 1)  # Bx1xE
        # prepend the cls token to the input
        x = concatenate([cls_tokens, x], dim=1)  # Bx(N+1)xE
        # add position embedding
        x += self.positions
        return x


class VisionTransformer(Module):
    """Vision transformer (ViT) module.

    The module is expected to be used as operator for different vision tasks.

    The method is inspired from existing implementations of the paper :cite:`dosovitskiy2020vit`.

    .. warning::
        This is an experimental API subject to changes in favor of flexibility.

    Args:
        image_size: the size of the input image.
        patch_size: the size of the patch to compute the embedding.
        in_channels: the number of channels for the input.
        embed_dim: the embedding dimension inside the transformer encoder.
        depth: the depth of the transformer.
        num_heads: the number of attention heads.
        dropout_rate: dropout rate.
        dropout_attn: attention dropout rate.
        backbone: an nn.Module to compute the image patches embeddings.

    Example:
        >>> img = torch.rand(1, 3, 224, 224)
        >>> vit = VisionTransformer(image_size=224, patch_size=16)
        >>> vit(img).shape
        torch.Size([1, 197, 768])
    """

    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        dropout_rate: float = 0.0,
        dropout_attn: float = 0.0,
        backbone: Module | None = None,
    ) -> None:
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_size = embed_dim

        self.patch_embedding = PatchEmbedding(in_channels, embed_dim, patch_size, image_size, backbone)
        hidden_dim = self.patch_embedding.out_channels
        self.encoder = TransformerEncoder(hidden_dim, depth, num_heads, dropout_rate, dropout_attn)
        self.norm = nn.LayerNorm(hidden_dim, 1e-6)

    @property
    def encoder_results(self) -> list[Tensor]:
        return self.encoder.results

    def forward(self, x: Tensor) -> Tensor:
        if not isinstance(x, Tensor):
            raise TypeError(f"Input x type is not a Tensor. Got: {type(x)}")

        if self.image_size not in (*x.shape[-2:],) and x.shape[-3] != self.in_channels:
            raise ValueError(
                f"Input image shape must be Bx{self.in_channels}x{self.image_size}x{self.image_size}. Got: {x.shape}"
            )

        out = self.patch_embedding(x)
        out = self.encoder(out)
        out = self.norm(out)
        return out

    @staticmethod
    def from_config(variant: str, pretrained: bool = False, **kwargs: Any) -> VisionTransformer:
        """Build ViT model based on the given config string. The format is ``vit_{size}/{patch_size}``.
        E.g. ``vit_b/16`` means ViT-Base, patch size 16x16. If ``pretrained=True``, AugReg weights are loaded.
        The weights are hosted on HuggingFace's model hub: https://huggingface.co/kornia.

        .. note::
            The available weights are: ``vit_l/16``, ``vit_b/16``, ``vit_s/16``, ``vit_ti/16``,
            ``vit_b/32``, ``vit_s/32``.

        Args:
            variant: ViT model variant e.g. ``vit_b/16``.
            pretrained: whether to load pre-trained AugReg weights.
            kwargs: other keyword arguments that will be passed to :func:`kornia.contrib.vit.VisionTransformer`.
        Returns:
            The respective ViT model

        Example:
            >>> from kornia.contrib import VisionTransformer
            >>> vit_model = VisionTransformer.from_config("vit_b/16", pretrained=True)
        """
        model_type, patch_size_str = variant.split("/")
        patch_size = int(patch_size_str)

        model_config = {
            "vit_ti": {"embed_dim": 192, "depth": 12, "num_heads": 3},
            "vit_s": {"embed_dim": 384, "depth": 12, "num_heads": 6},
            "vit_b": {"embed_dim": 768, "depth": 12, "num_heads": 12},
            "vit_l": {"embed_dim": 1024, "depth": 24, "num_heads": 16},
            "vit_h": {"embed_dim": 1280, "depth": 32, "num_heads": 16},
        }[model_type]
        kwargs.update(model_config, patch_size=patch_size)

        model = VisionTransformer(**kwargs)

        if pretrained:
            url = _get_weight_url(variant)
            state_dict = torch.hub.load_state_dict_from_url(url)
            model.load_state_dict(state_dict)

        return model


_AVAILABLE_WEIGHTS = ["vit_l/16", "vit_b/16", "vit_s/16", "vit_ti/16", "vit_b/32", "vit_s/32"]


def _get_weight_url(variant: str) -> str:
    """Return the URL of the model weights."""
    KORNIA_CHECK(variant in _AVAILABLE_WEIGHTS, f"Variant {variant} does not have pre-trained checkpoint")
    model_type, patch_size = variant.split("/")
    return f"https://huggingface.co/kornia/{model_type}{patch_size}_augreg_i21k_r224/resolve/main/{model_type}-{patch_size}.pth"
