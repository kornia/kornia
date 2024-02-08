"""Module that implement Vision Transformer (ViT).

Paper: https://paperswithcode.com/paper/an-image-is-worth-16x16-words-transformers-1

Based on: `https://towardsdatascience.com/implementing-visualttransformer-in-pytorch-184f9f16f632`

Added some tricks from: `https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py`
"""
from __future__ import annotations

import os
from typing import Any, Callable

import torch
from torch import nn

from kornia.core import Module, Tensor, concatenate
from kornia.core.check import KORNIA_CHECK

__all__ = ["VisionTransformer"]


# recommended checkpoint from https://github.com/google-research/vision_transformer
_base_url = "https://storage.googleapis.com/vit_models/augreg/"
_checkpoint_dict = {
    "vit_l/16": "L_16-i21k-300ep-lr_0.001-aug_strong1-wd_0.1-do_0.0-sd_0.0.npz",
    "vit_b/16": "B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0.npz",
    "vit_s/16": "S_16-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0.npz",
    "vit_ti/16": "Ti_16-i21k-300ep-lr_0.001-aug_none-wd_0.03-do_0.0-sd_0.0.npz",
    "vit_b/32": "B_32-i21k-300ep-lr_0.001-aug_light1-wd_0.1-do_0.0-sd_0.0.npz",
    "vit_s/32": "S_32-i21k-300ep-lr_0.001-aug_none-wd_0.1-do_0.0-sd_0.0.npz",
}


def download_to_torch_hub(url: str, progress: bool = True) -> str:
    torch_hub_dir = torch.hub.get_dir()
    filename = os.path.basename(url)
    file_path = os.path.join(torch_hub_dir, filename)

    if not os.path.exists(file_path):
        torch.hub.download_url_to_file(url, file_path, progress=progress)

    return file_path


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

    @torch.no_grad()
    def load_jax_checkpoint(self, checkpoint: str) -> VisionTransformer:
        import numpy as np

        if checkpoint.startswith("http"):
            checkpoint = download_to_torch_hub(checkpoint)

        jax_ckpt = np.load(checkpoint)

        def _get(key: str) -> Tensor:
            return torch.from_numpy(jax_ckpt[key])

        patch_embed = self.patch_embedding
        patch_embed.cls_token.copy_(_get("cls"))
        patch_embed.backbone.weight.copy_(_get("embedding/kernel").permute(3, 2, 0, 1))  # conv weight
        patch_embed.backbone.bias.copy_(_get("embedding/bias"))
        patch_embed.positions.copy_(_get("Transformer/posembed_input/pos_embedding").squeeze(0))  # resize

        for i, block in enumerate(self.encoder.blocks):
            prefix = f"Transformer/encoderblock_{i}/"
            block[0].fn[0].weight.copy_(_get(prefix + "LayerNorm_0/scale"))
            block[0].fn[0].bias.copy_(_get(prefix + "LayerNorm_0/bias"))

            mha_prefix = prefix + "MultiHeadDotProductAttention_1/"
            qkv_weight = [_get(mha_prefix + f"{x}/kernel") for x in ["query", "key", "value"]]
            block[0].fn[1].qkv.weight.copy_(concatenate(qkv_weight, 1).flatten(1).T)
            qkv_bias = [_get(mha_prefix + f"{x}/bias") for x in ["query", "key", "value"]]
            block[0].fn[1].qkv.bias.copy_(concatenate(qkv_bias, 0).flatten())
            block[0].fn[1].projection.weight.copy_(_get(mha_prefix + "out/kernel").flatten(0, 1).T)
            block[0].fn[1].projection.bias.copy_(_get(mha_prefix + "out/bias"))

            block[1].fn[0].weight.copy_(_get(prefix + "LayerNorm_2/scale"))
            block[1].fn[0].bias.copy_(_get(prefix + "LayerNorm_2/bias"))
            block[1].fn[1][0].weight.copy_(_get(prefix + "MlpBlock_3/Dense_0/kernel").T)
            block[1].fn[1][0].bias.copy_(_get(prefix + "MlpBlock_3/Dense_0/bias"))
            block[1].fn[1][3].weight.copy_(_get(prefix + "MlpBlock_3/Dense_1/kernel").T)
            block[1].fn[1][3].bias.copy_(_get(prefix + "MlpBlock_3/Dense_1/bias"))

        self.norm.weight.copy_(_get("Transformer/encoder_norm/scale"))
        self.norm.bias.copy_(_get("Transformer/encoder_norm/bias"))
        return self

    @staticmethod
    def from_config(variant: str, pretrained: bool = False, **kwargs: Any) -> VisionTransformer:
        """Build ViT model based on the given config string. The format is `vit_{size}/{patch_size}`.
        E.g. vit_b/16 means ViT-Base, patch size 16x16.

        Args:
            config: ViT model config
        Returns:
            The respective ViT model

        Example:
            >>> from kornia.contrib import VisionTransformer
            >>> vit_model = VisionTransformer.from_config("vit_b/16")
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
            KORNIA_CHECK(variant in _checkpoint_dict, f"Variant {variant} does not have pre-trained checkpoint")
            model.load_jax_checkpoint(_base_url + _checkpoint_dict[variant])

        return model
