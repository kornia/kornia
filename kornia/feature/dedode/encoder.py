from typing import Any, Optional, Tuple

import torch
from torch import nn

from kornia.core import Module, Tensor

from .vgg import vgg19_bn


class VGG19(Module):
    def __init__(self, amp: bool = False, amp_dtype: torch.dtype = torch.float16) -> None:
        super().__init__()
        self.layers = nn.ModuleList(vgg19_bn().features[:40])  # type: ignore
        # Maxpool layers: 6, 13, 26, 39
        self.amp = amp
        self.amp_dtype = amp_dtype

    def forward(self, x: Tensor, **kwargs):  # type: ignore[no-untyped-def]
        with torch.autocast("cuda", enabled=self.amp, dtype=self.amp_dtype):
            feats = []
            sizes = []
            for layer in self.layers:
                if isinstance(layer, nn.MaxPool2d):
                    feats.append(x)
                    sizes.append(x.shape[-2:])
                x = layer(x)
            return feats, sizes


class FrozenDINOv2(Module):
    def __init__(self, amp: bool = True, amp_dtype: torch.dtype = torch.float16, dinov2_weights: Optional[Any] = None):
        super().__init__()
        if dinov2_weights is None:
            dinov2_weights = torch.hub.load_state_dict_from_url(
                "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth", map_location="cpu"
            )
        from .transformer import vit_large

        vit_kwargs = dict(
            img_size=518,
            patch_size=14,
            init_values=1.0,
            ffn_layer="mlp",
            block_chunks=0,
        )
        dinov2_vitl14 = vit_large(**vit_kwargs).eval()
        dinov2_vitl14.load_state_dict(dinov2_weights)
        self.amp = amp
        self.amp_dtype = amp_dtype
        if self.amp:
            dinov2_vitl14 = dinov2_vitl14.to(self.amp_dtype)
        self.dinov2_vitl14 = [dinov2_vitl14]  # ugly hack to not show parameters to DDP

    def forward(self, x: Tensor):  # type: ignore[no-untyped-def]
        B, C, H, W = x.shape
        if self.dinov2_vitl14[0].device != x.device:
            self.dinov2_vitl14[0] = self.dinov2_vitl14[0].to(x.device).to(self.amp_dtype)
        with torch.inference_mode():
            dinov2_features_16 = self.dinov2_vitl14[0].forward_features(x.to(self.amp_dtype))
            features_16 = dinov2_features_16["x_norm_patchtokens"].permute(0, 2, 1).reshape(B, 1024, H // 14, W // 14)
        return [features_16.clone()], [(H // 14, W // 14)]  # clone from inference mode to use in autograd


class VGG_DINOv2(Module):
    def __init__(self, vgg_kwargs=None, dinov2_kwargs=None):  # type: ignore[no-untyped-def]
        if (vgg_kwargs is None) or (dinov2_kwargs is None):
            raise ValueError("Input kwargs please")
        super().__init__()
        self.vgg = VGG19(**vgg_kwargs)
        self.frozen_dinov2 = FrozenDINOv2(**dinov2_kwargs)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tuple[int, int]]:
        feats_vgg, sizes_vgg = self.vgg(x)
        feat_dinov2, size_dinov2 = self.frozen_dinov2(x)
        return feats_vgg + feat_dinov2, sizes_vgg + size_dinov2
