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

from typing import Any, Optional, Tuple

import torch
from torch import nn

from kornia.core.download import load_state_dict_from_url

from .vgg import vgg19_bn

# Module level, like every other weight registry in kornia, so the CI weights
# cache can enumerate it -- see tests/core/test_weights_prefetch.py. The DeDoDe
# G descriptor pulls this backbone in, so CI needs it cached.
urls: dict[str, str] = {
    "dinov2_vitl14": "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth"
}


class VGG19(nn.Module):
    """Implement the VGG19 encoder for feature encoding.

    Args:
        amp: Whether to use automatic mixed precision.
        amp_dtype: The data type for automatic mixed precision.
    """

    def __init__(self, amp: bool = False, amp_dtype: torch.dtype = torch.float16) -> None:
        super().__init__()
        self.layers = nn.ModuleList(vgg19_bn().features[:40])
        # Maxpool layers: 6, 13, 26, 39
        self.amp = amp
        self.amp_dtype = amp_dtype

    def forward(self, x: torch.Tensor, **kwargs):
        """Run this DeDoDe module forward.

        Args:
            x: Input image tensor with shape ``(B, 3, H, W)`` (BGR or RGB).
            **kwargs: Additional keyword arguments accepted for compatibility with the shared encoder interface. They
                are not used by this VGG19 batch-normalized backbone wrapper.

        Returns:
            Tuple of ``(features, sizes)`` where ``features`` is a list of intermediate feature maps
            captured before each max-pool, and ``sizes`` is a list of their spatial dimensions.
        """
        # AMP runs only for CUDA tensors.
        with torch.autocast("cuda", enabled=self.amp and x.is_cuda, dtype=self.amp_dtype):
            feats = []
            sizes = []
            for layer in self.layers:
                if isinstance(layer, nn.MaxPool2d):
                    feats.append(x)
                    sizes.append(x.shape[-2:])
                x = layer(x)
            return feats, sizes


class FrozenDINOv2(nn.Module):
    """Implement a wrapper for a frozen DINOv2 vision transformer backbone."""

    def __init__(self, amp: bool = True, amp_dtype: torch.dtype = torch.float16, dinov2_weights: Optional[Any] = None):
        super().__init__()
        if dinov2_weights is None:
            dinov2_weights = load_state_dict_from_url(urls["dinov2_vitl14"], map_location="cpu")
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
        # The backbone keeps the dtype of the checkpoint; `forward` casts it to ``amp_dtype`` once it runs on CUDA.
        self.dinov2_vitl14 = [dinov2_vitl14]  # ugly hack to not show parameters to DDP

    def forward(self, x: torch.Tensor):
        """Run this DeDoDe module forward.

        ``amp_dtype`` is CUDA mixed precision, as in the other DeDoDe components: on a CUDA input with ``amp=True``
        the backbone and the image are cast to ``amp_dtype``. On other devices (CPU, MPS), or with ``amp=False``, the
        backbone is cast to the dtype and device of ``x``, so the features come back in the dtype of ``x``.

        Args:
            x: Input image tensor with shape ``(B, 3, H, W)``.

        Returns:
            Tuple of ``([features_16], [(H//14, W//14)])`` where ``features_16`` has shape
            ``(B, 1024, H/14, W/14)``.
        """
        B, _C, H, W = x.shape
        use_amp = self.amp and x.is_cuda
        backbone = self.dinov2_vitl14[0]
        if use_amp:
            if backbone.device != x.device or backbone.cls_token.dtype != self.amp_dtype:
                # Cast on the host, then move: the half copy is what travels to the device.
                self.dinov2_vitl14[0] = backbone.to(self.amp_dtype).to(x.device)
            backbone_input = x.to(self.amp_dtype)
        else:
            if backbone.device != x.device or backbone.cls_token.dtype != x.dtype:
                self.dinov2_vitl14[0] = backbone.to(x.device, x.dtype)
            backbone_input = x
        with torch.inference_mode():
            dinov2_features_16 = self.dinov2_vitl14[0].forward_features(backbone_input)
            features_16 = dinov2_features_16["x_norm_patchtokens"].permute(0, 2, 1).reshape(B, 1024, H // 14, W // 14)
        return [features_16.clone()], [(H // 14, W // 14)]  # clone from inference mode to use in autograd


class VGG_DINOv2(nn.Module):
    """Implement a hybrid encoder combining VGG and DINOv2 features."""

    def __init__(self, vgg_kwargs=None, dinov2_kwargs=None):
        if (vgg_kwargs is None) or (dinov2_kwargs is None):
            raise ValueError("Input kwargs please")
        super().__init__()
        self.vgg = VGG19(**vgg_kwargs)
        self.frozen_dinov2 = FrozenDINOv2(**dinov2_kwargs)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """Run this DeDoDe module forward.

        Args:
            x: Input image tensor with shape ``(B, 3, H, W)``.

        Returns:
            Tuple of combined ``(features, sizes)`` lists from the VGG19 and DINOv2 encoders.
        """
        feats_vgg, sizes_vgg = self.vgg(x)
        feat_dinov2, size_dinov2 = self.frozen_dinov2(x)
        return feats_vgg + feat_dinov2, sizes_vgg + size_dinov2
