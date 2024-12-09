# EfficientViT: Multi-Scale Linear Attention for High-Resolution Dense Prediction
# Han Cai, Junyan Li, Muyan Hu, Chuang Gan, Song Han
# International Conference on Computer Vision (ICCV), 2023
from __future__ import annotations

from typing import Any, Optional

import torch
from torch import nn

from kornia.contrib.models.efficient_vit.utils import build_kwargs_from_config


class LayerNorm2d(nn.LayerNorm):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x - torch.mean(x, dim=1, keepdim=True)
        out = out / torch.sqrt(torch.square(out).mean(dim=1, keepdim=True) + self.eps)
        if self.elementwise_affine:
            out = out * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)
        return out


# register normalization function here
REGISTERED_NORM_DICT: dict[str, type] = {"bn2d": nn.BatchNorm2d, "ln": nn.LayerNorm, "ln2d": LayerNorm2d}


def build_norm(name: str = "bn2d", num_features: Optional[int] = None, **kwargs: Any) -> Optional[nn.Module]:
    if name in ["ln", "ln2d"]:
        kwargs["normalized_shape"] = num_features
    else:
        kwargs["num_features"] = num_features
    if name in REGISTERED_NORM_DICT:
        norm_cls = REGISTERED_NORM_DICT[name]
        args = build_kwargs_from_config(kwargs, norm_cls)
        return norm_cls(**args)
    else:
        return None
