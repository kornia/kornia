from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from kornia.core import Module, Tensor


class ConvNormAct(nn.Sequential):
    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, act: str = "relu", groups: int = 1
    ) -> None:
        super().__init__()
        if kernel_size % 2 == 0:
            # even kernel_size -> asymmetric padding
            # PPHGNetV2 (for RT-DETR) uses kernel 2
            # follow TensorFlow/PaddlePaddle: bottom/right side is padded 1 more than top/left
            # NOTE: this does not account for stride=2
            p1 = (kernel_size - 1) // 2
            p2 = kernel_size - 1 - p1
            self.pad = nn.ZeroPad2d((p1, p2, p1, p2))
            padding = 0
        else:
            padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, 1, groups, False)
        self.norm = nn.BatchNorm2d(out_channels)
        self.act = {"relu": nn.ReLU, "silu": nn.SiLU, "none": nn.Identity}[act](inplace=True)


# Lightly adapted from
# https://github.com/facebookresearch/MaskFormer/blob/main/mask_former/modeling/transformer/transformer_predictor.py
class MLP(Module):
    def __init__(
        self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int, sigmoid_output: bool = False
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim, *h], [*h, output_dim]))
        self.sigmoid_output = sigmoid_output

    def forward(self, x: Tensor) -> Tensor:
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        if self.sigmoid_output:
            x = F.sigmoid(x)
        return x


# Adapted from timm
# https://github.com/huggingface/pytorch-image-models/blob/v0.9.2/timm/layers/drop.py#L137
class DropPath(Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob: float = 0.0, scale_by_keep: bool = True) -> None:
        super().__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x: Tensor) -> Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        if keep_prob > 0.0 and self.scale_by_keep:
            random_tensor.div_(keep_prob)
        return x * random_tensor


# From https://github.com/facebookresearch/detectron2/blob/main/detectron2/layers/batch_norm.py
# Itself from https://github.com/facebookresearch/ConvNeXt/blob/d1fa8f6fef0a165b27399986cc2bdacc92777e40/models/convnext.py#L119  # noqa
class LayerNorm2d(Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / (s + self.eps).sqrt()
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x
