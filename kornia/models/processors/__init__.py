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
from typing import List, Tuple, Union

import torch
from torch import Tensor, nn

from kornia.geometry.transform import resize

from .naflex import NaFlex

__all__ = [
    "NaFlex",
    "OutputRangePostProcessor",
    "ResizePostProcessor",
    "ResizePreProcessor",
]


class ResizePreProcessor(nn.Module):
    r"""Resize a list of image tensors to the given size.

    Additionally, also returns the original image sizes for further post-processing.

    Args:
        height: Height of the resized image.
        width: Width of the resized image.
        interpolation_mode: Interpolation mode for image resizing. Supported values:
            ``nearest``, ``bilinear``, ``bicubic``, ``area``, ``nearest-exact``.
            Default: "bilinear".

    Example:
        >>> import torch
        >>> from kornia.models.processors import ResizePreProcessor
        >>> processor = ResizePreProcessor(height=224, width=224)
        >>> imgs = torch.randn(2, 3, 480, 640)
        >>> resized, sizes = processor(imgs)
        >>> print(resized.shape, sizes.shape)
        torch.Size([2, 3, 224, 224]) torch.Size([2, 2])
    """

    def __init__(self, height: int, width: int, interpolation_mode: str = "bilinear") -> None:
        super().__init__()
        self.size = (height, width)
        self.interpolation_mode = interpolation_mode

    def forward(self, imgs: Union[Tensor, List[Tensor]]) -> Tuple[Tensor, Tensor]:
        r"""Resize input images to the target size.

        Args:
            imgs: Input images, either a tensor of shape :math:`(B, C, H, W)` or
                a list of tensors of shape :math:`(C, H, W)`.

        Returns:
            Tuple containing:
                - resized_imgs: Resized images as a tensor of shape :math:`(B, C, H_{\text{new}}, W_{\text{new}})`.
                - original_sizes: Original image sizes of shape :math:`(B, 2)` containing (height, width).
        """
        resized_imgs: List[Tensor] = []

        iters = len(imgs) if isinstance(imgs, list) else imgs.shape[0]
        original_sizes = imgs[0].new_zeros((iters, 2))

        for i in range(iters):
            img = imgs[i]
            original_sizes[i, 0] = img.shape[-2]  # Height
            original_sizes[i, 1] = img.shape[-1]  # Width
            resized_imgs.append(resize(img[None], size=self.size, interpolation=self.interpolation_mode))

        return torch.cat(resized_imgs), original_sizes


class ResizePostProcessor(nn.Module):
    r"""Rescale model outputs back to the original image dimensions.

    Args:
        interpolation_mode: The algorithm used for upsampling. Supported values:
            ``nearest``, ``bilinear``, ``bicubic``, ``area``, ``nearest-exact``.
            Default: "bilinear".

    Example:
        >>> import torch
        >>> from kornia.models.processors import ResizePostProcessor
        >>> processor = ResizePostProcessor()
        >>> imgs = torch.randn(2, 3, 224, 224)
        >>> original_sizes = torch.tensor([[480, 640], [360, 480]])
        >>> out = processor(imgs, original_sizes)
    """

    def __init__(self, interpolation_mode: str = "bilinear") -> None:
        super().__init__()
        self.interpolation_mode = interpolation_mode

    def forward(self, imgs: Union[Tensor, List[Tensor]], original_sizes: Tensor) -> Union[Tensor, List[Tensor]]:
        r"""Resize model outputs back to original dimensions.

        Args:
            imgs: Input images, either a tensor of shape :math:`(B, C, H, W)` or
                a list of tensors of shape :math:`(C, H, W)`.
            original_sizes: Original image sizes of shape :math:`(B, 2)` containing
                (height, width) pairs.

        Returns:
            Resized images in a batch with original dimensions. If ONNX export is active,
            returns inputs unchanged with a warning.

        Warning:
            ResizePostProcessor is not fully supported in ONNX export mode.
        """
        import warnings

        resized_imgs: List[Tensor] = []

        if torch.onnx.is_in_onnx_export():
            warnings.warn(
                "ResizePostProcessor is not supported in ONNX export. "
                "The output will not be resized back to the original size.",
                stacklevel=2,
            )
            return imgs

        iters = len(imgs) if isinstance(imgs, list) else imgs.shape[0]

        for i in range(iters):
            img = imgs[i]
            size = original_sizes[i]
            resized_imgs.append(
                resize(img[None], size=size.cpu().long().numpy().tolist(), interpolation=self.interpolation_mode)
            )

        return resized_imgs


class OutputRangePostProcessor(nn.Module):
    r"""Clip and scale model outputs to a specific numerical range.

    Args:
        min_val: Minimum value for clipping. Default: 0.0.
        max_val: Maximum value for clipping. Default: 1.0.

    Example:
        >>> import torch
        >>> from kornia.models.processors import OutputRangePostProcessor
        >>> processor = OutputRangePostProcessor(min_val=0.0, max_val=1.0)
        >>> x = torch.randn(2, 3, 224, 224)
        >>> out = processor(x)
        >>> print(out.min(), out.max())
        tensor(0.) tensor(1.)
    """

    def __init__(self, min_val: float = 0.0, max_val: float = 1.0) -> None:
        super().__init__()
        self.min_val = min_val
        self.max_val = max_val

    def forward(self, imgs: Union[Tensor, List[Tensor]]) -> Union[Tensor, List[Tensor]]:
        r"""Clip output values to the specified range.

        Args:
            imgs: Input images, either a tensor or list of tensors.

        Returns:
            Clipped images with values in :math:`[\text{min\_val}, \text{max\_val}]`.
        """
        if isinstance(imgs, torch.Tensor):
            return torch.clamp(imgs, self.min_val, self.max_val)
        return [img.clamp(self.min_val, self.max_val) for img in imgs]
