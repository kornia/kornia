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

import warnings
from typing import List, Tuple, Union

import torch
from torch import Tensor

from kornia.core import Module, concatenate
from kornia.geometry.transform import resize

__all__ = ["OutputRangePostProcessor", "ResizePostProcessor", "ResizePreProcessor"]


class ResizePreProcessor(Module):
    """Resize a list of image tensors to the given size.

    Additionally, also returns the original image sizes for further post-processing.
    """

    def __init__(self, height: int, width: int, interpolation_mode: str = "bilinear") -> None:
        """Construct ResizePreprocessor module.

        Args:
        height: height of the resized image.
        width: width of the resized image.
        interpolation_mode: interpolation mode for image resizing. Supported values: ``nearest``, ``bilinear``,
            ``bicubic``, ``area``, and ``nearest-exact``.

        """
        super().__init__()
        self.size = (height, width)
        self.interpolation_mode = interpolation_mode

    def forward(self, imgs: Union[Tensor, List[Tensor]]) -> Tuple[Tensor, Tensor]:
        """Run forward.

        Returns:
        resized_imgs: resized images in a batch.
        original_sizes: the original image sizes of (height, width).

        """
        # TODO: support other input formats e.g. file path, numpy

        # Optimize: support batch tensors, or list of tensors of equal shape
        if isinstance(imgs, Tensor):
            iters = imgs.shape[0]
            original_sizes = imgs.new_zeros((iters, 2))
            # Get shape info up front (batch, ...) for last two dims
            # Vectorized version:
            original_sizes[:, 0] = imgs.shape[-2]
            original_sizes[:, 1] = imgs.shape[-1]
            resized_imgs = resize(imgs, size=self.size, interpolation=self.interpolation_mode)
            return resized_imgs, original_sizes

        else:  # handle list[Tensor]
            iters = len(imgs)
            # preallocate
            original_sizes = imgs[0].new_zeros((iters, 2))
            resized_imgs: List[Tensor] = []
            # Optimize: minimize repeated properties/method calls
            size = self.size
            interp = self.interpolation_mode
            for i, img in enumerate(imgs):
                h, w = img.shape[-2], img.shape[-1]
                original_sizes[i, 0] = h
                original_sizes[i, 1] = w
                # Call resize as batched for each image to avoid shape mismatch
                resized_imgs.append(resize(img[None], size=size, interpolation=interp))
            # Concatenate on batch dim
            out = concatenate(resized_imgs)
            return out, original_sizes


class ResizePostProcessor(Module):
    def __init__(self, interpolation_mode: str = "bilinear") -> None:
        super().__init__()
        self.interpolation_mode = interpolation_mode

    def forward(self, imgs: Union[Tensor, List[Tensor]], original_sizes: Tensor) -> Union[Tensor, List[Tensor]]:
        """Run forward.

        Returns:
        resized_imgs: resized images in a batch.
        original_sizes: the original image sizes of (height, width).

        """
        # TODO: support other input formats e.g. file path, numpy
        resized_imgs: list[Tensor] = []

        if torch.onnx.is_in_onnx_export():
            warnings.warn(
                "ResizePostProcessor is not supported in ONNX export. "
                "The output will not be resized back to the original size.",
                stacklevel=1,
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


class OutputRangePostProcessor(Module):
    def __init__(self, min_val: float = 0.0, max_val: float = 1.0) -> None:
        super().__init__()
        self.min_val = min_val
        self.max_val = max_val

    def forward(self, imgs: Union[Tensor, List[Tensor]]) -> Union[Tensor, List[Tensor]]:
        if isinstance(imgs, Tensor):
            return torch.clamp(imgs, self.min_val, self.max_val)
        return [img.clamp_(self.min_val, self.max_val) for img in imgs]
