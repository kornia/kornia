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
        resized_imgs: list[Tensor] = []

        iters = len(imgs) if isinstance(imgs, list) else imgs.shape[0]
        original_sizes = imgs[0].new_zeros((iters, 2))
        for i in range(iters):
            img = imgs[i]
            original_sizes[i, 0] = img.shape[-2]  # Height
            original_sizes[i, 1] = img.shape[-1]  # Width
            resized_imgs.append(resize(img[None], size=self.size, interpolation=self.interpolation_mode))
        return concatenate(resized_imgs), original_sizes


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
