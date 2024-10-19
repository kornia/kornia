import warnings
from typing import List, Tuple, Union

import torch
from torch import Tensor

from kornia.core import Module, concatenate
from kornia.geometry.transform import resize

__all__ = ["ResizePreProcessor", "ResizePostProcessor", "OutputRangePostProcessor"]


class ResizePreProcessor(Module):
    """This module resizes a list of image tensors to the given size.

    Additionally, also returns the original image sizes for further post-processing.
    """

    def __init__(self, height: int, width: int, interpolation_mode: str = "bilinear") -> None:
        """
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
        """
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
        """
        Returns:
            resized_imgs: resized images in a batch.
            original_sizes: the original image sizes of (height, width).
        """
        # TODO: support other input formats e.g. file path, numpy
        resized_imgs: list[Tensor] = []

        if torch.onnx.is_in_onnx_export():
            warnings.warn(
                "ResizePostProcessor is not supported in ONNX export. "
                "The output will not be resized back to the original size."
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
