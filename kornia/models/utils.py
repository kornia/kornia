import warnings
from typing import Union, List, Tuple

import torch
from torch import Tensor

from kornia.core import Module, concatenate
from kornia.geometry.transform import resize


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
        resized_imgs: Union[Tensor, list[Tensor]] = []

        if not torch.onnx.is_in_onnx_export():
            iters = len(imgs) if isinstance(imgs, list) else imgs.shape[0]
            for i in range(iters):
                img = imgs[i]
                size = original_sizes[i]
                resized_imgs.append(
                    resize(img[None], size=size.cpu().long().numpy().tolist(), interpolation=self.interpolation_mode)
                )
        else:
            warnings.warn(
                "ResizePostProcessor is not supported in ONNX export. "
                "The output will not be resized back to the original size."
            )
            resized_imgs = imgs

        return resized_imgs
