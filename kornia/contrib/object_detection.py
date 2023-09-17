from __future__ import annotations

from typing import Any

import torch

from kornia.core import Module, Tensor, concatenate


class ResizePreProcessor(Module):
    """This module resizes a list of image tensors to the given size, and also returns the original image sizes for
    further post-processing."""

    def __init__(self, size: tuple[int, int], interpolation_mode: str = "bilinear") -> None:
        """
        Args:
            size: images will be resized to this value. If a 2-integer tuple is given, it is interpreted as
                (height, width).
            interpolation_mode: interpolation mode for image resizing. Supported values: ``nearest``, ``bilinear``,
                ``bicubic``, ``area``, and ``nearest-exact``.
        """
        super().__init__()
        self.size = size
        self.interpolation_mode = interpolation_mode

    def forward(self, imgs: list[Tensor]) -> tuple[Tensor, dict[str, Any]]:
        # TODO: support other input formats e.g. file path, numpy
        resized_imgs, original_sizes = [], []
        for i in range(len(imgs)):
            img = imgs[i]
            original_sizes.append((img.shape[1], img.shape[2]))
            resized_imgs.append(
                torch.nn.functional.interpolate(img.unsqueeze(0), size=self.size, mode=self.interpolation_mode)
            )
        return concatenate(resized_imgs), {"original_size": original_sizes}


class ObjectDetector:
    """This class wraps an object detection model and performs pre-processing and post-processing."""

    def __init__(self, model: Module, pre_processor: Module, post_processor: Module) -> None:
        """Construct an Object Detector object.

        Args:
            model: an object detection model.
            pre_processor: a pre-processing module
            post_processor: a post-processing module.
        """
        super().__init__()
        self.model = model.eval()
        self.pre_processor = pre_processor.eval()
        self.post_processor = post_processor.eval()

    @torch.inference_mode()
    def predict(self, imgs: list[Tensor]) -> list[Tensor]:
        """Detect objects in a given list of images.

        Args:
            imgs: list of RGB images. Each image is a Tensor with shape :math:`(3, H, W)`.

        Returns:
            list of detections found in each image. For item in a batch, shape is :math:`(D, 6)`, where :math:`D` is the
            number of detections in the given image, :math:`6` represents class id, score, and `xywh` bounding box.
        """
        imgs, meta = self.pre_processor(imgs)
        out = self.model(imgs)
        detections = self.post_processor(out, meta)
        return detections

    def compile(
        self,
        *,
        fullgraph: bool = False,
        dynamic: bool = False,
        backend: str = 'inductor',
        mode: str | None = None,
        options: dict[str, str | int | bool] | None = None,
        disable: bool = False,
    ) -> None:
        """Compile the internal object detection model with :py:func:`torch.compile()`."""
        self.model = torch.compile(  # type: ignore
            self.model,
            fullgraph=fullgraph,
            dynamic=dynamic,
            backend=backend,
            mode=mode,
            options=options,
            disable=disable,
        )
