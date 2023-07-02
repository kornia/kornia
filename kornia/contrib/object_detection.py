from __future__ import annotations

import torch
import torch.nn.functional as F

from kornia.core import Module, Tensor, concatenate


class ObjectDetector:
    """This class wraps an object detection model and performs pre-processing and post-processing."""

    def __init__(
        self,
        model: Module,
        post_processor: Module,
        eval_size: int | tuple[int, int],
        interpolation_mode: str = "bilinear",
    ) -> None:
        """Construct an Object Detector object.

        Args:
            model: an object detection model.
            post_processor: a post-processor module.
            eval_size: images will be resized to this value. If a 2-integer tuple is given, it is interpreted as
                (height, width). If an integer is given, images will be resized to a square.
            interpolation_mode: interpolation mode for image resizing. Supported values: ``nearest``, ``bilinear``,
                ``bicubic``, ``area``, and ``nearest-exact``.
        """
        super().__init__()
        self.model = model
        self.post_processor = post_processor
        self.eval_size = eval_size
        self.interpolation_mode = interpolation_mode

    def predict(self, imgs: list[Tensor]) -> list[Tensor]:
        """Detect objects in a given list of images.

        Args:
            imgs: list of RGB images. Each image is a Tensor with shape :math:`(3, H, W)`.

        Returns:
            list of detections found in each image. For item in a batch, shape is :math:`(D, 6)`, where :math:`D` is the
            number of detections in the given image, :math:`6` represents class id, score, and `xywh` bounding box.
        """
        # TODO: support other input formats e.g. file path, numpy
        # NOTE: antialias=False is used in F.interpolate()
        original_sizes = [(img.shape[1], img.shape[2]) for img in imgs]
        resized_imgs = [F.interpolate(img.unsqueeze(0), self.eval_size, mode=self.interpolation_mode) for img in imgs]

        out = self.model(concatenate(resized_imgs))
        detections = self.post_processor(out, original_sizes)
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
