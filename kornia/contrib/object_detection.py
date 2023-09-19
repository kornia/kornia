from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import torch

from kornia.core import Module, Tensor, concatenate
from kornia.core.check import KORNIA_CHECK_SHAPE
from kornia.image.base import ImageSize

__all__ = [
    "BoundingBoxDataFormat",
    "BoundingBox",
    "ObjectDetectionResult",
    "detection_to_result",
    "ResizePreProcessor",
    "ObjectDetector",
]


class BoundingBoxDataFormat(Enum):
    """Enum class that maps bounding box data format."""

    XYWH = 0
    XYXY = 1
    CXCYCWH = 2


@dataclass(frozen=True)
class BoundingBox:
    """Bounding box.

    Args:
        x: x coordinate of the top-left corner.
        y: y coordinate of the top-left corner.
        width: width of the bounding box.
        height: height of the bounding box.
    """

    data: tuple[float, float, float, float]
    data_format: BoundingBoxDataFormat
    is_normalized: bool


@dataclass(frozen=True)
class ObjectDetectionResult:
    """Object detection result.

    Args:
        class_id: class id of the detected object.
        score: confidence score of the detection.
        bbox: bounding box of the detected object in xywh format.
    """

    class_id: int
    score: float
    bbox: BoundingBox


def detection_to_result(detections: Tensor) -> list[ObjectDetectionResult]:
    """Convert a detection tensor to a list of :py:class:`ObjectDetectionResult`.

    Args:
        detection: tensor with shape :math:`(D, 6)`, where :math:`D` is the number of detections in the given image,
            :math:`6` represents class id, score, and `xywh` bounding box.

    Returns:
        list of :py:class:`ObjectDetectionResult`.
    """
    KORNIA_CHECK_SHAPE(detections, ["D", 6])

    results: list[ObjectDetectionResult] = []
    for det in detections:
        det = det.squeeze().tolist()
        if len(det) != 6:
            continue
        results.append(
            ObjectDetectionResult(
                class_id=int(det[0]),
                score=det[1],
                bbox=BoundingBox(
                    data=(det[2], det[3], det[4], det[5]), data_format=BoundingBoxDataFormat.XYWH, is_normalized=False
                ),
            )
        )
    return results


class ResizePreProcessor(Module):
    """This module resizes a list of image tensors to the given size.

    Additionally, also returns the original image sizes for further post-processing.
    """

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

    def forward(self, imgs: list[Tensor]) -> tuple[Tensor, list[ImageSize]]:
        # TODO: support other input formats e.g. file path, numpy
        resized_imgs, original_sizes = [], []
        for i in range(len(imgs)):
            img = imgs[i]
            # NOTE: assume that image layout is CHW
            original_sizes.append(ImageSize(height=img.shape[1], width=img.shape[2]))
            resized_imgs.append(
                # TODO: fix kornia resize to support onnx
                torch.nn.functional.interpolate(img.unsqueeze(0), size=self.size, mode=self.interpolation_mode)
            )
        return concatenate(resized_imgs), original_sizes


class ObjectDetector(Module):
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
    def forward(self, images: list[Tensor]) -> list[Tensor]:
        """Detect objects in a given list of images.

        Args:
            images: list of RGB images. Each image is a Tensor with shape :math:`(3, H, W)`.

        Returns:
            list of detections found in each image. For item in a batch, shape is :math:`(D, 6)`, where :math:`D` is the
            number of detections in the given image, :math:`6` represents class id, score, and `xywh` bounding box.
        """
        images, images_sizes = self.pre_processor(images)
        logits, boxes = self.model(images)
        detections = self.post_processor(logits, boxes, images_sizes)
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
