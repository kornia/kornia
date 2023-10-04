from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import torch

from kornia.core import Module, Tensor, concatenate
from kornia.core.check import KORNIA_CHECK_SHAPE
from kornia.image.base import ImageSize

__all__ = [
    "BoundingBoxDataFormat",
    "BoundingBox",
    "results_from_detections",
    "ResizePreProcessor",
    "ObjectDetector",
    "ObjectDetectorResult",
]


class BoundingBoxDataFormat(Enum):
    """Enum class that maps bounding box data format."""

    XYWH = 0
    XYXY = 1
    CXCYWH = 2
    CENTER_XYWH = 2


# NOTE: probably we should use a more generic name like BoundingBox2D
# and add a BoundingBox3D class for 3D bounding boxes. Also for serialization
# we should have an explicit class for each format to make it more production ready
# specially to serialize to protobuf and not saturate at a high rates.


@dataclass(frozen=True)
class BoundingBox:
    """Bounding box data class.

    Useful for representing bounding boxes in different formats for object detection.

    Args:
        data: tuple of bounding box data. The length of the tuple depends on the data format.
        data_format: bounding box data format.
    """

    data: tuple[float, float, float, float]
    data_format: BoundingBoxDataFormat


@dataclass(frozen=True)
class ObjectDetectorResult:
    """Object detection result.

    Args:
        class_id: class id of the detected object.
        confidence: confidence score of the detected object.
        bbox: bounding box of the detected object in xywh format.
    """

    class_id: int
    confidence: float
    bbox: BoundingBox


def results_from_detections(detections: Tensor, format: str | BoundingBoxDataFormat) -> list[ObjectDetectorResult]:
    """Convert a detection tensor to a list of :py:class:`ObjectDetectorResult`.

    Args:
        detections: tensor with shape :math:`(D, 6)`, where :math:`D` is the number of detections in the given image,
            :math:`6` represents class id, score, and `xywh` bounding box.

    Returns:
        list of :py:class:`ObjectDetectorResult`.
    """
    KORNIA_CHECK_SHAPE(detections, ["D", "6"])

    if isinstance(format, str):
        format = BoundingBoxDataFormat[format.upper()]

    results: list[ObjectDetectorResult] = []
    for det in detections:
        det = det.squeeze().tolist()
        if len(det) != 6:
            continue
        results.append(
            ObjectDetectorResult(
                class_id=int(det[0]),
                confidence=det[1],
                bbox=BoundingBox(data=(det[2], det[3], det[4], det[5]), data_format=format),
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


# TODO: move this to kornia.models as AlgorithmicModel api
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
        mode: Optional[str] = None,
        options: Optional[dict[str, str | int | bool]] = None,
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
