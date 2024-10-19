from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional, Union

import torch

from kornia.core import Tensor
from kornia.core.check import KORNIA_CHECK_SHAPE
from kornia.core.external import PILImage as Image
from kornia.core.external import onnx
from kornia.models.base import ModelBase
from kornia.utils.draw import draw_rectangle

__all__ = [
    "BoundingBoxDataFormat",
    "BoundingBox",
    "results_from_detections",
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


class ObjectDetector(ModelBase):
    """This class wraps an object detection model and performs pre-processing and post-processing."""

    name: str = "detection"

    @torch.inference_mode()
    def forward(self, images: Union[Tensor, list[Tensor]]) -> Union[Tensor, list[Tensor]]:
        """Detect objects in a given list of images.

        Args:
            images: If list of RGB images. Each image is a Tensor with shape :math:`(3, H, W)`.
                If Tensor, a Tensor with shape :math:`(B, 3, H, W)`.

        Returns:
            list of detections found in each image. For item in a batch, shape is :math:`(D, 6)`, where :math:`D` is the
            number of detections in the given image, :math:`6` represents class id, score, and `xywh` bounding box.
        """
        images, images_sizes = self.pre_processor(images)
        logits, boxes = self.model(images)
        detections = self.post_processor(logits, boxes, images_sizes)
        return detections

    def visualize(
        self, images: Union[Tensor, list[Tensor]], detections: Optional[Tensor] = None, output_type: str = "torch"
    ) -> Union[Tensor, list[Tensor], list[Image.Image]]:  # type: ignore
        """Very simple drawing.

        Needs to be more fancy later.
        """
        dets = detections or self.forward(images)
        output = []
        for image, detection in zip(images, dets):
            out_img = image[None].clone()
            for out in detection:
                out_img = draw_rectangle(
                    out_img,
                    torch.Tensor([[[out[-4], out[-3], out[-4] + out[-2], out[-3] + out[-1]]]]),
                )
            output.append(out_img[0])

        return self._tensor_to_type(output, output_type, is_batch=isinstance(images, Tensor))

    def save(
        self, images: Union[Tensor, list[Tensor]], detections: Optional[Tensor] = None, directory: Optional[str] = None
    ) -> None:
        """Saves the output image(s) to a directory.

        Args:
            images: input tensor.
            detections: detection tensor.
            directory: directory to save the images.
        """
        outputs = self.visualize(images, detections)
        self._save_outputs(outputs, directory)

    def to_onnx(  # type: ignore[override]
        self,
        onnx_name: Optional[str] = None,
        image_size: Optional[int] = 640,
        include_pre_and_post_processor: bool = True,
        save: bool = True,
        additional_metadata: list[tuple[str, str]] = [],
        **kwargs: Any,
    ) -> onnx.ModelProto:  # type: ignore
        """Exports an RT-DETR object detection model to ONNX format.

        Either `model_name` or `config` must be provided. If neither is provided,
        a default pretrained model (`rtdetr_r18vd`) will be built.

        Args:
            onnx_name:
                The name of the output ONNX file. If not provided, a default name in the
                format "Kornia-<ClassName>.onnx" will be used.
            image_size:
                The size to which input images will be resized during preprocessing.
                If None, image_size will be dynamic.
                For RTDETR, recommended scales include [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800].
            include_pre_and_post_processor:
                Whether to include the pre-processor and post-processor in the exported model.
            save:
                If to save the model or load it.
            additional_metadata:
                Additional metadata to add to the ONNX model.
        """
        if onnx_name is None:
            onnx_name = f"kornia_{self.name}_{image_size}.onnx"

        return super().to_onnx(
            onnx_name,
            input_shape=[-1, 3, image_size or -1, image_size or -1],
            output_shape=[-1, -1, 6],
            pseudo_shape=[1, 3, image_size or 352, image_size or 352],
            model=self if include_pre_and_post_processor else self.model,
            save=save,
            additional_metadata=additional_metadata,
            **kwargs,
        )

    def compile(
        self,
        *,
        fullgraph: bool = False,
        dynamic: bool = False,
        backend: str = "inductor",
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
