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

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, ClassVar, List, Optional, Tuple, Union

import torch
from torch import nn

from kornia.core.check import KORNIA_CHECK_SHAPE
from kornia.core.external import PILImage as Image
from kornia.core.external import onnx
from kornia.core.mixin.onnx import ONNXExportMixin
from kornia.models.base import ModelBase
from kornia.models.processors import ResizePreProcessor
from kornia.utils.draw import draw_rectangle

__all__ = [
    "BoundingBox",
    "BoundingBoxDataFormat",
    "BoxFiltering",
    "ObjectDetector",
    "ObjectDetectorResult",
    "RTDETRDetectorBuilder",
    "results_from_detections",
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


def results_from_detections(
    detections: torch.Tensor, format: str | BoundingBoxDataFormat
) -> list[ObjectDetectorResult]:
    """Convert a detection torch.tensor to a list of :py:class:`ObjectDetectorResult`.

    Args:
        detections: torch.tensor with shape :math:`(D, 6)`, torch.where :math:`D` is the number of
            detections in the given image,
            :math:`6` represents class id, score, and `xywh` bounding box.
        format: detection format.

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


class ObjectDetector(ModelBase, ONNXExportMixin):
    """Wrap an object detection model and perform pre-processing and post-processing."""

    name: str = "detection"

    def __init__(self, model: nn.Module, pre_processor: nn.Module, post_processor: nn.Module) -> None:
        """Initialize ObjectDetector.

        Args:
            model: The object detection model.
            pre_processor: Pre-processing module (e.g., ResizePreProcessor).
            post_processor: Post-processing module (e.g., DETRPostProcessor).
        """
        super().__init__()
        self.model = model.eval()
        self.pre_processor = pre_processor
        self.post_processor = post_processor

    @staticmethod
    def from_config(config: Any) -> ObjectDetector:
        """Build ObjectDetector from config.

        This is a placeholder to satisfy the abstract method requirement.
        Use kornia.contrib.object_detection.RTDETRDetectorBuilder.build() or instantiate ObjectDetector directly.

        Args:
            config: Configuration object (not used, kept for interface compatibility).

        Returns:
            ObjectDetector instance.

        """
        raise NotImplementedError(
            "ObjectDetector.from_config() is not implemented. "
            "Use kornia.contrib.object_detection.RtdetrBuilder.build() or instantiate ObjectDetector directly."
        )

    @torch.inference_mode()
    def forward(self, images: Union[torch.Tensor, list[torch.Tensor]]) -> Union[torch.Tensor, list[torch.Tensor]]:
        """Detect objects in a given list of images.

        Args:
            images: If list of RGB images. Each image is a torch.Tensor with shape :math:`(3, H, W)`.
                If torch.Tensor, a torch.Tensor with shape :math:`(B, 3, H, W)`.

        Returns:
            list of detections found in each image. For item in a batch, shape is :math:`(D, 6)`,
            torch.where :math:`D` is the
            number of detections in the given image, :math:`6` represents class id, score, and `xywh` bounding box.

        """
        images, images_sizes = self.pre_processor(images)
        logits, boxes = self.model(images)
        detections = self.post_processor(logits, boxes, images_sizes)
        return detections

    def visualize(
        self,
        images: Union[torch.Tensor, list[torch.Tensor]],
        detections: Optional[torch.Tensor] = None,
        output_type: str = "torch",
    ) -> Union[torch.Tensor, list[torch.Tensor], list[Image.Image]]:  # type: ignore
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

        return self._tensor_to_type(output, output_type, is_batch=isinstance(images, torch.Tensor))

    def save(
        self,
        images: Union[torch.Tensor, list[torch.Tensor]],
        detections: Optional[torch.Tensor] = None,
        directory: Optional[str] = None,
    ) -> None:
        """Save the output image(s) to a directory.

        Args:
            images: input torch.tensor.
            detections: detection torch.tensor.
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
        additional_metadata: Optional[list[tuple[str, str]]] = None,
        **kwargs: Any,
    ) -> onnx.ModelProto:  # type: ignore
        """Export an RT-DETR object detection model to ONNX format.

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
            kwargs: Additional arguments to convert to onnx.

        """
        if onnx_name is None:
            onnx_name = f"kornia_{self.name}_{image_size}.onnx"

        return ONNXExportMixin.to_onnx(
            self,
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


class BoxFiltering(nn.Module, ONNXExportMixin):
    """Filter boxes according to the desired threshold.

    Args:
        confidence_threshold: an 0-d scalar that represents the desired threshold.
        classes_to_keep: a 1-d list of classes to keep. If None, keep all classes.
        filter_as_zero: whether to filter boxes as zero.

    """

    ONNX_DEFAULT_INPUTSHAPE: ClassVar[List[int]] = [-1, -1, 6]
    ONNX_DEFAULT_OUTPUTSHAPE: ClassVar[List[int]] = [-1, -1, 6]
    ONNX_EXPORT_PSEUDO_SHAPE: ClassVar[List[int]] = [5, 20, 6]

    def __init__(
        self,
        confidence_threshold: Optional[Union[torch.Tensor, float]] = None,
        classes_to_keep: Optional[Union[torch.Tensor, List[int]]] = None,
        filter_as_zero: bool = False,
    ) -> None:
        super().__init__()
        self.filter_as_zero = filter_as_zero
        self.classes_to_keep = None
        self.confidence_threshold = None
        if classes_to_keep is not None:
            self.classes_to_keep = (
                classes_to_keep if isinstance(classes_to_keep, torch.Tensor) else torch.tensor(classes_to_keep)
            )
        if confidence_threshold is not None:
            self.confidence_threshold = (
                confidence_threshold or confidence_threshold
                if isinstance(confidence_threshold, torch.Tensor)
                else torch.tensor(confidence_threshold)
            )

    def forward(
        self,
        boxes: torch.Tensor,
        confidence_threshold: Optional[torch.Tensor] = None,
        classes_to_keep: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        """Filter boxes according to the desired threshold.

        To be ONNX-friendly, the inputs for direct forwarding need to be all tensors.

        Args:
            boxes: [B, D, 6], torch.where B is the batchsize,  D is the number of detections in the image,
                6 represent (class_id, confidence_score, x, y, w, h).
            confidence_threshold: an 0-d scalar that represents the desired threshold.
            classes_to_keep: a 1-d torch.tensor of classes to keep. If None, keep all classes.

        Returns:
            Union[torch.Tensor, List[torch.Tensor]]
                If `filter_as_zero` is True, return a torch.tensor of shape [D, 6], torch.where D is the total number of
                detections as input.
                If `filter_as_zero` is False, return a list of tensors of shape [D, 6], torch.where D is the number of
                valid detections for each element in the batch.

        """
        # Apply confidence filtering
        zero_tensor = torch.tensor(0.0, device=boxes.device, dtype=boxes.dtype)
        confidence_threshold = (
            confidence_threshold or self.confidence_threshold or zero_tensor
        )  # If None, use 0 as threshold
        confidence_mask = boxes[:, :, 1] > confidence_threshold  # [B, D]

        # Apply class filtering
        classes_to_keep = classes_to_keep or self.classes_to_keep
        if classes_to_keep is not None:
            class_ids = boxes[:, :, 0:1]  # [B, D, 1]
            classes_to_keep = classes_to_keep.view(1, 1, -1)  # [1, 1, C] for broadcasting
            class_mask = (class_ids == classes_to_keep).any(dim=-1)  # [B, D]
        else:
            # If no class filtering is needed, just use a mask of all `True`
            class_mask = (confidence_mask * 0 + 1).bool()

        # Combine the confidence and class masks
        combined_mask = confidence_mask & class_mask  # [B, D]

        if self.filter_as_zero:
            filtered_boxes = boxes * combined_mask[:, :, None]
            return filtered_boxes

        filtered_boxes_list = []
        for i in range(boxes.shape[0]):
            box = boxes[i]
            mask = combined_mask[i]  # [D]
            valid_boxes = box[mask]
            filtered_boxes_list.append(valid_boxes)

        return filtered_boxes_list

    def _create_dummy_input(
        self, input_shape: List[int], pseudo_shape: Optional[List[int]] = None
    ) -> Union[Tuple[Any, ...], torch.Tensor]:
        pseudo_input = torch.rand(
            *[
                ((self.ONNX_EXPORT_PSEUDO_SHAPE[i] if pseudo_shape is None else pseudo_shape[i]) if dim == -1 else dim)
                for i, dim in enumerate(input_shape)
            ]
        )
        if self.confidence_threshold is None:
            return pseudo_input, 0.1
        return pseudo_input


class RTDETRDetectorBuilder:
    """A builder class for constructing RT-DETR object detection models.

    This class provides static methods to:
        - Build an object detection model from a model name or configuration.
        - Export the model to ONNX format for inference.

    .. code-block:: python

        images = kornia.utils.sample.get_sample_images()
        model = RTDETRDetectorBuilder.build()
        model.save(images)
    """

    @staticmethod
    def build(
        model_name: Optional[str] = None,
        config: Optional[Any] = None,
        pretrained: bool = True,
        image_size: Optional[int] = None,
        confidence_threshold: Optional[float] = None,
        confidence_filtering: Optional[bool] = None,
    ) -> ObjectDetector:
        """Build and returns an RT-DETR object detector model.

        Either `model_name` or `config` must be provided. If neither is provided,
        a default pretrained model (`rtdetr_r18vd`) will be built.

        Args:
            model_name:
                Name of the RT-DETR model to load. Can be one of the available pretrained models.
                Including 'rtdetr_r18vd', 'rtdetr_r34vd', 'rtdetr_r50vd_m', 'rtdetr_r50vd', 'rtdetr_r101vd'.
            config:
                A custom configuration object for building the RT-DETR model.
            pretrained:
                Whether to load a pretrained version of the model (applies when `model_name` is provided).
            image_size:
                The size to which input images will be resized during preprocessing.
                If None, no resizing will be inferred from config file. Recommended scales include
                [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800].
            confidence_threshold: Threshold to filter results based on confidence scores.
            confidence_filtering: Whether to filter results based on confidence scores.


        Returns:
            ObjectDetector
                An object detector instance initialized with the specified model, preprocessor, and post-processor.

        """
        import warnings

        from kornia.models.rt_detr import DETRPostProcessor
        from kornia.models.rt_detr.model import RTDETR, RTDETRConfig

        if model_name is not None and config is not None:
            raise ValueError("Either `model_name` or `config` should be `None`.")

        if config is not None:
            model = RTDETR.from_config(config)
            image_size = image_size or config.input_size
        elif model_name is not None:
            if pretrained:
                model = RTDETR.from_pretrained(model_name)
                image_size = RTDETRConfig.from_name(model_name).input_size
            else:
                model = RTDETR.from_name(model_name)
                image_size = RTDETRConfig.from_name(model_name).input_size
        else:
            warnings.warn("No `model_name` or `config` found. Will build pretrained `rtdetr_r18vd`.", stacklevel=1)
            model = RTDETR.from_pretrained("rtdetr_r18vd")
            image_size = RTDETRConfig.from_name("rtdetr_r18vd").input_size

        if confidence_threshold is None:
            confidence_threshold = config.confidence_threshold if config is not None else 0.3

        return ObjectDetector(
            model,
            ResizePreProcessor(image_size, image_size),
            DETRPostProcessor(
                confidence_threshold=confidence_threshold,
                confidence_filtering=confidence_filtering or not torch.onnx.is_in_onnx_export(),
                num_classes=model.decoder.num_classes,
                num_top_queries=model.decoder.num_queries,
            ),
        )
