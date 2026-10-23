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
from typing import Any, ClassVar, List, Optional, Union

import torch
from torch import nn

from kornia.constants import _get_by_name
from kornia.core.check import KORNIA_CHECK_SHAPE
from kornia.core.external import PILImage as Image
from kornia.core.external import onnx
from kornia.core.mixin.onnx import ONNXExportMixin, _preserved_rng, _returns_per_image_outputs, _uses_dynamic_shapes
from kornia.image.draw import draw_rectangle
from kornia.models.base import ModelBase
from kornia.models.processors import ResizePreProcessor

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
    """Convert a detection torch.Tensor to a list of :py:class:`ObjectDetectorResult`.

    Args:
        detections: torch.Tensor with shape :math:`(D, 6)`, where :math:`D` is the number of
            detections in the given image,
            :math:`6` represents class id, score, and `xywh` bounding box.
        format: detection format.

    Returns:
        list of :py:class:`ObjectDetectorResult`.

    """
    KORNIA_CHECK_SHAPE(detections, ["D", "6"])

    if isinstance(format, str):
        format = _get_by_name(BoundingBoxDataFormat, format, format)

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
            "Use kornia.contrib.object_detection.RTDETRDetectorBuilder.build() or instantiate ObjectDetector directly."
        )

    @torch.inference_mode()
    def forward(self, images: Union[torch.Tensor, list[torch.Tensor]]) -> Union[torch.Tensor, list[torch.Tensor]]:
        """Detect objects in a given list of images.

        Args:
            images: If list of RGB images. Each image is a torch.Tensor with shape :math:`(3, H, W)`.
                If torch.Tensor, a torch.Tensor with shape :math:`(B, 3, H, W)`.

        Returns:
            list of detections found in each image. For item in a batch, shape is :math:`(D, 6)`,
            where :math:`D` is the
            number of detections in the given image, :math:`6` represents class id, score, and `xywh` bounding box.

        """
        images, images_sizes = self.pre_processor(images)
        logits, boxes = self.model(images)
        return self.post_processor(logits, boxes, images_sizes)

    def visualize(
        self,
        images: Union[torch.Tensor, list[torch.Tensor]],
        detections: Optional[torch.Tensor] = None,
        output_type: str = "torch",
    ) -> Union[torch.Tensor, list[torch.Tensor], list[Image.Image]]:
        """Very simple drawing.

        Needs to be more fancy later.
        """
        dets = self.forward(images) if detections is None else detections
        output = []
        for image, detection in zip(images, dets):
            out_img = image[None].clone()
            for out in detection:
                box = torch.Tensor([[[out[-4], out[-3], out[-4] + out[-2], out[-3] + out[-1]]]])
                # a detection that reaches or crosses the border stays visible on all four sides
                box[..., 0::2] = box[..., 0::2].clamp(0, out_img.shape[-1] - 1)
                box[..., 1::2] = box[..., 1::2].clamp(0, out_img.shape[-2] - 1)
                out_img = draw_rectangle(out_img, box)
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
            images: input torch.Tensor.
            detections: detection torch.Tensor.
            directory: directory to save the images.

        """
        outputs = self.visualize(images, detections)
        self._save_outputs(outputs, directory)

    def to_onnx(
        self,
        onnx_name: Optional[str] = None,
        image_size: Optional[int] = 640,
        include_pre_and_post_processor: bool = True,
        save: bool = True,
        additional_metadata: Optional[list[tuple[str, str]]] = None,
        **kwargs: Any,
    ) -> onnx.ModelProto:
        """Export an RT-DETR object detection model to ONNX format.

        Exports this detector, as built by :meth:`RTDETRDetectorBuilder.build` or the constructor.

        Args:
            onnx_name:
                The name of the output ONNX file. If not provided, ``"kornia_<name>_<image_size>.onnx"``, with
                the detector's ``name``, for example ``"kornia_detection_640.onnx"``
                (``"kornia_detection_None.onnx"`` for ``image_size=None``).
            image_size:
                The height and width of the exported graph's input. If None, both are dynamic. With the pre- and
                post-processor, the pre-processor resizes the input to the size it was built with, whatever
                ``image_size`` is; without them, the model runs at the size of its input.
                For RTDETR, recommended scales include [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800].
            include_pre_and_post_processor:
                Whether to include the pre-processor and post-processor in the exported model.
            save:
                If True, write the model to ``onnx_name``; it is returned either way.
            additional_metadata:
                Additional metadata to add to the ONNX model.
            kwargs: Additional keyword arguments of :func:`torch.onnx.export`.

        Note:
            With a post-processor that returns a list or tuple of per-image detections, as a
            ``kornia.models.rt_detr.DETRPostProcessor`` that filters by confidence does, the model is
            exported for a batch of one image: no ONNX output holds a list as long as the batch. Which kind of output
            the detector returns is read from one forward pass for one image at the export size, in eval mode and
            without gradients, and a tuple as long as the batch from a second one at another batch size, which tells
            per-image results from a fixed set of batched outputs such as ``(boxes, scores)``. A per-image result
            requires a fixed batch; a tensor result is checked again at the actual tracing batch before accepting a
            dynamic graph. Build an RT-DETR detector with ``confidence_filtering=False`` to export a batch of any
            size. A ``dynamic_axes`` or ``dynamic_shapes`` argument that makes the batch of such an export dynamic
            raises a ``ValueError`` after the export, before anything is saved.

            Without the pre- and post-processor, a model that declares ``ONNX_OUTPUT_NAMES``, as
            :class:`~kornia.models.rt_detr.RTDETR` does (``pred_logits`` and ``pred_boxes``), has its outputs exported
            under those names, each with a dynamic batch; any other model has one output, ``output``.

        """
        if onnx_name is None:
            onnx_name = f"kornia_{self.name}_{image_size}.onnx"
        pseudo_shape = [1, 3, image_size or 352, image_size or 352]
        # Exported for a dynamic batch, a per-image sequence comes out as the traced batch's detections, or the export
        # fails: such a detector is exported for one image.
        per_image = None
        if include_pre_and_post_processor:
            # The probe's input draws from the global generator, as the export's dummy does, and gives back what it
            # drew.
            with _preserved_rng():
                per_image = _returns_per_image_outputs(self, torch.rand(pseudo_shape))
        batch = 1 if per_image else -1
        input_shape = [batch, 3, image_size or -1, image_size or -1]

        output_names = None if include_pre_and_post_processor else getattr(self.model, "ONNX_OUTPUT_NAMES", None)
        if output_names:
            # The bare model's outputs by name, each with a dynamic batch: listed in ``dynamic_axes`` for the
            # TorchScript exporter, derived by the dynamo exporter from the input's dynamic dimensions.
            if kwargs.get("output_names") is None:  # absent, or ``torch.onnx.export``'s own default of None
                kwargs["output_names"] = list(output_names)
            if not _uses_dynamic_shapes(kwargs) and "dynamic_axes" not in kwargs and "dynamic_shapes" not in kwargs:
                input_axes = {i: f"dim_{i}" for i, dim in enumerate(input_shape) if dim == -1}
                # Keyed by the names the graph gets, the caller's if given.
                output_axes = {name: {0: "dim_0"} for name in kwargs["output_names"]}
                kwargs["dynamic_axes"] = {"input": input_axes, **output_axes}

        with self._onnx_output_kind_kept(per_image):
            return ONNXExportMixin.to_onnx(
                self,
                onnx_name,
                input_shape=input_shape,
                output_shape=[-1, -1, 6],
                pseudo_shape=pseudo_shape,
                model=self if include_pre_and_post_processor else self.model,
                save=save,
                additional_metadata=additional_metadata,
                **kwargs,
            )

    def _onnx_fixed_batch_hint(self, module: Any, dummy_input: Any) -> Optional[str]:
        if module is not self or not self._onnx_returns_per_image_outputs(dummy_input):
            return None
        return (
            "Use a post-processor that returns one tensor for the batch, such as a DETRPostProcessor with "
            "confidence_filtering=False, to export a dynamic batch."
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
        self.model = torch.compile(
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

    Note:
        The exported graph has one input, ``input``, the boxes. It filters at ``confidence_threshold``, held as a
        constant, and at the eager default of ``0.0`` (keep boxes with a positive confidence) when none was given.

    Note:
        Without ``filter_as_zero``, ONNX export produces one output tensor per image, so the batch must be fixed.
        :meth:`to_onnx` defaults to a batch of one image. Use ``filter_as_zero=True`` to export a dynamic batch
        as one ``(B, D, 6)`` tensor.

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
        self.classes_to_keep = None if classes_to_keep is None else torch.as_tensor(classes_to_keep)
        self.confidence_threshold = None if confidence_threshold is None else torch.as_tensor(confidence_threshold)

    def forward(
        self,
        boxes: torch.Tensor,
        confidence_threshold: Optional[torch.Tensor] = None,
        classes_to_keep: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        """Filter boxes according to the desired threshold.

        To be ONNX-friendly, the inputs for direct forwarding need to be all tensors.

        Args:
            boxes: [B, D, 6], where B is the batchsize,  D is the number of detections in the image,
                6 represent (class_id, confidence_score, x, y, w, h).
            confidence_threshold: an 0-d scalar that represents the desired threshold.
            classes_to_keep: a 1-d torch.Tensor of classes to keep. If None, keep all classes.

        Returns:
            Union[torch.Tensor, List[torch.Tensor]]
                If `filter_as_zero` is True, return a torch.Tensor of shape [D, 6], where D is the total number of
                detections as input.
                If `filter_as_zero` is False, return a list of tensors of shape [D, 6], where D is the number of
                valid detections for each element in the batch.

        """
        # Apply confidence filtering
        # Explicit `None` tests: `x or y` would read a tensor's truth value, which graph capture cannot do.
        if confidence_threshold is None:
            confidence_threshold = self.confidence_threshold
        if confidence_threshold is None:
            confidence_threshold = torch.tensor(0.0, device=boxes.device, dtype=boxes.dtype)  # use 0 as threshold
        confidence_mask = boxes[:, :, 1] > confidence_threshold  # [B, D]

        # Apply class filtering
        if classes_to_keep is None:
            classes_to_keep = self.classes_to_keep
        if classes_to_keep is not None:
            class_ids = boxes[:, :, 0:1]  # [B, D, 1]
            # `.to(device)` leaves the class list on the device it was built on, so follow the boxes. The 0-d
            # threshold needs no move: a 0-d CPU tensor works next to boxes on any device.
            classes_to_keep = classes_to_keep.to(boxes.device).view(1, 1, -1)  # [1, 1, C] for broadcasting
            class_mask = (class_ids == classes_to_keep).any(dim=-1)  # [B, D]
        else:
            # If no class filtering is needed, just use a mask of all `True`
            class_mask = torch.ones_like(confidence_mask)

        # Combine the confidence and class masks
        combined_mask = confidence_mask & class_mask  # [B, D]

        if self.filter_as_zero:
            return boxes * combined_mask[:, :, None]

        filtered_boxes_list = []
        for i in range(boxes.shape[0]):
            box = boxes[i]
            mask = combined_mask[i]  # [D]
            valid_boxes = box[mask]
            filtered_boxes_list.append(valid_boxes)

        return filtered_boxes_list

    def to_onnx(
        self,
        onnx_name: Optional[str] = None,
        input_shape: Optional[list[int]] = None,
        output_shape: Optional[list[int]] = None,
        pseudo_shape: Optional[list[int]] = None,
        model: Optional[nn.Module] = None,
        save: bool = True,
        additional_metadata: Optional[list[tuple[str, str]]] = None,
        **kwargs: Any,
    ) -> onnx.ModelProto:
        """Export the filter with a fixed batch for per-image list outputs.

        Args:
            onnx_name: The output file name. Defaults to ``Kornia-BoxFiltering.onnx``.
            input_shape: The ``[B, D, 6]`` input shape, with ``-1`` for a dynamic dimension. Defaults to
                ``[1, -1, 6]`` without ``filter_as_zero`` and ``[-1, -1, 6]`` with it.
            output_shape: The output shape used by the legacy TorchScript exporter. Defaults to ``[-1, 6]``
                without ``filter_as_zero`` and ``[-1, -1, 6]`` with it. The dynamo exporter infers output shapes.
            pseudo_shape: The dummy input shape used for tracing. Without ``filter_as_zero`` and ``input_shape``,
                its batch is the exported batch. Explicit input and pseudo shapes must agree on fixed dimensions.
            model: The model to export. Defaults to this filter.
            save: Whether to save the exported model to ``onnx_name``.
            additional_metadata: Additional metadata to add to the ONNX model.
            **kwargs: Additional arguments to :func:`torch.onnx.export`.

        Returns:
            The exported ONNX model.

        Raises:
            ValueError: If ``input_shape`` marks the batch ``-1`` without ``filter_as_zero``, or a ``dynamic_axes``
                or ``dynamic_shapes`` argument makes the exported batch dynamic without it.

        Note:
            The exported graph takes the boxes as its only input and filters at the threshold given to the
            constructor, or at ``0.0`` when there is none, as eager does.

        Note:
            The exporters flatten the Python list into a fixed number of graph outputs. Without ``filter_as_zero``,
            the batch is therefore fixed: at the batch of ``input_shape``, else of ``pseudo_shape``, else at 1. ONNX
            Runtime rejects other batch sizes. Each image's number of retained detections remains dynamic.
            Use ``filter_as_zero=True`` for one ``(B, D, 6)`` output with a dynamic batch.

        """
        if not self.filter_as_zero:
            if input_shape is None:
                input_shape = [1 if pseudo_shape is None else pseudo_shape[0], *self.ONNX_DEFAULT_INPUTSHAPE[1:]]
            if input_shape and input_shape[0] == -1:
                raise ValueError(
                    "BoxFiltering returns one ONNX output per image without filter_as_zero and requires a fixed batch. "
                    "Fix the batch in input_shape, or use filter_as_zero=True to export a dynamic batch."
                )
            if output_shape is None:
                output_shape = [-1, 6]
        return super().to_onnx(
            onnx_name=onnx_name,
            input_shape=input_shape,
            output_shape=output_shape,
            pseudo_shape=pseudo_shape,
            model=model,
            save=save,
            additional_metadata=additional_metadata,
            **kwargs,
        )

    def _onnx_fixed_batch_hint(self, module: Any, dummy_input: Any) -> Optional[str]:
        if not isinstance(module, BoxFiltering) or module.filter_as_zero:
            return None
        return "Use filter_as_zero=True to export a dynamic batch as one (B, D, 6) output."


class RTDETRDetectorBuilder:
    """A builder class for constructing RT-DETR object detection models.

    This class provides static methods to:
        - Build an object detection model from a model name or configuration.
        - Export the model to ONNX format for inference.

    Note:
        To use this model, load image tensors and call ``model.save(images)``.
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
                The size to which input images will be resized during preprocessing, used only with `config`. If
                None, `config.input_size` is used. With `model_name`, or without either, it is ignored: the input is
                resized to the 640 px the published weights were trained for. Recommended scales include
                [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800].
            confidence_threshold: Threshold to filter results based on confidence scores.
            confidence_filtering: Whether to filter results based on confidence scores. None filters.


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
                confidence_filtering=True if confidence_filtering is None else confidence_filtering,
                num_classes=model.decoder.num_classes,
                num_top_queries=model.decoder.num_queries,
            ),
        )
