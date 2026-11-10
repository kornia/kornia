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

"""Detect objects in RGB images with a network, then filter the detections or convert them to Python objects.

A detection is a row ``(class_id, score, x, y, w, h)``: the box's top-left corner, width and height in the pixels of
the input image.

Which class to use:

- Objects of the 80 COCO classes, from a pretrained RT-DETR network: :class:`RTDETRDetectorBuilder`.
- Your own detection network with its own pre- and post-processing: :class:`ObjectDetector`.
- Keep the detections above a score and in given classes: :class:`BoxFiltering`.
- The detections of one image as Python objects: :func:`results_from_detections`, which returns
  :class:`ObjectDetectorResult` items holding a :class:`BoundingBox` tagged with a :class:`BoundingBoxDataFormat`.
"""

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
    """Name the layout of the four values of a BoundingBox: xywh, xyxy or center xywh.

    ``BoundingBoxDataFormat.XYWH`` tags a :class:`BoundingBox` whose values are the top-left ``x, y``, the width and
    the height; ``XYXY`` the top-left and bottom-right corners; ``CXCYWH`` the center, the width and the height.
    ``CENTER_XYWH`` is another name of ``CXCYWH``, the same member.

    See Also:
        - :class:`BoundingBox`: the box whose values this names.
        - :class:`~kornia.geometry.boxes.Boxes`: a batch of boxes as tensors, converted between the ``xyxy``,
          ``xywh`` and corner layouts.

    """

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
    """Hold one box as four Python floats tagged with their BoundingBoxDataFormat.

    ``box = BoundingBox(data=(x, y, w, h), data_format=BoundingBoxDataFormat.XYWH)`` is a frozen dataclass: ``data``
    holds the four values and ``data_format`` says how to read them; nothing converts or checks them.
    :func:`results_from_detections` builds one per detection.

    Args:
        data: tuple of bounding box data. The length of the tuple depends on the data format.
        data_format: bounding box data format.

    See Also:
        - :class:`BoundingBoxDataFormat`: the layouts that ``data_format`` names.
        - :class:`ObjectDetectorResult`: one detection, holding this box with its class and confidence.
        - :class:`~kornia.geometry.boxes.Boxes`: a batch of boxes as tensors, with layout conversions.

    """

    data: tuple[float, float, float, float]
    data_format: BoundingBoxDataFormat


@dataclass(frozen=True)
class ObjectDetectorResult:
    """Hold one detection as a class id, a confidence and a BoundingBox.

    ``result = ObjectDetectorResult(class_id, confidence, bbox)`` is a frozen dataclass of Python values.
    :func:`results_from_detections` builds one per row of the ``(D, 6)`` detections of an image.

    Args:
        class_id: class id of the detected object.
        confidence: confidence score of the detected object.
        bbox: bounding box of the detected object, whose ``data_format`` says the layout of its values.

    See Also:
        - :func:`results_from_detections`: builds these from a detection tensor.
        - :class:`BoundingBox`: the box it holds.

    """

    class_id: int
    confidence: float
    bbox: BoundingBox


def results_from_detections(
    detections: torch.Tensor, format: str | BoundingBoxDataFormat
) -> list[ObjectDetectorResult]:
    """Convert the ``(D, 6)`` detections of one image to a list of ObjectDetectorResult.

    ``results = results_from_detections(dets, "xywh")`` takes ``D`` rows ``(class_id, score, b0, b1, b2, b3)`` and
    returns ``D`` :class:`ObjectDetectorResult` items. ``format`` only tags each
    :class:`BoundingBox`: the four box values are copied, not converted, so pass the layout the detections are in,
    ``"xywh"`` for those of :class:`ObjectDetector`.

    Args:
        detections: torch.Tensor with shape ``(D, 6)``, where :math:`D` is the number of
            detections in the given image: the class id, the score and the four box values.
        format: the layout of the box values, a :class:`BoundingBoxDataFormat` or its name in any case.

    Returns:
        list of :class:`ObjectDetectorResult`, one per row, in the order of the rows.

    Raises:
        ShapeError: If ``detections`` is not ``(D, 6)``.
        ValueError: If ``format`` names no :class:`BoundingBoxDataFormat`.

    See Also:
        - :class:`BoxFiltering`: keeps the rows above a score and in given classes, before the conversion.
        - :class:`ObjectDetector`: returns one detection tensor per image in this layout.

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
    """Run a detection network on RGB images and return per-image (class, score, xywh box) rows.

    ``detector = ObjectDetector(model, pre_processor, post_processor)`` wraps a detection network.
    ``dets = detector(images)`` takes a ``(B, 3, H, W)`` tensor or a list of ``(3, H, W)`` images, runs
    ``pre_processor``, ``model`` and ``post_processor`` in turn, and returns what the post-processor returns.

    See Also:
        - :class:`RTDETRDetectorBuilder`: builds this wrapper around a pretrained RT-DETR network.
        - :class:`BoxFiltering`: keeps the detections above a score and in given classes.
        - :func:`results_from_detections`: converts the detections of one image to Python objects.

    """

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
        """Raise ``NotImplementedError``: build with :meth:`RTDETRDetectorBuilder.build` or the constructor.

        Args:
            config: Configuration object (not used, kept for interface compatibility).

        Raises:
            NotImplementedError: Always.

        """
        raise NotImplementedError(
            "ObjectDetector.from_config() is not implemented. "
            "Use kornia.contrib.object_detection.RTDETRDetectorBuilder.build() or instantiate ObjectDetector directly."
        )

    @torch.inference_mode()
    def forward(self, images: Union[torch.Tensor, list[torch.Tensor]]) -> Union[torch.Tensor, list[torch.Tensor]]:
        """Detect objects in a given list of images.

        Args:
            images: If list of RGB images. Each image is a torch.Tensor with shape ``(3, H, W)``.
                If torch.Tensor, a torch.Tensor with shape ``(B, 3, H, W)``.

        Returns:
            What the post-processor returns. For a detector from :meth:`RTDETRDetectorBuilder.build`, a list with
            one ``(D, 6)`` tensor per image, where :math:`D` is the number of detections in that image, each row the
            class id, the score and the ``x, y, w, h`` box in that image's pixels; with
            ``confidence_filtering=False`` or a threshold of 0, one ``(B, Q, 6)`` tensor of the ``Q`` best detections
            of each image.

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
        """Draw the boxes of the detections on copies of the images.

        Args:
            images: input images, a ``(B, 3, H, W)`` tensor or a list of ``(3, H, W)`` images.
            detections: the detections of each image, rows ending with the ``x, y, w, h`` box; computed with
                :meth:`forward` when None.
            output_type: ``"torch"`` or ``"pil"``.

        Returns:
            A list with one ``(3, H, W)`` image per input image, or a list of PIL images for ``output_type="pil"``.

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
        """Save the images with their boxes drawn by :meth:`visualize` as PNG files.

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
    """Keep the detections above a confidence threshold and in given classes, per image.

    ``kept = BoxFiltering(confidence_threshold=0.3, classes_to_keep=[0, 2])(dets)`` keeps the rows
    ``(class_id, score, x, y, w, h)`` of ``(B, D, 6)`` detections that score strictly above the threshold and are in
    ``classes_to_keep``. It returns a list of ``B`` tensors ``(D_b, 6)``, or, with ``filter_as_zero=True``, the
    ``(B, D, 6)`` input with the other rows zeroed.

    Args:
        confidence_threshold: a float or 0-d tensor; the rows scoring strictly above it are kept. If None, 0.
        classes_to_keep: a list or 1-d tensor of the class ids to keep. If None, keep all classes.
        filter_as_zero: whether to set the other rows to zero in one ``(B, D, 6)`` tensor instead of dropping them.

    See Also:
        - :class:`~kornia.models.rt_detr.DETRPostProcessor`: RT-DETR's post-processor, which filters its detections
          by score with this class.
        - :func:`results_from_detections`: converts the kept rows of one image to Python objects.

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
            boxes: ``(B, D, 6)``, where B is the batch size, D is the number of detections in each image,
                6 represent (class_id, confidence_score, x, y, w, h).
            confidence_threshold: a 0-d tensor that replaces the constructor's threshold for this call. If None,
                the constructor's.
            classes_to_keep: a 1-d torch.Tensor of classes to keep, replacing the constructor's for this call. If
                None, the constructor's.

        Returns:
            If ``filter_as_zero`` is True, the ``(B, D, 6)`` input with the filtered-out rows set to zero.
            If ``filter_as_zero`` is False, a list of ``B`` tensors ``(D_b, 6)``, the rows kept in each image.

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
    """Build an ObjectDetector around a pretrained RT-DETR network for the 80 COCO classes.

    ``detector = RTDETRDetectorBuilder.build("rtdetr_r18vd")`` downloads the released weights on first use and
    returns an :class:`ObjectDetector`. ``dets = detector(images)`` takes RGB images in :math:`[0, 1]` and returns a
    list with one ``(D, 6)`` tensor per image, rows ``(class_id, score, x, y, w, h)`` scoring above 0.3. The box is in
    that image's pixels, and ``class_id`` is the COCO class index, from 0 to 79.

    See Also:
        - :class:`ObjectDetector`: the wrapper it returns, for your own detection network.
        - :class:`~kornia.models.rt_detr.RTDETR`: the network, to build from a configuration or to fine-tune.

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
        """Build an RT-DETR object detector.

        Pass at most one of ``model_name`` and ``config``; both raise ``ValueError``. Without either, it warns and
        builds the pretrained ``rtdetr_r18vd``.

        Args:
            model_name:
                Name of the RT-DETR model to load. Can be one of the available pretrained models.
                Including 'rtdetr_r18vd', 'rtdetr_r34vd', 'rtdetr_r50vd_m', 'rtdetr_r50vd', 'rtdetr_r101vd'.
            config:
                A custom configuration object for building the RT-DETR model.
            pretrained:
                Whether to load a pretrained version of the model (applies when ``model_name`` is provided).
            image_size:
                The size to which input images will be resized during preprocessing, used only with ``config``. If
                None, ``config.input_size`` is used. With ``model_name``, or without either, it is ignored: the input is
                resized to the 640 px the published weights were trained for. Recommended scales include
                [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800].
            confidence_threshold: The detections scoring strictly above it are kept. If None,
                ``config.confidence_threshold`` with ``config``, else 0.3.
            confidence_filtering: Whether to filter results based on confidence scores. None filters. Without
                filtering, or with a threshold of 0, the detector returns one ``(B, Q, 6)`` tensor, ``Q`` the
                decoder's number of queries (300 for the released models).

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
