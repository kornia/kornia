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

import warnings
from typing import Any, Optional

import torch

from kornia.contrib.detection.base import (
    BoundingBox as BoundingBoxBase,
)
from kornia.contrib.detection.base import (
    BoundingBoxDataFormat,
)
from kornia.contrib.detection.base import (
    ObjectDetector as ObjectDetectorBase,
)
from kornia.contrib.detection.base import (
    ObjectDetectorResult as ObjectDetectorResultBase,
)
from kornia.contrib.detection.base import (
    results_from_detections as results_from_detections_base,
)
from kornia.models.processors import ResizePreProcessor as ResizePreProcessorBase
from kornia.models.rt_detr import DETRPostProcessor
from kornia.models.rt_detr.model import RTDETR, RTDETRConfig

__all__ = [
    "BoundingBox",
    "BoundingBoxDataFormat",
    "ObjectDetector",
    "ObjectDetectorBuilder",
    "ObjectDetectorResult",
    "ResizePreProcessor",
    "results_from_detections",
]


class BoundingBox(BoundingBoxBase):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        warnings.warn(
            "BoundingBox is deprecated and will be removed in v0.8.0. "
            "Use kornia.contrib.detection.BoundingBox instead.",
            DeprecationWarning,
            stacklevel=1,
        )


def results_from_detections(*args: Any, **kwargs: Any) -> list[ObjectDetectorResultBase]:
    """Return detector results."""
    warnings.warn(
        "results_from_detections is deprecated and will be removed in v0.8.0. "
        "Use kornia.contrib.detection.results_from_detections instead.",
        DeprecationWarning,
        stacklevel=1,
    )
    return results_from_detections_base(*args, **kwargs)


class ResizePreProcessor(ResizePreProcessorBase):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        warnings.warn(
            "ResizePreProcessor is deprecated and will be removed in v0.8.0. "
            "Use kornia.models.processors.ResizePreProcessor instead.",
            DeprecationWarning,
            stacklevel=1,
        )


class ObjectDetector(ObjectDetectorBase):  # type: ignore[misc]
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        warnings.warn(
            "ObjectDetector is deprecated and will be removed in v0.8.0. "
            "Use kornia.contrib.detection.ObjectDetector instead.",
            DeprecationWarning,
            stacklevel=1,
        )


class ObjectDetectorResult(ObjectDetectorResultBase):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        warnings.warn(
            "ObjectDetectorResult is deprecated and will be removed in v0.8.0. "
            "Use kornia.contrib.detection.ObjectDetectorResult instead.",
            DeprecationWarning,
            stacklevel=1,
        )


class ObjectDetectorBuilder:
    """A builder class for constructing RT-DETR object detection models.

    This class provides static methods to:
        - Build an object detection model from a model name or configuration.
        - Export the model to ONNX format for inference.

    .. code-block:: python

        images = kornia.utils.sample.get_sample_images()
        model = ObjectDetectorBuilder.build()
        model.save(images)
    """

    @staticmethod
    def build(
        model_name: Optional[str] = None,
        config: Optional[RTDETRConfig] = None,
        pretrained: bool = True,
        image_size: Optional[int] = None,
        confidence_threshold: Optional[float] = None,
        confidence_filtering: Optional[bool] = None,
    ) -> ObjectDetectorBase:
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

        from kornia.models.processors import ResizePreProcessor

        return ObjectDetectorBase(
            model,
            ResizePreProcessor(image_size, image_size),
            DETRPostProcessor(
                confidence_threshold=confidence_threshold,
                confidence_filtering=confidence_filtering or not torch.onnx.is_in_onnx_export(),
                num_classes=model.decoder.num_classes,
                num_top_queries=model.decoder.num_queries,
            ),
        )
