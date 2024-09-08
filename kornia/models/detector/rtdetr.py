import warnings
from typing import Optional

import torch
from torch import nn

from kornia.contrib.models.rt_detr import DETRPostProcessor
from kornia.contrib.models.rt_detr.model import RTDETR, RTDETRConfig
from kornia.contrib.object_detection import ObjectDetector, ResizePreProcessor
from kornia.core import rand

__all__ = ["RTDETRDetectorBuilder"]


class RTDETRDetectorBuilder:
    """A builder class for constructing RT-DETR object detection models.

    This class provides static methods to:
        - Build an object detection model from a model name or configuration.
        - Export the model to ONNX format for inference.
    """

    @staticmethod
    def build(
        model_name: Optional[str] = None,
        config: Optional[RTDETRConfig] = None,
        pretrained: bool = True,
        image_size: Optional[int] = 640,
        confidence_threshold: float = 0.5,
    ) -> ObjectDetector:
        """Builds and returns an RT-DETR object detector model.

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
                If None, no resizing will be performed before passing to the model. Recommended scales include
                [480, 512, 544, 576, 608, 640, 640, 640, 672, 704, 736, 768, 800].
            confidence_threshold:
                The confidence threshold used during post-processing to filter detections.

        Returns:
            ObjectDetector
                An object detector instance initialized with the specified model, preprocessor, and post-processor.
        """
        if model_name is not None and config is not None:
            raise ValueError("Either `model_name` or `config` should be `None`.")

        if config is not None:
            model = RTDETR.from_config(config)
        elif model_name is not None:
            if pretrained:
                model = RTDETR.from_pretrained(model_name)
            else:
                model = RTDETR.from_name(model_name)
        else:
            warnings.warn("No `model_name` or `config` found. Will build pretrained `rtdetr_r18vd`.")
            model = RTDETR.from_pretrained("rtdetr_r18vd")

        return ObjectDetector(
            model,
            ResizePreProcessor((image_size, image_size)) if image_size is not None else nn.Identity(),
            DETRPostProcessor(confidence_threshold),
        )

    @staticmethod
    def to_onnx(
        model_name: Optional[str] = None,
        onnx_name: Optional[str] = None,
        config: Optional[RTDETRConfig] = None,
        pretrained: bool = True,
        image_size: Optional[int] = 640,
        confidence_threshold: float = 0.5,
    ) -> tuple[str, ObjectDetector]:
        """Exports an RT-DETR object detection model to ONNX format.

        Either `model_name` or `config` must be provided. If neither is provided,
        a default pretrained model (`rtdetr_r18vd`) will be built.

        Args:
            model_name:
                Name of the RT-DETR model to load. Can be one of the available pretrained models.
            config:
                A custom configuration object for building the RT-DETR model.
            pretrained:
                Whether to load a pretrained version of the model (applies when `model_name` is provided).
            image_size:
                The size to which input images will be resized during preprocessing.
                If None, image_size will be dynamic. Recommended scales include
                [480, 512, 544, 576, 608, 640, 640, 640, 672, 704, 736, 768, 800].
            confidence_threshold:
                The confidence threshold used during post-processing to filter detections.

        Returns:
            - The name of the ONNX model.
            - The exported torch model.
        """

        detector = RTDETRDetectorBuilder.build(
            model_name=model_name,
            config=config,
            pretrained=pretrained,
            image_size=image_size,
            confidence_threshold=confidence_threshold,
        )
        if onnx_name is None:
            _model_name = model_name
            if model_name is None and config is not None:
                _model_name = "rtdetr-customized"
            elif model_name is None and config is None:
                _model_name = "rtdetr_r18vd"
            onnx_name = f"Kornia-RTDETR-{_model_name}-{image_size}.onnx"

        val_image = rand(1, 3, image_size, image_size)
        if image_size is None:
            val_image = rand(1, 3, 640, 640)

        dynamic_axes = {
            "input": {0: "batch_size", 2: "height", 3: "width"},
            "output": {0: "batch_size"}
        }
        torch.onnx.export(
            detector,
            val_image,
            onnx_name,
            export_params=True,
            opset_version=17,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes=dynamic_axes,
        )

        return onnx_name, detector
