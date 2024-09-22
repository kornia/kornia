import warnings
from typing import Optional

import torch

from kornia.contrib.models.rt_detr import DETRPostProcessor
from kornia.contrib.models.rt_detr.model import RTDETR, RTDETRConfig
from kornia.models.detection.base import ObjectDetector
from kornia.models.utils import ResizePreProcessor

__all__ = ["RTDETRDetectorBuilder"]


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
        config: Optional[RTDETRConfig] = None,
        pretrained: bool = True,
        image_size: Optional[int] = None,
        confidence_threshold: Optional[float] = None,
        confidence_filtering: Optional[bool] = None,
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
                If None, no resizing will be inferred from config file. Recommended scales include
                [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800].

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
            warnings.warn("No `model_name` or `config` found. Will build pretrained `rtdetr_r18vd`.")
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
