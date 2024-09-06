from typing import Optional
import warnings

from kornia.core import Module
from kornia.contrib.models.rt_detr.model import RTDETR, RTDETRConfig
from kornia.contrib.models.rt_detr import DETRPostProcessor
from kornia.contrib.object_detection import ResizePreProcessor, ObjectDetector


class RTDETRDetectorBuilder:

    @staticmethod
    def build(
        model_name: Optional[str] = None,
        config: Optional[RTDETRConfig] = None,
        pretrained: bool = True,
        image_size: int = 640,
        confidence_threshold: float = 0.5
    ) -> ObjectDetector:
        if (model_name is not None and config is not None):
            raise ValueError("Either `model_name` or `config` should be `None`.")
        
        if model_name is None and config is None:
            warnings.warn("No `model_name` or `config` found. Will build `rtdetr_r18vd`.")
            model_name = "rtdetr_r18vd"
            
        if config is not None:
            model = RTDETR.from_config(config)
        else:
            if pretrained:
                model = RTDETR.from_pretrained(model_name)
            else:
                model = RTDETR.from_name(model_name)
        
        return ObjectDetector(
            model,
            ResizePreProcessor(image_size),
            DETRPostProcessor(confidence_threshold)
        )
