import warnings
from typing import Optional

from kornia.contrib.models.rt_detr import DETRPostProcessor
from kornia.contrib.models.rt_detr.model import RTDETR, RTDETRConfig
from kornia.contrib.object_detection import ObjectDetector, ResizePreProcessor


class RTDETRDetectorBuilder:
    @staticmethod
    def build(
        model_name: Optional[str] = None,
        config: Optional[RTDETRConfig] = None,
        pretrained: bool = True,
        image_size: int = 640,
        confidence_threshold: float = 0.5,
    ) -> ObjectDetector:
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

        return ObjectDetector(model, ResizePreProcessor(image_size), DETRPostProcessor(confidence_threshold))
