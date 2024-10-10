from __future__ import annotations

import warnings
from typing import Any

from kornia.models.detection.base import (
    BoundingBox as BoundingBoxBase,
)
from kornia.models.detection.base import (
    BoundingBoxDataFormat,
)
from kornia.models.detection.base import (
    ObjectDetector as ObjectDetectorBase,
)
from kornia.models.detection.base import (
    ObjectDetectorResult as ObjectDetectorResultBase,
)
from kornia.models.detection.base import (
    results_from_detections as results_from_detections_base,
)
from kornia.models.utils import ResizePreProcessor as ResizePreProcessorBase

__all__ = [
    "BoundingBoxDataFormat",
    "BoundingBox",
    "results_from_detections",
    "ResizePreProcessor",
    "ObjectDetector",
    "ObjectDetectorResult",
]


class BoundingBox(BoundingBoxBase):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        warnings.warn(
            "BoundingBox is deprecated and will be removed in v0.8.0. "
            "Use kornia.models.detection.BoundingBox instead.",
            DeprecationWarning,
        )


def results_from_detections(*args: Any, **kwargs: Any) -> list[ObjectDetectorResultBase]:
    warnings.warn(
        "results_from_detections is deprecated and will be removed in v0.8.0. "
        "Use kornia.models.detection.results_from_detections instead.",
        DeprecationWarning,
    )
    return results_from_detections_base(*args, **kwargs)


class ResizePreProcessor(ResizePreProcessorBase):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        warnings.warn(
            "ResizePreProcessor is deprecated and will be removed in v0.8.0. "
            "Use kornia.models.utils.ResizePreProcessor instead.",
            DeprecationWarning,
        )


class ObjectDetector(ObjectDetectorBase):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        warnings.warn(
            "ObjectDetector is deprecated and will be removed in v0.8.0. "
            "Use kornia.models.detection.ObjectDetector instead.",
            DeprecationWarning,
        )


class ObjectDetectorResult(ObjectDetectorResultBase):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        warnings.warn(
            "ObjectDetectorResult is deprecated and will be removed in v0.8.0. "
            "Use kornia.models.detection.ObjectDetectorResult instead.",
            DeprecationWarning,
        )
