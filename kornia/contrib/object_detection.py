import warnings

from kornia.models.detector.base import (
    BoundingBox as BoundingBoxBase,
)
from kornia.models.detector.base import (
    BoundingBoxDataFormat,
)
from kornia.models.detector.base import (
    ObjectDetector as ObjectDetectorBase,
)
from kornia.models.detector.base import (
    ObjectDetectorResult as ObjectDetectorResultBase,
)
from kornia.models.detector.base import (
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
    warnings.warn(
        "BoundingBox is deprecated and will be removed in v0.8.0. Use kornia.models.detector.BoundingBox instead.",
        DeprecationWarning,
    )


def results_from_detections(*args, **kwargs):
    warnings.warn(
        "results_from_detections is deprecated and will be removed in v0.8.0. Use kornia.models.detector.results_from_detections instead.",
        DeprecationWarning,
    )
    return results_from_detections_base(*args, **kwargs)


class ResizePreProcessor(ResizePreProcessorBase):
    warnings.warn(
        "ResizePreProcessor is deprecated and will be removed in v0.8.0. Use kornia.models.utils.ResizePreProcessor instead.",
        DeprecationWarning,
    )


class ObjectDetector(ObjectDetectorBase):
    warnings.warn(
        "ObjectDetector is deprecated and will be removed in v0.8.0. Use kornia.models.detector.ObjectDetector instead.",
        DeprecationWarning,
    )


class ObjectDetectorResult(ObjectDetectorResultBase):
    warnings.warn(
        "ObjectDetectorResult is deprecated and will be removed in v0.8.0. Use kornia.models.detector.ObjectDetectorResult instead.",
        DeprecationWarning,
    )
