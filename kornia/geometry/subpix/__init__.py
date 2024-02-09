from __future__ import annotations

from kornia.geometry.subpix.dsnt import render_gaussian2d, spatial_expectation2d, spatial_softmax2d
from kornia.geometry.subpix.nms import NonMaximaSuppression2d, NonMaximaSuppression3d, nms2d, nms3d
from kornia.geometry.subpix.spatial_soft_argmax import (
    ConvQuadInterp3d,
    ConvSoftArgmax2d,
    ConvSoftArgmax3d,
    SpatialSoftArgmax2d,
    conv_quad_interp3d,
    conv_soft_argmax2d,
    conv_soft_argmax3d,
    spatial_soft_argmax2d,
)

__all__ = [
    "conv_soft_argmax2d",
    "conv_soft_argmax3d",
    "ConvSoftArgmax2d",
    "ConvSoftArgmax3d",
    "spatial_soft_argmax2d",
    "SpatialSoftArgmax2d",
    "conv_quad_interp3d",
    "ConvQuadInterp3d",
    "NonMaximaSuppression2d",
    "NonMaximaSuppression3d",
    "nms2d",
    "nms3d",
    "spatial_softmax2d",
    "spatial_expectation2d",
    "render_gaussian2d",
]
