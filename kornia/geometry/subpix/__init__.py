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

"""Sub-pixel locations from heatmaps and scale-space volumes: soft-argmax, strict NMS and quadratic refinement.

2D coordinates are ``(x, y)`` = (column, row) and 3D ones ``(d, x, y)``, depth first (the scale level of a
scale-space volume), with integer pixel centres; see :ref:`Coordinates and sizes <coordinate-conventions>`. A
``normalized_coordinates=True`` result is in :math:`[-1, 1]`, corner-aligned; the default differs between functions.
A class in parentheses is the module form of the function before it: it stores the arguments.

Which function to use:

- One location per heatmap channel, the softmax-weighted mean over the whole map: :func:`spatial_soft_argmax2d`
  (:class:`SpatialSoftArgmax2d`), or its two steps, :func:`spatial_softmax2d` and :func:`spatial_expectation2d`.
  The other direction, a Gaussian heatmap around a given mean: :func:`render_gaussian2d`.
- One location per sliding window, for maps with several peaks: :func:`conv_soft_argmax2d`
  (:class:`ConvSoftArgmax2d`), and for ``(B, C, D, H, W)`` volumes :func:`conv_soft_argmax3d`
  (:class:`ConvSoftArgmax3d`).
- Strict local maxima: of a map, :func:`nms2d` (:class:`NonMaximaSuppression2d`); of a volume, :func:`nms3d`
  (:class:`NonMaximaSuppression3d`); the maxima and minima of a volume in one call, :func:`nms3d_minmax`.
- Sub-voxel refinement of the maxima of a scale-space volume by a quadratic fit: :class:`AdaptiveQuadInterp3d`, which
  by default runs :func:`conv_quad_interp3d` (:class:`ConvQuadInterp3d`) on CUDA and :func:`iterative_quad_interp3d`
  (:class:`IterativeQuadInterp3d`) elsewhere.
"""

from __future__ import annotations

from .dsnt import render_gaussian2d, spatial_expectation2d, spatial_softmax2d
from .nms import NonMaximaSuppression2d, NonMaximaSuppression3d, nms2d, nms3d, nms3d_minmax
from .spatial_soft_argmax import (
    AdaptiveQuadInterp3d,
    ConvQuadInterp3d,
    ConvSoftArgmax2d,
    ConvSoftArgmax3d,
    IterativeQuadInterp3d,
    SpatialSoftArgmax2d,
    conv_quad_interp3d,
    conv_soft_argmax2d,
    conv_soft_argmax3d,
    iterative_quad_interp3d,
    spatial_soft_argmax2d,
)

__all__ = [
    "AdaptiveQuadInterp3d",
    "ConvQuadInterp3d",
    "ConvSoftArgmax2d",
    "ConvSoftArgmax3d",
    "IterativeQuadInterp3d",
    "NonMaximaSuppression2d",
    "NonMaximaSuppression3d",
    "SpatialSoftArgmax2d",
    "conv_quad_interp3d",
    "conv_soft_argmax2d",
    "conv_soft_argmax3d",
    "iterative_quad_interp3d",
    "nms2d",
    "nms3d",
    "nms3d_minmax",
    "render_gaussian2d",
    "spatial_expectation2d",
    "spatial_soft_argmax2d",
    "spatial_softmax2d",
]
