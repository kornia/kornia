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

from .dsnt import render_gaussian2d, spatial_expectation2d, spatial_softmax2d
from .nms import NonMaximaSuppression2d, NonMaximaSuppression3d, nms2d, nms3d
from .spatial_soft_argmax import (
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
    "ConvQuadInterp3d",
    "ConvSoftArgmax2d",
    "ConvSoftArgmax3d",
    "NonMaximaSuppression2d",
    "NonMaximaSuppression3d",
    "SpatialSoftArgmax2d",
    "conv_quad_interp3d",
    "conv_soft_argmax2d",
    "conv_soft_argmax3d",
    "nms2d",
    "nms3d",
    "render_gaussian2d",
    "spatial_expectation2d",
    "spatial_soft_argmax2d",
    "spatial_softmax2d",
]
