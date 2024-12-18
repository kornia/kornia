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

from .bilateral import BilateralBlur, JointBilateralBlur, bilateral_blur, joint_bilateral_blur
from .blur import BoxBlur, box_blur
from .blur_pool import (
    BlurPool2D,
    EdgeAwareBlurPool2D,
    MaxBlurPool2D,
    blur_pool2d,
    edge_aware_blur_pool2d,
    max_blur_pool2d,
)
from .canny import Canny, canny
from .dexined import DexiNed
from .dissolving import StableDiffusionDissolving
from .filter import filter2d, filter2d_separable, filter3d
from .gaussian import GaussianBlur2d, gaussian_blur2d, gaussian_blur2d_t
from .guided import GuidedBlur, guided_blur
from .in_range import InRange, in_range
from .kernels import (
    gaussian,
    get_binary_kernel2d,
    get_box_kernel1d,
    get_box_kernel2d,
    get_diff_kernel2d,
    get_gaussian_discrete_kernel1d,
    get_gaussian_erf_kernel1d,
    get_gaussian_kernel1d,
    get_gaussian_kernel1d_t,
    get_gaussian_kernel2d,
    get_gaussian_kernel2d_t,
    get_gaussian_kernel3d,
    get_gaussian_kernel3d_t,
    get_hanning_kernel1d,
    get_hanning_kernel2d,
    get_laplacian_kernel1d,
    get_laplacian_kernel2d,
    get_sobel_kernel2d,
    get_spatial_gradient_kernel2d,
    get_spatial_gradient_kernel3d,
    laplacian_1d,
)
from .kernels_geometry import get_motion_kernel2d, get_motion_kernel3d
from .laplacian import Laplacian, laplacian
from .median import MedianBlur, median_blur
from .motion import MotionBlur, MotionBlur3D, motion_blur, motion_blur3d
from .sobel import Sobel, SpatialGradient, SpatialGradient3d, sobel, spatial_gradient, spatial_gradient3d
from .unsharp import UnsharpMask, unsharp_mask

__all__ = [
    "BilateralBlur",
    "BlurPool2D",
    "BoxBlur",
    "Canny",
    "DexiNed",
    "EdgeAwareBlurPool2D",
    "GaussianBlur2d",
    "GuidedBlur",
    "InRange",
    "JointBilateralBlur",
    "Laplacian",
    "MaxBlurPool2D",
    "MedianBlur",
    "MotionBlur",
    "MotionBlur3D",
    "Sobel",
    "SpatialGradient",
    "SpatialGradient3d",
    "StableDiffusionDissolving",
    "UnsharpMask",
    "bilateral_blur",
    "blur_pool2d",
    "box_blur",
    "canny",
    "filter2d",
    "filter2d_separable",
    "filter3d",
    "gaussian",
    "gaussian_blur2d",
    "gaussian_blur2d_t",
    "get_binary_kernel2d",
    "get_box_kernel1d",
    "get_box_kernel2d",
    "get_diff_kernel2d",
    "get_gaussian_discrete_kernel1d",
    "get_gaussian_erf_kernel1d",
    "get_gaussian_kernel1d",
    "get_gaussian_kernel1d_t",
    "get_gaussian_kernel2d",
    "get_gaussian_kernel2d_t",
    "get_gaussian_kernel3d",
    "get_gaussian_kernel3d_t",
    "get_hanning_kernel1d",
    "get_hanning_kernel2d",
    "get_laplacian_kernel1d",
    "get_laplacian_kernel2d",
    "get_motion_kernel2d",
    "get_motion_kernel3d",
    "get_sobel_kernel2d",
    "get_spatial_gradient_kernel2d",
    "get_spatial_gradient_kernel3d",
    "guided_blur",
    "in_range",
    "joint_bilateral_blur",
    "laplacian",
    "laplacian_1d",
    "max_blur_pool2d",
    "median_blur",
    "motion_blur",
    "motion_blur3d",
    "sobel",
    "spatial_gradient",
    "spatial_gradient3d",
    "unsharp_mask",
]
