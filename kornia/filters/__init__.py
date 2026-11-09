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

"""Blur, sharpen, threshold and differentiate images, or filter them with your own kernel.

Filters take images ``(B, C, H, W)``; the 3D filters take volumes ``(B, C, D, H, W)``. A pair ``kernel_size`` is
``(kH, kW)``, square in the blur pools, and a pair ``sigma`` is ``(sigma_y, sigma_x)``, rows first. A ``border_type``
defaults to ``'reflect'`` in 2D and to ``'replicate'`` in 3D. :func:`sobel`, :func:`spatial_gradient` and
:func:`spatial_gradient3d` always replicate the border, and :func:`blur_pool2d` and :func:`max_blur_pool2d` zero-pad
it.

Which function to use, with the module form after the function:

- Average a window: :func:`box_blur`, :class:`BoxBlur`.
- Blur with a Gaussian: :func:`gaussian_blur2d`, :class:`GaussianBlur2d`.
- Remove isolated outliers and keep edges sharp: :func:`median_blur`, :class:`MedianBlur`.
- Smooth and keep strong edges: :func:`bilateral_blur`, :class:`BilateralBlur`; keep the edges of a second image:
  :func:`joint_bilateral_blur`, :class:`JointBilateralBlur`, or :func:`guided_blur`, :class:`GuidedBlur`.
- Blur along a line, as camera or object motion does: :func:`motion_blur`, :class:`MotionBlur`; for a volume:
  :func:`motion_blur3d`, :class:`MotionBlur3D`.
- Sharpen: :func:`unsharp_mask`, :class:`UnsharpMask`.
- Downsample a feature map with less aliasing: :func:`blur_pool2d`, :class:`BlurPool2D`; max-pool it first:
  :func:`max_blur_pool2d`, :class:`MaxBlurPool2D`.
- Blur a feature map at its size and keep the pixels on edges: :func:`edge_aware_blur_pool2d`,
  :class:`EdgeAwareBlurPool2D`.
- Dissolve an image with a Stable Diffusion model: :class:`StableDiffusionDissolving`.
- Edge strength: :func:`sobel`, :class:`Sobel`.
- Signed x and y derivatives: :func:`spatial_gradient`, :class:`SpatialGradient`; of a volume:
  :func:`spatial_gradient3d`, :class:`SpatialGradient3d`.
- Second-derivative response: :func:`laplacian`, :class:`Laplacian`.
- Thin binary edge map: :func:`canny`, :class:`Canny`.
- Keep the pixels inside a value range: :func:`in_range`, :class:`InRange`.
- Threshold each image plane at its Otsu level: :func:`otsu_threshold`, :class:`OtsuThreshold`.
- Filter with your own 2D kernel: :func:`filter2d`, or :func:`correlate2d` and :func:`convolve2d`, which fix
  ``behaviour``; for two 1D kernels: :func:`filter2d_separable`; for a large kernel: :func:`fft_conv`.
- Filter a volume with your own kernel: :func:`filter3d`, or :func:`correlate3d` and :func:`convolve3d`.
- Gaussian kernels: :func:`get_gaussian_kernel1d`, :func:`get_gaussian_kernel2d`, :func:`get_gaussian_kernel3d`; sampled
  around any mean: :func:`gaussian`; discrete, from Bessel functions: :func:`get_gaussian_discrete_kernel1d`;
  integrated per pixel: :func:`get_gaussian_erf_kernel1d`.
- Box kernels: :func:`get_box_kernel1d`, :func:`get_box_kernel2d`.
- Hanning windows: :func:`get_hanning_kernel1d`, :func:`get_hanning_kernel2d`.
- Laplacian kernels: :func:`get_laplacian_kernel1d`, :func:`get_laplacian_kernel2d`; without size checks:
  :func:`laplacian_1d`.
- Derivative kernels: :func:`get_sobel_kernel2d`, :func:`get_diff_kernel2d`, :func:`get_spatial_gradient_kernel2d`,
  :func:`get_spatial_gradient_kernel3d`.
- Motion-blur kernels: :func:`get_motion_kernel2d`, :func:`get_motion_kernel3d`.
- One-hot kernels that pick each pixel of a window: :func:`get_binary_kernel2d`.
"""

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
from .dissolving import StableDiffusionDissolving
from .filter import convolve2d, convolve3d, correlate2d, correlate3d, fft_conv, filter2d, filter2d_separable, filter3d
from .gaussian import GaussianBlur2d, gaussian_blur2d
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
    get_gaussian_kernel2d,
    get_gaussian_kernel3d,
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
from .otsu_thresholding import OtsuThreshold, otsu_threshold
from .sobel import Sobel, SpatialGradient, SpatialGradient3d, sobel, spatial_gradient, spatial_gradient3d
from .unsharp import UnsharpMask, unsharp_mask

__all__ = [
    "BilateralBlur",
    "BlurPool2D",
    "BoxBlur",
    "Canny",
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
    "OtsuThreshold",
    "Sobel",
    "SpatialGradient",
    "SpatialGradient3d",
    "StableDiffusionDissolving",
    "UnsharpMask",
    "bilateral_blur",
    "blur_pool2d",
    "box_blur",
    "canny",
    "convolve2d",
    "convolve3d",
    "correlate2d",
    "correlate3d",
    "edge_aware_blur_pool2d",
    "fft_conv",
    "filter2d",
    "filter2d_separable",
    "filter3d",
    "gaussian",
    "gaussian_blur2d",
    "get_binary_kernel2d",
    "get_box_kernel1d",
    "get_box_kernel2d",
    "get_diff_kernel2d",
    "get_gaussian_discrete_kernel1d",
    "get_gaussian_erf_kernel1d",
    "get_gaussian_kernel1d",
    "get_gaussian_kernel2d",
    "get_gaussian_kernel3d",
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
    "otsu_threshold",
    "sobel",
    "spatial_gradient",
    "spatial_gradient3d",
    "unsharp_mask",
]
