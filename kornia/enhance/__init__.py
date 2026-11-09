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

"""Adjust, equalize, normalize and whiten images, simulate JPEG compression, and sum them cumulatively.

Which function to use, with the module form after the function:

- Brightness: add a factor with :func:`adjust_brightness`, :class:`AdjustBrightness`; multiply by it, as PIL does,
  with :func:`adjust_brightness_accumulative`, :class:`AdjustBrightnessAccumulative`.
- Contrast: multiply by a factor with :func:`adjust_contrast`, :class:`AdjustContrast`; blend with the mean gray level,
  as PIL does, with :func:`adjust_contrast_with_mean_subtraction`, :class:`AdjustContrastWithMeanSubtraction`.
- Gamma, logarithmic and sigmoid curves: :func:`adjust_gamma`, :class:`AdjustGamma`; :func:`adjust_log`,
  :class:`AdjustLog`; :func:`adjust_sigmoid`, :class:`AdjustSigmoid`.
- Hue of an RGB image: :func:`adjust_hue`, :class:`AdjustHue`; of an HSV image: :func:`adjust_hue_raw`.
- Saturation of an RGB image: scale the HSV saturation with :func:`adjust_saturation`, :class:`AdjustSaturation`; blend
  with the gray image, as PIL does, with :func:`adjust_saturation_with_gray_subtraction`,
  :class:`AdjustSaturationWithGraySubtraction`; of an HSV image: :func:`adjust_saturation_raw`.
- Shift the RGB channels by different amounts: :func:`shift_rgb`.
- Invert: :func:`invert`, :class:`Invert`; invert the bright pixels: :func:`solarize`; keep the top bits:
  :func:`posterize`.
- Sharpen or blur: :func:`sharpness`.
- Blend two images: :func:`add_weighted`, :class:`AddWeighted`.
- Multiply by a constant: :class:`Rescale`.
- Threshold at a fixed level: :func:`threshold`, :class:`Threshold`, :class:`ThresholdType`.
- Equalize each channel's histogram: :func:`equalize`; of a volume: :func:`equalize3d`; tile by tile with a clipped
  histogram: :func:`equalize_clahe`.
- Histograms: :func:`histogram` of sample rows, :func:`histogram2d` of two rows, and
  :func:`image_histogram2d` of each image channel.
- Normalize with a mean and standard deviation: :func:`normalize`, :class:`Normalize`; undo it: :func:`denormalize`,
  :class:`Denormalize`; scale each channel plane to a value range: :func:`normalize_min_max`.
- Whiten data with ZCA: :func:`zca_whiten`, or fit once with :class:`ZCAWhitening`; the pieces are :func:`zca_mean` and
  :func:`linear_transform`.
- Compress and decompress with a differentiable JPEG model: :func:`jpeg_codec_differentiable`,
  :class:`JPEGCodecDifferentiable`.
- Cumulative sums: the integral image of the last two axes with :func:`integral_image`, :class:`IntegralImage`; along
  any axes with :func:`integral_tensor`, :class:`IntegralTensor`.
"""

from .adjust import (
    AdjustBrightness,
    AdjustBrightnessAccumulative,
    AdjustContrast,
    AdjustContrastWithMeanSubtraction,
    AdjustGamma,
    AdjustHue,
    AdjustLog,
    AdjustSaturation,
    AdjustSaturationWithGraySubtraction,
    AdjustSigmoid,
    Invert,
    adjust_brightness,
    adjust_brightness_accumulative,
    adjust_contrast,
    adjust_contrast_with_mean_subtraction,
    adjust_gamma,
    adjust_hue,
    adjust_hue_raw,
    adjust_log,
    adjust_saturation,
    adjust_saturation_raw,
    adjust_saturation_with_gray_subtraction,
    adjust_sigmoid,
    equalize,
    equalize3d,
    invert,
    posterize,
    sharpness,
    solarize,
)
from .core import AddWeighted, add_weighted
from .equalization import equalize_clahe
from .histogram import histogram, histogram2d, image_histogram2d
from .integral import IntegralImage, IntegralTensor, integral_image, integral_tensor
from .jpeg import JPEGCodecDifferentiable, jpeg_codec_differentiable
from .normalize import Denormalize, Normalize, denormalize, normalize, normalize_min_max
from .rescale import Rescale
from .shift_rgb import shift_rgb
from .threshold import Threshold, ThresholdType, threshold
from .zca import ZCAWhitening, linear_transform, zca_mean, zca_whiten

__all__ = [
    "AddWeighted",
    "AdjustBrightness",
    "AdjustBrightnessAccumulative",
    "AdjustContrast",
    "AdjustContrastWithMeanSubtraction",
    "AdjustGamma",
    "AdjustHue",
    "AdjustLog",
    "AdjustSaturation",
    "AdjustSaturationWithGraySubtraction",
    "AdjustSigmoid",
    "Denormalize",
    "IntegralImage",
    "IntegralTensor",
    "Invert",
    "JPEGCodecDifferentiable",
    "Normalize",
    "Rescale",
    "Threshold",
    "ThresholdType",
    "ZCAWhitening",
    "add_weighted",
    "adjust_brightness",
    "adjust_brightness_accumulative",
    "adjust_contrast",
    "adjust_contrast_with_mean_subtraction",
    "adjust_gamma",
    "adjust_hue",
    "adjust_hue_raw",
    "adjust_log",
    "adjust_saturation",
    "adjust_saturation_raw",
    "adjust_saturation_with_gray_subtraction",
    "adjust_sigmoid",
    "denormalize",
    "equalize",
    "equalize3d",
    "equalize_clahe",
    "histogram",
    "histogram2d",
    "image_histogram2d",
    "integral_image",
    "integral_tensor",
    "invert",
    "jpeg_codec_differentiable",
    "linear_transform",
    "normalize",
    "normalize_min_max",
    "posterize",
    "sharpness",
    "shift_rgb",
    "solarize",
    "threshold",
    "zca_mean",
    "zca_whiten",
]
