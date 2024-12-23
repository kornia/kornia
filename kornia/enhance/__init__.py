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
    "JPEGCodecDifferentiable",
    "Normalize",
    "Rescale",
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
    "zca_mean",
    "zca_whiten",
]
