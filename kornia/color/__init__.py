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

from .colormap import AUTUMN, ApplyColorMap, ColorMap, ColorMapType, RGBColor, apply_colormap
from .gray import BgrToGrayscale, GrayscaleToRgb, RgbToGrayscale, bgr_to_grayscale, grayscale_to_rgb, rgb_to_grayscale
from .hls import HlsToRgb, RgbToHls, hls_to_rgb, rgb_to_hls
from .hsv import HsvToRgb, RgbToHsv, hsv_to_rgb, rgb_to_hsv
from .lab import LabToRgb, RgbToLab, lab_to_rgb, rgb_to_lab
from .luv import LuvToRgb, RgbToLuv, luv_to_rgb, rgb_to_luv
from .raw import CFA, RawToRgb, RawToRgb2x2Downscaled, RgbToRaw, raw_to_rgb, raw_to_rgb_2x2_downscaled, rgb_to_raw
from .rgb import (
    BgrToRgb,
    BgrToRgba,
    LinearRgbToRgb,
    NormalsToRgb255,
    Rgb255ToNormals,
    Rgb255ToRgb,
    RgbaToBgr,
    RgbaToRgb,
    RgbToBgr,
    RgbToLinearRgb,
    RgbToRgb255,
    RgbToRgba,
    bgr_to_rgb,
    bgr_to_rgba,
    linear_rgb_to_rgb,
    normals_to_rgb255,
    rgb255_to_normals,
    rgb255_to_rgb,
    rgb_to_bgr,
    rgb_to_linear_rgb,
    rgb_to_rgb255,
    rgb_to_rgba,
    rgba_to_bgr,
    rgba_to_rgb,
)
from .sepia import Sepia, sepia_from_rgb
from .xyz import RgbToXyz, XyzToRgb, rgb_to_xyz, xyz_to_rgb
from .ycbcr import RgbToYcbcr, YcbcrToRgb, rgb_to_y, rgb_to_ycbcr, ycbcr_to_rgb
from .yuv import (
    RgbToYuv,
    RgbToYuv420,
    RgbToYuv422,
    Yuv420ToRgb,
    Yuv422ToRgb,
    YuvToRgb,
    rgb_to_yuv,
    rgb_to_yuv420,
    rgb_to_yuv422,
    yuv420_to_rgb,
    yuv422_to_rgb,
    yuv_to_rgb,
)

__all__ = [
    "AUTUMN",
    "CFA",
    "ApplyColorMap",
    "BgrToGrayscale",
    "BgrToRgb",
    "BgrToRgba",
    "ColorMap",
    "ColorMapType",
    "GrayscaleToRgb",
    "HlsToRgb",
    "HsvToRgb",
    "LabToRgb",
    "LinearRgbToRgb",
    "LuvToRgb",
    "NormalsToRgb255",
    "RGBColor",
    "RawToRgb",
    "RawToRgb2x2Downscaled",
    "Rgb255ToNormals",
    "Rgb255ToRgb",
    "RgbToBgr",
    "RgbToGrayscale",
    "RgbToHls",
    "RgbToHsv",
    "RgbToLab",
    "RgbToLinearRgb",
    "RgbToLuv",
    "RgbToRaw",
    "RgbToRgb255",
    "RgbToRgba",
    "RgbToXyz",
    "RgbToYcbcr",
    "RgbToYuv",
    "RgbToYuv420",
    "RgbToYuv422",
    "RgbaToBgr",
    "RgbaToRgb",
    "Sepia",
    "XyzToRgb",
    "YcbcrToRgb",
    "Yuv420ToRgb",
    "Yuv422ToRgb",
    "YuvToRgb",
    "apply_colormap",
    "bgr_to_grayscale",
    "bgr_to_rgb",
    "bgr_to_rgba",
    "grayscale_to_rgb",
    "hls_to_rgb",
    "hsv_to_rgb",
    "lab_to_rgb",
    "linear_rgb_to_rgb",
    "luv_to_rgb",
    "normals_to_rgb255",
    "raw_to_rgb",
    "raw_to_rgb_2x2_downscaled",
    "rgb255_to_normals",
    "rgb255_to_rgb",
    "rgb_to_bgr",
    "rgb_to_grayscale",
    "rgb_to_hls",
    "rgb_to_hsv",
    "rgb_to_lab",
    "rgb_to_linear_rgb",
    "rgb_to_luv",
    "rgb_to_raw",
    "rgb_to_rgb255",
    "rgb_to_rgba",
    "rgb_to_xyz",
    "rgb_to_y",
    "rgb_to_ycbcr",
    "rgb_to_yuv",
    "rgb_to_yuv420",
    "rgb_to_yuv422",
    "rgba_to_bgr",
    "rgba_to_rgb",
    "sepia",
    "xyz_to_rgb",
    "ycbcr_to_rgb",
    "yuv420_to_rgb",
    "yuv422_to_rgb",
    "yuv_to_rgb",
]

sepia = sepia_from_rgb
