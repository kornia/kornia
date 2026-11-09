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

r"""Convert images between color spaces, demosaic Bayer raw images and color single-channel images with a colormap.

Color images are ``(*, C, H, W)`` with the channels at axis -3, and the conversions take any number of leading
dimensions; :func:`apply_colormap` takes ``(C, H, W)`` or ``(B, C, H, W)``. Values are expected in :math:`[0, 1]`,
except for these:

- The ``*rgb255*`` helpers work in :math:`[0, 255]`, and the normals they convert in :math:`[-1, 1]`.
- :func:`rgb_to_xyz` takes linear RGB and :func:`xyz_to_rgb` returns it. Convert with :func:`rgb_to_linear_rgb` and
  :func:`linear_rgb_to_rgb`. XYZ Z reaches 1.09 at white.
- Lab and Luv have L in :math:`[0, 100]`; Lab a and b lie in :math:`[-128, 127]`, and sRGB gives Luv u in
  :math:`[-84, 176]` and v in :math:`[-135, 108]`.
- HLS and HSV hue is in radians, :math:`[0, 2\pi)`.
- YUV has U and V centred on 0, and YCbCr has Cb and Cr centred on 0.5.
- :func:`apply_colormap` also takes values in :math:`[0, 255]`.

Which function to use, with the module form after the function:

- RGB to grayscale: :func:`rgb_to_grayscale`, :class:`RgbToGrayscale`; from BGR: :func:`bgr_to_grayscale`,
  :class:`BgrToGrayscale`; only the luma of YCbCr: :func:`rgb_to_y`.
- Grayscale to RGB: :func:`grayscale_to_rgb`, :class:`GrayscaleToRgb`.
- RGB and BGR: :func:`rgb_to_bgr`, :class:`RgbToBgr`; :func:`bgr_to_rgb`, :class:`BgrToRgb`.
- Add alpha: :func:`rgb_to_rgba`, :class:`RgbToRgba`; from BGR: :func:`bgr_to_rgba`, :class:`BgrToRgba`.
- Remove alpha by compositing over white: :func:`rgba_to_rgb`, which also takes another background,
  :class:`RgbaToRgb`; to BGR: :func:`rgba_to_bgr`, :class:`RgbaToBgr`.
- Scale between :math:`[0, 1]` and :math:`[0, 255]`: :func:`rgb_to_rgb255`, :class:`RgbToRgb255`;
  :func:`rgb255_to_rgb`, :class:`Rgb255ToRgb`.
- Surface normals and RGB in :math:`[0, 255]`: :func:`normals_to_rgb255`, :class:`NormalsToRgb255`;
  :func:`rgb255_to_normals`, :class:`Rgb255ToNormals`.
- Remove or apply the sRGB curve: :func:`rgb_to_linear_rgb`, :class:`RgbToLinearRgb`; :func:`linear_rgb_to_rgb`,
  :class:`LinearRgbToRgb`.
- Hue, lightness and saturation: :func:`rgb_to_hls`, :class:`RgbToHls`; :func:`hls_to_rgb`, :class:`HlsToRgb`.
- Hue, saturation and value: :func:`rgb_to_hsv`, :class:`RgbToHsv`; :func:`hsv_to_rgb`, :class:`HsvToRgb`.
- Perceptual spaces: Lab with :func:`rgb_to_lab`, :class:`RgbToLab`, :func:`lab_to_rgb`, :class:`LabToRgb`; Luv with
  :func:`rgb_to_luv`, :class:`RgbToLuv`, :func:`luv_to_rgb`, :class:`LuvToRgb`.
- CIE XYZ: :func:`rgb_to_xyz`, :class:`RgbToXyz`; :func:`xyz_to_rgb`, :class:`XyzToRgb`.
- Full-range YCbCr: :func:`rgb_to_ycbcr`, :class:`RgbToYcbcr`; :func:`ycbcr_to_rgb`, :class:`YcbcrToRgb`.
- Analog YUV: :func:`rgb_to_yuv`, :class:`RgbToYuv`; :func:`yuv_to_rgb`, :class:`YuvToRgb`.
- YUV planes with subsampled UV: 4:2:0, half the height and width, with :func:`rgb_to_yuv420`, :class:`RgbToYuv420`,
  :func:`yuv420_to_rgb`, :class:`Yuv420ToRgb`; 4:2:2, half the width, with :func:`rgb_to_yuv422`, :class:`RgbToYuv422`,
  :func:`yuv422_to_rgb`, :class:`Yuv422ToRgb`.
- Sepia tone: :func:`sepia`, :class:`Sepia`.
- Sample an RGB image into a Bayer mosaic: :func:`rgb_to_raw`, :class:`RgbToRaw`; demosaic it: :func:`raw_to_rgb`,
  :class:`RawToRgb`; demosaic to half the size: :func:`raw_to_rgb_2x2_downscaled`, :class:`RawToRgb2x2Downscaled`.
  :class:`CFA` names the pattern of the mosaic.
- Color each channel of an image with a colormap: :func:`apply_colormap`, :class:`ApplyColorMap`. Build the palette
  with :class:`ColorMap`, pick a built-in one from :class:`ColorMapType`, and write your own RGB triples as
  :class:`RGBColor` lists.
"""

from .colormap import ApplyColorMap, ColorMap, ColorMapType, RGBColor, apply_colormap
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
