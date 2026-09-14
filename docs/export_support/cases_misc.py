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

"""ONNX export survey — area `misc`: everything in color/enhance/filters/losses/metrics/utils/morphology
not already covered by the first pass (survey.json)."""

import sys

import torch
from harness import case, run_cases

import kornia as K
import kornia.color as KC
import kornia.enhance as KE
import kornia.filters as KF
import kornia.losses as KL
import kornia.metrics as KM
from kornia.color import CFA, ColorMap, ColorMapType
from kornia.enhance import ThresholdType
from kornia.utils import torch_meshgrid as _torch_meshgrid

torch.manual_seed(0)

IMG = torch.rand(1, 3, 32, 40)
IMG2 = torch.rand(1, 3, 32, 40)
IMGB = torch.rand(2, 3, 48, 64)
GRAY = torch.rand(1, 1, 32, 40)
RGBA = torch.rand(1, 4, 32, 40)
VOL = torch.rand(1, 1, 8, 16, 16)
VOL3 = torch.rand(1, 3, 8, 16, 16)
RAW = torch.rand(1, 1, 32, 40)
Y420 = torch.rand(1, 1, 32, 40)
UV420 = torch.rand(1, 2, 16, 20)
UV422 = torch.rand(1, 2, 32, 20)
NORMALS = torch.nn.functional.normalize(torch.randn(1, 3, 32, 40), dim=1)
IMG255 = torch.rand(1, 3, 32, 40) * 255
GRAYINT = (torch.rand(1, 1, 32, 40) * 255).long()

CASES = []


def add(*a, **k):
    CASES.append(case(*a, **k))


# ------------------------------------------------------------------------------------------------ color
g = "color"
# module forms of the already-surveyed functions
for name, mod, inp in [
    ("RgbToGrayscale", KC.RgbToGrayscale(), IMG),
    ("BgrToGrayscale", KC.BgrToGrayscale(), IMG),
    ("GrayscaleToRgb", KC.GrayscaleToRgb(), GRAY),
    ("RgbToBgr", KC.RgbToBgr(), IMG),
    ("BgrToRgb", KC.BgrToRgb(), IMG),
    ("RgbToRgba", KC.RgbToRgba(1.0), IMG),
    ("BgrToRgba", KC.BgrToRgba(1.0), IMG),
    ("RgbaToRgb", KC.RgbaToRgb(), RGBA),
    ("RgbaToBgr", KC.RgbaToBgr(), RGBA),
    ("RgbToLinearRgb", KC.RgbToLinearRgb(), IMG),
    ("LinearRgbToRgb", KC.LinearRgbToRgb(), IMG),
    ("RgbToHsv", KC.RgbToHsv(), IMG),
    ("HsvToRgb", KC.HsvToRgb(), IMG),
    ("RgbToHls", KC.RgbToHls(), IMG),
    ("HlsToRgb", KC.HlsToRgb(), IMG),
    ("RgbToLab", KC.RgbToLab(), IMG),
    ("RgbToLuv", KC.RgbToLuv(), IMG),
    ("RgbToXyz", KC.RgbToXyz(), IMG),
    ("XyzToRgb", KC.XyzToRgb(), IMG),
    ("RgbToYcbcr", KC.RgbToYcbcr(), IMG),
    ("YcbcrToRgb", KC.YcbcrToRgb(), IMG),
    ("RgbToYuv", KC.RgbToYuv(), IMG),
    ("YuvToRgb", KC.YuvToRgb(), IMG),
    ("RgbToYuv420", KC.RgbToYuv420(), IMG),
    ("RgbToYuv422", KC.RgbToYuv422(), IMG),
    ("Sepia", KC.Sepia(), IMG),
    ("RgbToRgb255", KC.RgbToRgb255(), IMG),
    ("Rgb255ToRgb", KC.Rgb255ToRgb(), IMG255),
    ("NormalsToRgb255", KC.NormalsToRgb255(), NORMALS),
    ("Rgb255ToNormals", KC.Rgb255ToNormals(), IMG255),
    ("RawToRgb", KC.RawToRgb(CFA.BG), RAW),
    ("RgbToRaw", KC.RgbToRaw(CFA.BG), IMG),
    ("RawToRgb2x2Downscaled", KC.RawToRgb2x2Downscaled(CFA.BG), RAW),
]:
    add(f"color.{name}", g, mod, [inp])
add(
    "color.RgbToGrayscale[weights]",
    g,
    KC.RgbToGrayscale(rgb_weights=torch.tensor([0.2, 0.5, 0.3])),
    [IMG],
    note="rgb_weights given at construction -> constant",
)
add("color.LabToRgb", g, KC.LabToRgb(), [KC.rgb_to_lab(IMG)])
add("color.LuvToRgb", g, KC.LuvToRgb(), [KC.rgb_to_luv(IMG)])
add("color.Yuv420ToRgb", g, KC.Yuv420ToRgb(), [Y420, UV420 - 0.5])
add("color.Yuv422ToRgb", g, KC.Yuv422ToRgb(), [Y420, UV422 - 0.5])
# functions not in the first pass
add("color.bgr_to_grayscale", g, KC.bgr_to_grayscale, [IMG])
add("color.rgb_to_y", g, KC.rgb_to_y, [IMG])
add("color.bgr_to_rgba", g, KC.bgr_to_rgba, [IMG], {"alpha_val": 1.0}, note="alpha baked")
add("color.bgr_to_rgba[tensor_alpha]", g, KC.bgr_to_rgba, [IMG, torch.rand(1, 1, 32, 40)], note="alpha as live tensor")
add("color.rgba_to_rgb", g, KC.rgba_to_rgb, [RGBA])
add(
    "color.rgba_to_rgb[bg]",
    g,
    KC.rgba_to_rgb,
    [RGBA, torch.tensor([0.0, 0.0, 1.0]).view(1, 3, 1, 1)],
    note="background color live tensor (1,3,1,1)",
)
add(
    "color.rgba_to_rgb[bg_tuple]",
    g,
    KC.rgba_to_rgb,
    [RGBA],
    {"background_color": (0.0, 0.0, 1.0)},
    note="tuple bg baked",
)
add("color.rgba_to_bgr", g, KC.rgba_to_bgr, [RGBA])
add("color.rgb_to_rgb255", g, KC.rgb_to_rgb255, [IMG])
add("color.rgb255_to_rgb", g, KC.rgb255_to_rgb, [IMG255])
add("color.normals_to_rgb255", g, KC.normals_to_rgb255, [NORMALS])
add("color.rgb255_to_normals", g, KC.rgb255_to_normals, [IMG255])
add("color.sepia_from_rgb", g, KC.sepia_from_rgb, [IMG])
add("color.sepia_from_rgb[no_rescale]", g, KC.sepia_from_rgb, [IMG], {"rescale": False})
add("color.rgb_to_yuv420", g, KC.rgb_to_yuv420, [IMG], note="returns (y, uv)")
add("color.rgb_to_yuv422", g, KC.rgb_to_yuv422, [IMG], note="returns (y, uv)")
add("color.yuv420_to_rgb", g, KC.yuv420_to_rgb, [Y420, UV420 - 0.5])
add("color.yuv422_to_rgb", g, KC.yuv422_to_rgb, [Y420, UV422 - 0.5])
for cfa in CFA:
    add(f"color.raw_to_rgb[{cfa.name}]", g, KC.raw_to_rgb, [RAW], {"cfa": cfa}, note="cfa baked")
add("color.raw_to_rgb[batch]", g, KC.raw_to_rgb, [torch.rand(2, 1, 48, 64)], {"cfa": CFA.RG}, tags=("batch>1",))
add("color.rgb_to_raw", g, KC.rgb_to_raw, [IMG], {"cfa": CFA.BG})
add("color.rgb_to_raw[GR]", g, KC.rgb_to_raw, [IMG], {"cfa": CFA.GR})
add("color.raw_to_rgb_2x2_downscaled", g, KC.raw_to_rgb_2x2_downscaled, [RAW], {"cfa": CFA.BG})
add("color.raw_to_rgb_2x2_downscaled[RG]", g, KC.raw_to_rgb_2x2_downscaled, [RAW], {"cfa": CFA.RG})
add("color.rgb_to_lab[batch]", g, KC.rgb_to_lab, [IMGB], tags=("batch>1",))
add("color.rgb_to_hsv[eps]", g, KC.rgb_to_hsv, [IMG], {"eps": 1e-8})
add("color.lab_to_rgb[noclip]", g, KC.lab_to_rgb, [KC.rgb_to_lab(IMG)], {"clip": False})
# colormap
cm_vir = ColorMap(base=ColorMapType.viridis, num_colors=64)
cm_jet = ColorMap(base="jet", num_colors=16)
add("color.ApplyColorMap", g, KC.ApplyColorMap(cm_vir), [GRAY], note="viridis/64")
add("color.ApplyColorMap[jet16,int]", g, KC.ApplyColorMap(cm_jet), [GRAYINT], note="jet/16, int64 input 0..255")
add(
    "color.apply_colormap[autumn]",
    g,
    KC.apply_colormap,
    [GRAY],
    {"colormap": ColorMap(base=ColorMapType.autumn)},
    note="colormap object baked",
)
add(
    "color.apply_colormap[custom_rgb]",
    g,
    KC.apply_colormap,
    [GRAY],
    {"colormap": ColorMap(base=[[0.0, 0.5, 1.0], [1.0, 0.5, 0.0]], num_colors=8)},
    note="custom 2-stop map, 8 colors",
)
add(
    "color.apply_colormap[batch]",
    g,
    KC.apply_colormap,
    [torch.rand(2, 1, 48, 64)],
    {"colormap": ColorMap(base=ColorMapType.turbo)},
    tags=("batch>1",),
)
add(
    "color.ColorMap",
    g,
    None,
    [],
    skip="ColorMap is a lookup-table container (constructor only), covered via apply_colormap",
)
add("color.ColorMapType", g, None, [], skip="Enum")
add("color.CFA", g, None, [], skip="Enum")
add("color.RGBColor", g, None, [], skip="type alias")

# ------------------------------------------------------------------------------------------------ enhance
g = "enhance"
add("enhance.Normalize", g, KE.Normalize(mean=torch.tensor([0.5, 0.4, 0.3]), std=torch.tensor([0.2, 0.25, 0.3])), [IMG])
add("enhance.Normalize[float]", g, KE.Normalize(mean=0.5, std=0.2), [IMG])
add(
    "enhance.Denormalize",
    g,
    KE.Denormalize(mean=torch.tensor([0.5, 0.4, 0.3]), std=torch.tensor([0.2, 0.25, 0.3])),
    [IMG],
)
add(
    "enhance.normalize[live_stats]",
    g,
    KE.normalize,
    [IMG, torch.tensor([0.5, 0.4, 0.3]), torch.tensor([0.2, 0.25, 0.3])],
    note="mean/std as live tensors",
)
add(
    "enhance.denormalize[live_stats]",
    g,
    KE.denormalize,
    [IMG, torch.tensor([0.5, 0.4, 0.3]), torch.tensor([0.2, 0.25, 0.3])],
    note="mean/std as live tensors",
)
add("enhance.AddWeighted", g, KE.AddWeighted(0.6, 0.4, 0.1), [IMG, IMG2])
add(
    "enhance.AddWeighted[tensor]",
    g,
    KE.AddWeighted(torch.full((1, 3, 32, 40), 0.6), torch.full((1, 3, 32, 40), 0.4), torch.full((1, 3, 32, 40), 0.1)),
    [IMG, IMG2],
    note="tensor alpha/beta/gamma must equal src shape (kornia check); constructed -> constants",
)
add(
    "enhance.add_weighted[live_weights]",
    g,
    KE.add_weighted,
    [IMG, torch.full((1, 3, 32, 40), 0.6), IMG2, torch.full((1, 3, 32, 40), 0.4), torch.full((1, 3, 32, 40), 0.1)],
    note="alpha/beta/gamma live tensors (must be full src shape)",
)
add("enhance.AdjustBrightness", g, KE.AdjustBrightness(0.2), [IMG])
add("enhance.AdjustBrightnessAccumulative", g, KE.AdjustBrightnessAccumulative(0.2), [IMG])
add("enhance.adjust_brightness_accumulative", g, KE.adjust_brightness_accumulative, [IMG], {"factor": 0.2})
add(
    "enhance.adjust_brightness_accumulative[live]",
    g,
    KE.adjust_brightness_accumulative,
    [IMG, torch.tensor([0.2])],
    note="factor live",
)
add("enhance.AdjustContrast", g, KE.AdjustContrast(0.8), [IMG])
add("enhance.AdjustContrastWithMeanSubtraction", g, KE.AdjustContrastWithMeanSubtraction(0.8), [IMG])
add("enhance.AdjustGamma", g, KE.AdjustGamma(1.5, 1.2), [IMG])
add("enhance.AdjustHue", g, KE.AdjustHue(0.5), [IMG])
add("enhance.AdjustSaturation", g, KE.AdjustSaturation(1.5), [IMG])
add("enhance.AdjustSaturationWithGraySubtraction", g, KE.AdjustSaturationWithGraySubtraction(1.5), [IMG])
add(
    "enhance.adjust_saturation_with_gray_subtraction",
    g,
    KE.adjust_saturation_with_gray_subtraction,
    [IMG],
    {"factor": 1.5},
)
add(
    "enhance.adjust_saturation_with_gray_subtraction[live]",
    g,
    KE.adjust_saturation_with_gray_subtraction,
    [IMG, torch.tensor([1.5])],
    note="factor live",
)
add(
    "enhance.adjust_saturation_raw",
    g,
    KE.adjust_saturation_raw,
    [KC.rgb_to_hsv(IMG)],
    {"factor": 1.5},
    note="HSV input",
)
add("enhance.adjust_hue_raw", g, KE.adjust_hue_raw, [KC.rgb_to_hsv(IMG)], {"factor": 0.5}, note="HSV input")
add("enhance.adjust_hue_raw[live]", g, KE.adjust_hue_raw, [KC.rgb_to_hsv(IMG), torch.tensor([0.5])], note="factor live")
add("enhance.AdjustLog", g, KE.AdjustLog(gain=1.0), [IMG])
add("enhance.AdjustLog[inv]", g, KE.AdjustLog(gain=1.0, inv=True), [IMG])
add("enhance.AdjustSigmoid", g, KE.AdjustSigmoid(), [IMG])
add("enhance.AdjustSigmoid[inv]", g, KE.AdjustSigmoid(inv=True), [IMG])
add("enhance.Invert", g, KE.Invert(), [IMG])
add("enhance.Invert[max_val]", g, KE.Invert(max_val=torch.tensor(255.0)), [IMG255])
add("enhance.Rescale", g, KE.Rescale(2.0), [IMG])
add("enhance.Threshold", g, KE.Threshold(0.5, 1.0), [IMG], note="THRESH_BINARY")
add("enhance.threshold", g, KE.threshold, [IMG], {"thresh": 0.5, "maxval": 1.0})
add("enhance.threshold[live]", g, KE.threshold, [IMG, torch.tensor(0.5), torch.tensor(1.0)], note="thresh/maxval live")
for t in ThresholdType:
    if t.name == "THRESH_BINARY":
        continue
    add(
        f"enhance.threshold[{t.name}]",
        g,
        KE.threshold,
        [IMG],
        {"thresh": 0.5, "maxval": 1.0, "type": t},
        note="THRESH_OTSU raises NotImplementedError in kornia (eager-fail is kornia's, not the spec)"
        if t.name == "THRESH_OTSU"
        else "",
    )
add("enhance.ThresholdType", g, None, [], skip="Enum")
add("enhance.IntegralImage", g, KE.IntegralImage(), [IMG])
add("enhance.IntegralTensor", g, KE.IntegralTensor((-2, -1)), [IMG])
add("enhance.integral_image", g, KE.integral_image, [IMG])
add("enhance.integral_tensor", g, KE.integral_tensor, [IMG], {"dim": (-2, -1)})
add("enhance.integral_tensor[3d]", g, KE.integral_tensor, [VOL], {"dim": (-3, -2, -1)}, tags=("3d",))
add("enhance.shift_rgb", g, KE.shift_rgb, [IMG, torch.tensor([0.1]), torch.tensor([-0.1]), torch.tensor([0.05])])
add(
    "enhance.shift_rgb[batch]",
    g,
    KE.shift_rgb,
    [IMGB, torch.tensor([0.1, 0.2]), torch.tensor([-0.1, 0.0]), torch.tensor([0.05, 0.1])],
    tags=("batch>1",),
)
add("enhance.equalize3d", g, KE.equalize3d, [VOL3], tags=("3d",))
add("enhance.equalize[batch]", g, KE.equalize, [IMGB], tags=("batch>1",))
add(
    "enhance.histogram",
    g,
    KE.histogram,
    [torch.rand(1, 64), torch.linspace(0, 1, 32), torch.tensor(0.05)],
    note="bins and bandwidth live tensors",
)
add(
    "enhance.histogram2d",
    g,
    KE.histogram2d,
    [torch.rand(2, 64), torch.rand(2, 64), torch.linspace(0, 1, 16), torch.tensor(0.05)],
    note="bins/bandwidth live",
)
add(
    "enhance.image_histogram2d",
    g,
    KE.image_histogram2d,
    [IMG],
    {"min": 0.0, "max": 1.0, "n_bins": 32},
    note="n_bins baked; returns (hist, pdf-or-zeros)",
)
add(
    "enhance.image_histogram2d[pdf,gaussian]",
    g,
    KE.image_histogram2d,
    [IMG],
    {"min": 0.0, "max": 1.0, "n_bins": 32, "return_pdf": True, "kernel": "gaussian"},
)
add(
    "enhance.image_histogram2d[centers]",
    g,
    KE.image_histogram2d,
    [
        IMG,
    ],
    {"min": 0.0, "max": 1.0, "n_bins": 16, "centers": torch.linspace(0, 1, 16), "kernel": "epanechnikov"},
    note="centers tensor passed as kwarg -> constant",
)
add("enhance.normalize_min_max[range]", g, KE.normalize_min_max, [IMG], {"min_val": -1.0, "max_val": 1.0})
add(
    "enhance.linear_transform",
    g,
    KE.linear_transform,
    [torch.rand(8, 3, 4, 5), torch.eye(60) * 0.5 + 0.01, torch.rand(1, 60)],
    note="transform matrix/mean live",
)
add("enhance.zca_mean", g, KE.zca_mean, [torch.rand(16, 3, 4, 5)], note="returns (mean, T, None)")
add(
    "enhance.zca_mean[inverse]",
    g,
    KE.zca_mean,
    [torch.rand(16, 3, 4, 5)],
    {"return_inverse": True},
    note="returns (mean, T, T_inv)",
)
add("enhance.zca_whiten", g, KE.zca_whiten, [torch.rand(16, 3, 4, 5)])
_zca_data = torch.rand(16, 3, 4, 5)
_zca = KE.ZCAWhitening().fit(_zca_data)
add(
    "enhance.ZCAWhitening[transform]",
    g,
    _zca,
    [_zca_data[:4]],
    note="fit() done eagerly; exports transform with fitted stats",
)
_zca_inv = KE.ZCAWhitening(compute_inv=True).fit(_zca_data)
add(
    "enhance.ZCAWhitening[inverse_transform]",
    g,
    _zca_inv,
    [_zca_inv(_zca_data[:4])],
    method="inverse_transform",
    note="compute_inv=True",
)
add(
    "enhance.ZCAWhitening[include_fit]",
    g,
    KE.ZCAWhitening(),
    [_zca_data],
    {"include_fit": True},
    note="fit inside graph (SVD)",
)
add(
    "enhance.JPEGCodecDifferentiable",
    g,
    KE.JPEGCodecDifferentiable(),
    [torch.rand(1, 3, 32, 48), torch.tensor([50.0])],
    note="jpeg_quality live; H,W multiple of 16",
    atol=2e-3,
)
add(
    "enhance.JPEGCodecDifferentiable[batch]",
    g,
    KE.JPEGCodecDifferentiable(),
    [torch.rand(2, 3, 48, 64), torch.tensor([99.0, 10.0])],
    tags=("batch>1",),
    atol=2e-3,
)
add(
    "enhance.JPEGCodecDifferentiable[custom_qt]",
    g,
    KE.JPEGCodecDifferentiable(
        quantization_table_y=torch.randint(1, 100, (8, 8)).float(),
        quantization_table_c=torch.randint(1, 100, (8, 8)).float(),
    ),
    [torch.rand(1, 3, 32, 48), torch.tensor([50.0])],
    atol=2e-3,
)

# ------------------------------------------------------------------------------------------------ filters
g = "filters"
KER3 = torch.ones(1, 3, 3, 3) / 27
KER2 = torch.rand(1, 3, 5)
add("filters.filter3d", g, KF.filter3d, [VOL, KER3], tags=("3d",))
add(
    "filters.filter3d[reflect,conv]",
    g,
    KF.filter3d,
    [VOL, torch.rand(1, 3, 3, 3)],
    {"border_type": "reflect", "behaviour": "conv"},
    tags=("3d",),
)
add(
    "filters.filter3d[constant,normalized]",
    g,
    KF.filter3d,
    [VOL3, torch.rand(1, 3, 3, 3)],
    {"border_type": "constant", "normalized": True},
    tags=("3d",),
)
add("filters.filter3d[circular]", g, KF.filter3d, [VOL, KER3], {"border_type": "circular"}, tags=("3d",))
add(
    "filters.filter3d[batch_kernels]",
    g,
    KF.filter3d,
    [torch.rand(2, 1, 8, 16, 16), torch.rand(2, 3, 3, 3)],
    tags=("3d", "batch>1"),
    note="per-sample kernel B=2",
)
add("filters.convolve3d", g, KF.convolve3d, [VOL, torch.rand(1, 3, 3, 3)], tags=("3d",))
add("filters.correlate3d", g, KF.correlate3d, [VOL, torch.rand(1, 3, 3, 3)], tags=("3d",))
add("filters.convolve2d", g, KF.convolve2d, [IMG, KER2])
add("filters.convolve2d[valid]", g, KF.convolve2d, [IMG, KER2], {"padding": "valid"})
add("filters.correlate2d", g, KF.correlate2d, [IMG, KER2])
add("filters.fft_conv", g, KF.fft_conv, [IMG, torch.rand(1, 5, 5)], note="corr, reflect, same")
add(
    "filters.fft_conv[conv,valid]",
    g,
    KF.fft_conv,
    [IMG, torch.rand(1, 5, 5)],
    {"behaviour": "conv", "padding": "valid"},
)
add(
    "filters.fft_conv[normalized]",
    g,
    KF.fft_conv,
    [IMG, torch.rand(1, 5, 5)],
    {"normalized": True, "border_type": "replicate"},
)
add("filters.filter2d_separable", g, KF.filter2d_separable, [IMG, torch.rand(1, 5), torch.rand(1, 3)])
add(
    "filters.filter2d_separable[batch_kernels]",
    g,
    KF.filter2d_separable,
    [IMGB, torch.rand(2, 5), torch.rand(2, 3)],
    tags=("batch>1",),
)
for bt in ("constant", "replicate", "circular"):
    add(f"filters.filter2d[{bt}]", g, KF.filter2d, [IMG, torch.rand(1, 3, 5)], {"border_type": bt})
add("filters.filter2d[valid]", g, KF.filter2d, [IMG, torch.rand(1, 3, 5)], {"padding": "valid"})
add("filters.filter2d[normalized]", g, KF.filter2d, [IMG, torch.rand(1, 3, 5)], {"normalized": True})
add("filters.filter2d[batch_kernels]", g, KF.filter2d, [IMGB, torch.rand(2, 3, 5)], tags=("batch>1",))
add("filters.filter2d[even_kernel]", g, KF.filter2d, [IMG, torch.rand(1, 4, 4)], note="even kernel size")
add("filters.filter2d[conv]", g, KF.filter2d, [IMG, torch.rand(1, 3, 5)], {"behaviour": "conv"})
add(
    "filters.gaussian_blur2d[tensor_sigma]",
    g,
    lambda x, s: KF.gaussian_blur2d(x, (5, 5), s),
    [IMG, torch.tensor([[1.5, 2.0]])],
    note="sigma live tensor (B,2); kernel_size baked",
)
add(
    "filters.gaussian_blur2d[nonseparable]",
    g,
    KF.gaussian_blur2d,
    [IMG],
    {"kernel_size": (5, 3), "sigma": (1.5, 1.0), "separable": False},
)
add(
    "filters.gaussian_blur2d[tensor_sigma,batch]",
    g,
    lambda x, s: KF.gaussian_blur2d(x, (5, 5), s),
    [IMGB, torch.tensor([[1.5, 2.0], [0.7, 0.9]])],
    tags=("batch>1",),
)
add(
    "filters.gaussian_blur2d[tensor_sigma,nonsep]",
    g,
    lambda x, s: KF.gaussian_blur2d(x, (5, 5), s, separable=False),
    [IMG, torch.tensor([[1.5, 2.0]])],
)
add("filters.gaussian_blur2d_t", g, None, [], skip="deprecated alias of gaussian_blur2d")
add("filters.get_gaussian_kernel1d_t", g, None, [], skip="deprecated alias of get_gaussian_kernel1d")
add("filters.get_gaussian_kernel2d_t", g, None, [], skip="deprecated alias of get_gaussian_kernel2d")
add("filters.get_gaussian_kernel3d_t", g, None, [], skip="deprecated alias of get_gaussian_kernel3d")
add(
    "filters.gaussian",
    g,
    lambda s: KF.gaussian(5, s),
    [torch.tensor([[1.5]])],
    note="sigma live (1,1); window_size baked",
)
add("filters.gaussian[mean]", g, lambda s: KF.gaussian(5, s, mean=0.5), [torch.tensor([[1.5]])])
add(
    "filters.get_gaussian_kernel1d",
    g,
    lambda s: KF.get_gaussian_kernel1d(5, s),
    [torch.tensor([[1.5]])],
    note="sigma live",
)
add(
    "filters.get_gaussian_kernel1d[even]",
    g,
    lambda s: KF.get_gaussian_kernel1d(4, s, force_even=True),
    [torch.tensor([[1.5]])],
)
add(
    "filters.get_gaussian_kernel1d[float_sigma]",
    g,
    lambda d: KF.get_gaussian_kernel1d(5, 1.5) + d * 0,
    [torch.zeros(1)],
    note="all-constant graph; dummy input",
)
add(
    "filters.get_gaussian_discrete_kernel1d",
    g,
    lambda s: KF.get_gaussian_discrete_kernel1d(5, s),
    [torch.tensor([[1.5]])],
    note="sigma live",
)
add(
    "filters.get_gaussian_erf_kernel1d",
    g,
    lambda s: KF.get_gaussian_erf_kernel1d(5, s),
    [torch.tensor([[1.5]])],
    note="sigma live",
)
add(
    "filters.get_gaussian_kernel2d",
    g,
    lambda s: KF.get_gaussian_kernel2d((5, 3), s),
    [torch.tensor([[1.5, 2.0]])],
    note="sigma live",
)
add(
    "filters.get_gaussian_kernel2d[batch]",
    g,
    lambda s: KF.get_gaussian_kernel2d((5, 5), s),
    [torch.tensor([[1.5, 2.0], [0.5, 0.9]])],
    tags=("batch>1",),
)
add(
    "filters.get_gaussian_kernel3d",
    g,
    lambda s: KF.get_gaussian_kernel3d((3, 5, 3), s),
    [torch.tensor([[1.5, 2.0, 1.0]])],
    tags=("3d",),
    note="sigma live",
)
add(
    "filters.get_hanning_kernel1d",
    g,
    lambda d: KF.get_hanning_kernel1d(9) + d * 0,
    [torch.zeros(1)],
    note="constant graph",
)
add(
    "filters.get_hanning_kernel2d",
    g,
    lambda d: KF.get_hanning_kernel2d((5, 7)) + d * 0,
    [torch.zeros(1)],
    note="constant graph",
)
add(
    "filters.get_laplacian_kernel1d",
    g,
    lambda d: KF.get_laplacian_kernel1d(5) + d * 0,
    [torch.zeros(1)],
    note="constant graph",
)
add(
    "filters.get_laplacian_kernel2d",
    g,
    lambda d: KF.get_laplacian_kernel2d((5, 3)) + d * 0,
    [torch.zeros(1)],
    note="constant graph",
)
add("filters.laplacian_1d", g, lambda d: KF.laplacian_1d(5) + d * 0, [torch.zeros(1)], note="constant graph")
add("filters.get_box_kernel1d", g, lambda d: KF.get_box_kernel1d(5) + d * 0, [torch.zeros(1)], note="constant graph")
add(
    "filters.get_box_kernel2d",
    g,
    lambda d: KF.get_box_kernel2d((3, 5)) + d * 0,
    [torch.zeros(1)],
    note="constant graph",
)
add(
    "filters.get_binary_kernel2d",
    g,
    lambda d: KF.get_binary_kernel2d((3, 3)) + d * 0,
    [torch.zeros(1)],
    note="constant graph",
)
add("filters.get_diff_kernel2d", g, lambda d: KF.get_diff_kernel2d() + d * 0, [torch.zeros(1)], note="constant graph")
add("filters.get_sobel_kernel2d", g, lambda d: KF.get_sobel_kernel2d() + d * 0, [torch.zeros(1)], note="constant graph")
add(
    "filters.get_spatial_gradient_kernel2d[sobel,2]",
    g,
    lambda d: KF.get_spatial_gradient_kernel2d("sobel", 2) + d * 0,
    [torch.zeros(1)],
    note="constant graph",
)
add(
    "filters.get_spatial_gradient_kernel2d[diff,1]",
    g,
    lambda d: KF.get_spatial_gradient_kernel2d("diff", 1) + d * 0,
    [torch.zeros(1)],
    note="constant graph",
)
add(
    "filters.get_spatial_gradient_kernel3d[diff,1]",
    g,
    lambda d: KF.get_spatial_gradient_kernel3d("diff", 1) + d * 0,
    [torch.zeros(1)],
    note="constant graph",
    tags=("3d",),
)
add(
    "filters.get_spatial_gradient_kernel3d[diff,2]",
    g,
    lambda d: KF.get_spatial_gradient_kernel3d("diff", 2) + d * 0,
    [torch.zeros(1)],
    note="constant graph",
    tags=("3d",),
)
add(
    "filters.get_motion_kernel2d",
    g,
    lambda a, d: KF.get_motion_kernel2d(5, a, d),
    [torch.tensor([30.0]), torch.tensor([0.5])],
    note="angle/direction live; kernel_size baked",
)
add(
    "filters.get_motion_kernel2d[bilinear]",
    g,
    lambda a, d: KF.get_motion_kernel2d(5, a, d, mode="bilinear"),
    [torch.tensor([30.0]), torch.tensor([0.5])],
)
add(
    "filters.get_motion_kernel2d[float]",
    g,
    lambda d: KF.get_motion_kernel2d(5, 30.0, 0.5) + d * 0,
    [torch.zeros(1)],
    note="constant graph",
)
add(
    "filters.get_motion_kernel3d",
    g,
    lambda a, d: KF.get_motion_kernel3d(5, a, d),
    [torch.tensor([[30.0, 10.0, 20.0]]), torch.tensor([0.5])],
    tags=("3d",),
    note="angle/direction live",
)
add(
    "filters.get_motion_kernel3d[float]",
    g,
    lambda d: KF.get_motion_kernel3d(5, (30.0, 10.0, 20.0), 0.5) + d * 0,
    [torch.zeros(1)],
    tags=("3d",),
    note="constant graph",
)
add(
    "filters.motion_blur3d",
    g,
    KF.motion_blur3d,
    [VOL],
    {"kernel_size": 3, "angle": (30.0, 10.0, 20.0), "direction": 0.5},
    tags=("3d",),
)
add(
    "filters.motion_blur3d[live]",
    g,
    lambda x, a, d: KF.motion_blur3d(x, 3, a, d),
    [VOL, torch.tensor([[30.0, 10.0, 20.0]]), torch.tensor([0.5])],
    tags=("3d",),
    note="angle/direction live tensors",
)
add(
    "filters.motion_blur3d[bilinear,reflect]",
    g,
    KF.motion_blur3d,
    [VOL],
    {"kernel_size": 3, "angle": (30.0, 10.0, 20.0), "direction": 0.5, "mode": "bilinear", "border_type": "reflect"},
    tags=("3d",),
)
add("filters.MotionBlur3D", g, KF.MotionBlur3D(3, (30.0, 10.0, 20.0), 0.5), [VOL], tags=("3d",))
add("filters.MotionBlur", g, KF.MotionBlur(5, 30.0, 0.5), [IMG])
add(
    "filters.motion_blur[live]",
    g,
    lambda x, a, d: KF.motion_blur(x, 5, a, d),
    [IMG, torch.tensor([30.0]), torch.tensor([0.5])],
    note="angle/direction live tensors",
)
add("filters.spatial_gradient3d", g, KF.spatial_gradient3d, [VOL], tags=("3d",))
add("filters.spatial_gradient3d[order2]", g, KF.spatial_gradient3d, [VOL], {"order": 2}, tags=("3d",))
add("filters.SpatialGradient3d", g, KF.SpatialGradient3d(), [VOL3], tags=("3d",))
add("filters.spatial_gradient[order2]", g, KF.spatial_gradient, [IMG], {"order": 2})
add("filters.spatial_gradient[diff]", g, KF.spatial_gradient, [IMG], {"mode": "diff"})
add("filters.spatial_gradient[diff,order2]", g, KF.spatial_gradient, [IMG], {"mode": "diff", "order": 2})
add("filters.spatial_gradient[unnormalized]", g, KF.spatial_gradient, [IMG], {"normalized": False})
add("filters.SpatialGradient", g, KF.SpatialGradient(), [IMG])
add("filters.SpatialGradient[diff,2]", g, KF.SpatialGradient(mode="diff", order=2), [IMG])
add("filters.Sobel", g, KF.Sobel(), [IMG])
add("filters.Sobel[unnormalized]", g, KF.Sobel(normalized=False), [IMG])
add("filters.Laplacian", g, KF.Laplacian(5), [IMG])
add("filters.Laplacian[unnormalized,replicate]", g, KF.Laplacian(3, border_type="replicate", normalized=False), [IMG])
add("filters.BoxBlur", g, KF.BoxBlur((3, 5)), [IMG])
add("filters.BoxBlur[separable]", g, KF.BoxBlur((3, 5), separable=True), [IMG])
add("filters.GaussianBlur2d", g, KF.GaussianBlur2d((5, 5), (1.5, 1.5)), [IMG])
add(
    "filters.GaussianBlur2d[tensor_sigma]",
    g,
    KF.GaussianBlur2d((5, 5), torch.tensor([[1.5, 2.0]])),
    [IMG],
    note="tensor sigma fixed at construction -> constant",
)
add("filters.GaussianBlur2d[nonsep]", g, KF.GaussianBlur2d((5, 3), (1.5, 1.0), separable=False), [IMG])
add("filters.MedianBlur", g, KF.MedianBlur((3, 3)), [IMG])
add("filters.MedianBlur[5x3]", g, KF.MedianBlur((5, 3)), [IMG])
add("filters.median_blur[batch]", g, KF.median_blur, [IMGB], {"kernel_size": (3, 3)}, tags=("batch>1",))
add("filters.UnsharpMask", g, KF.UnsharpMask((5, 5), (1.5, 1.5)), [IMG])
add("filters.BilateralBlur", g, KF.BilateralBlur((5, 5), 0.1, (1.5, 1.5)), [IMG])
add("filters.BilateralBlur[l2]", g, KF.BilateralBlur((5, 5), 0.1, (1.5, 1.5), color_distance_type="l2"), [IMG])
add(
    "filters.bilateral_blur[live_sigma]",
    g,
    lambda x, sc, ss: KF.bilateral_blur(x, (5, 5), sc, ss),
    [IMG, torch.tensor([0.1]), torch.tensor([[1.5, 1.5]])],
    note="sigma_color/sigma_space live",
)
add("filters.JointBilateralBlur", g, KF.JointBilateralBlur((5, 5), 0.1, (1.5, 1.5)), [IMG, IMG2])
add("filters.GuidedBlur", g, KF.GuidedBlur((5, 5), 0.01), [IMG, IMG2], note="known problematic")
add("filters.GuidedBlur[subsample2]", g, KF.GuidedBlur((5, 5), 0.01, subsample=2), [IMG, IMG2])
add(
    "filters.guided_blur[gray_guidance]",
    g,
    KF.guided_blur,
    [GRAY, IMG],
    {"kernel_size": (5, 5), "eps": 0.01},
    note="single-channel guidance",
)
add(
    "filters.guided_blur[eps_tensor]",
    g,
    lambda gd, x, e: KF.guided_blur(gd, x, (5, 5), e),
    [IMG, IMG2, torch.tensor([0.01])],
    note="eps live",
)
add("filters.Canny", g, KF.Canny(), [IMG], note="known fail (data-dependent hysteresis loop)")
add("filters.Canny[no_hysteresis]", g, KF.Canny(hysteresis=False), [IMG])
add("filters.canny[no_hysteresis]", g, KF.canny, [IMG], {"hysteresis": False})
add("filters.canny[gray,no_hysteresis]", g, KF.canny, [GRAY], {"hysteresis": False})
add("filters.BlurPool2D", g, KF.BlurPool2D(3), [IMG])
add("filters.BlurPool2D[stride1]", g, KF.BlurPool2D(3, stride=1), [IMG])
add("filters.MaxBlurPool2D", g, KF.MaxBlurPool2D(3), [IMG])
add(
    "filters.MaxBlurPool2D[ceil]",
    g,
    KF.MaxBlurPool2D(3, stride=2, max_pool_size=3, ceil_mode=True),
    [torch.rand(1, 3, 33, 41)],
)
add("filters.EdgeAwareBlurPool2D", g, KF.EdgeAwareBlurPool2D(3), [IMG])
add("filters.edge_aware_blur_pool2d", g, KF.edge_aware_blur_pool2d, [IMG], {"kernel_size": 3})
add(
    "filters.edge_aware_blur_pool2d[k5]",
    g,
    KF.edge_aware_blur_pool2d,
    [IMGB],
    {"kernel_size": 5, "edge_threshold": 2.0},
    tags=("batch>1",),
)
add("filters.InRange", g, KF.InRange((0.2, 0.2, 0.2), (0.8, 0.8, 0.8)), [IMG])
add("filters.InRange[return_mask]", g, KF.InRange((0.2, 0.2, 0.2), (0.8, 0.8, 0.8), return_mask=True), [IMG])
add(
    "filters.in_range[live_bounds]",
    g,
    KF.in_range,
    [IMG, torch.tensor([[[0.2]], [[0.2]], [[0.2]]]).view(1, 3, 1, 1), torch.tensor([0.8, 0.8, 0.8]).view(1, 3, 1, 1)],
    note="lower/upper live tensors",
)
add(
    "filters.in_range[return_mask]",
    g,
    KF.in_range,
    [IMG],
    {"lower": (0.2, 0.2, 0.2), "upper": (0.8, 0.8, 0.8), "return_mask": True},
)
add("filters.otsu_threshold", g, KF.otsu_threshold, [IMG255], note="returns (thresholded, threshold)")
add("filters.otsu_threshold[differentiable]", g, KF.otsu_threshold, [IMG255], {"slow_and_differentiable": True})
add("filters.otsu_threshold[return_mask]", g, KF.otsu_threshold, [IMG255], {"return_mask": True})
add("filters.OtsuThreshold", g, KF.OtsuThreshold(), [IMG255])
add("filters.StableDiffusionDissolving", g, None, [], skip="weights > 500 MB (Stable Diffusion 1.5 via diffusers)")
add(
    "filters.laplacian[batch,rect]",
    g,
    KF.laplacian,
    [IMGB],
    {"kernel_size": (5, 3)},
    tags=("batch>1",),
    note="rectangular kernel (even sizes are rejected by kornia)",
)
add("filters.sobel[batch]", g, KF.sobel, [IMGB], tags=("batch>1",))
add("filters.box_blur[separable]", g, KF.box_blur, [IMG], {"kernel_size": (3, 5), "separable": True})
add(
    "filters.unsharp_mask[tensor_sigma]",
    g,
    lambda x, s: KF.unsharp_mask(x, (5, 5), s),
    [IMG, torch.tensor([[1.5, 1.5]])],
    note="sigma live",
)

# ------------------------------------------------------------------------------------------------ losses
g = "losses"
LOGITS = torch.randn(2, 4, 16, 20)
LABELS = torch.randint(0, 4, (2, 16, 20))
BIN_LOGITS = torch.randn(2, 1, 16, 20)
BIN_TARGET = (torch.rand(2, 1, 16, 20) > 0.5).float()
add("losses.ssim_loss", g, KL.ssim_loss, [IMG, IMG2], {"window_size": 5})
add(
    "losses.ssim_loss[valid,none]",
    g,
    KL.ssim_loss,
    [IMG, IMG2],
    {"window_size": 5, "padding": "valid", "reduction": "none"},
)
add("losses.SSIMLoss", g, KL.SSIMLoss(5), [IMG, IMG2])
add("losses.ssim3d_loss", g, KL.ssim3d_loss, [VOL, torch.rand(1, 1, 8, 16, 16)], {"window_size": 3}, tags=("3d",))
add("losses.SSIM3DLoss", g, KL.SSIM3DLoss(3), [VOL, torch.rand(1, 1, 8, 16, 16)], tags=("3d",))
add(
    "losses.MS_SSIMLoss",
    g,
    KL.MS_SSIMLoss(),
    [torch.rand(1, 3, 64, 80), torch.rand(1, 3, 64, 80)],
    note="default sigmas up to 8",
)
add("losses.MS_SSIMLoss[small_sigmas]", g, KL.MS_SSIMLoss(sigmas=(0.5, 1.0, 2.0)), [IMG, IMG2])
add("losses.psnr_loss", g, KL.psnr_loss, [IMG, IMG2], {"max_val": 1.0})
add("losses.PSNRLoss", g, KL.PSNRLoss(1.0), [IMG, IMG2])
add("losses.total_variation", g, KL.total_variation, [IMG])
add("losses.total_variation[mean]", g, KL.total_variation, [IMG], {"reduction": "mean"})
add("losses.TotalVariation", g, KL.TotalVariation(), [IMG])
add("losses.inverse_depth_smoothness_loss", g, KL.inverse_depth_smoothness_loss, [GRAY, IMG])
add("losses.InverseDepthSmoothnessLoss", g, KL.InverseDepthSmoothnessLoss(), [GRAY, IMG])
add("losses.dice_loss", g, KL.dice_loss, [LOGITS, LABELS])
add(
    "losses.dice_loss[macro,weight]",
    g,
    KL.dice_loss,
    [LOGITS, LABELS],
    {"average": "macro", "weight": torch.tensor([1.0, 2.0, 0.5, 1.0])},
)
add(
    "losses.dice_loss[ignore_index]",
    g,
    KL.dice_loss,
    [LOGITS, LABELS.clone().masked_fill(LABELS == 3, -100)],
    {"ignore_index": -100},
    note="labels contain -100",
)
add("losses.DiceLoss", g, KL.DiceLoss(), [LOGITS, LABELS])
add("losses.tversky_loss", g, KL.tversky_loss, [LOGITS, LABELS], {"alpha": 0.5, "beta": 0.5})
add("losses.TverskyLoss", g, KL.TverskyLoss(0.3, 0.7), [LOGITS, LABELS])
add("losses.focal_loss", g, KL.focal_loss, [LOGITS, LABELS], {"alpha": 0.5})
add(
    "losses.focal_loss[mean,weight]",
    g,
    KL.focal_loss,
    [LOGITS, LABELS],
    {"alpha": 0.5, "reduction": "mean", "weight": torch.tensor([1.0, 2.0, 0.5, 1.0])},
)
add(
    "losses.focal_loss[ignore_index]",
    g,
    KL.focal_loss,
    [LOGITS, LABELS.clone().masked_fill(LABELS == 3, -100)],
    {"alpha": None, "reduction": "mean"},
)
add("losses.FocalLoss", g, KL.FocalLoss(0.5, reduction="mean"), [LOGITS, LABELS])
add("losses.binary_focal_loss_with_logits", g, KL.binary_focal_loss_with_logits, [BIN_LOGITS, BIN_TARGET])
add(
    "losses.binary_focal_loss_with_logits[pos_weight,mean]",
    g,
    KL.binary_focal_loss_with_logits,
    [BIN_LOGITS, BIN_TARGET],
    {"reduction": "mean", "pos_weight": torch.tensor([2.0]), "weight": torch.tensor([1.5])},
)
add(
    "losses.BinaryFocalLossWithLogits",
    g,
    KL.BinaryFocalLossWithLogits(0.25, reduction="mean"),
    [BIN_LOGITS, BIN_TARGET],
)
add("losses.lovasz_hinge_loss", g, KL.lovasz_hinge_loss, [BIN_LOGITS, BIN_TARGET[:, 0].long()])
add("losses.LovaszHingeLoss", g, KL.LovaszHingeLoss(), [BIN_LOGITS, BIN_TARGET[:, 0].long()])
add("losses.lovasz_softmax_loss", g, KL.lovasz_softmax_loss, [LOGITS, LABELS])
add(
    "losses.lovasz_softmax_loss[weight]",
    g,
    KL.lovasz_softmax_loss,
    [LOGITS, LABELS],
    {"weight": torch.tensor([1.0, 2.0, 0.5, 1.0])},
)
add("losses.LovaszSoftmaxLoss", g, KL.LovaszSoftmaxLoss(), [LOGITS, LABELS])
add("losses.hausdorff_er_loss", g, KL.HausdorffERLoss(), [LOGITS, LABELS[:, None]], note="k=10 erosions")
add("losses.HausdorffERLoss[k3,sum]", g, KL.HausdorffERLoss(k=3, reduction="sum"), [LOGITS, LABELS[:, None]])
add(
    "losses.HausdorffERLoss3D",
    g,
    KL.HausdorffERLoss3D(k=3),
    [torch.randn(1, 3, 8, 16, 16), torch.randint(0, 3, (1, 1, 8, 16, 16))],
    tags=("3d",),
)
for nm, fn, cls in [
    ("charbonnier", KL.charbonnier_loss, KL.CharbonnierLoss),
    ("cauchy", KL.cauchy_loss, KL.CauchyLoss),
    ("geman_mcclure", KL.geman_mcclure_loss, KL.GemanMcclureLoss),
    ("welsch", KL.welsch_loss, KL.WelschLoss),
]:
    add(f"losses.{nm}_loss", g, fn, [IMG, IMG2])
    add(f"losses.{nm}_loss[mean]", g, fn, [IMG, IMG2], {"reduction": "mean"})
    add(f"losses.{cls.__name__}", g, cls(reduction="sum"), [IMG, IMG2])
PROB = torch.softmax(torch.randn(2, 3, 8, 10).flatten(-2), -1).view(2, 3, 8, 10)
PROB2 = torch.softmax(torch.randn(2, 3, 8, 10).flatten(-2), -1).view(2, 3, 8, 10)
add("losses.js_div_loss_2d", g, KL.js_div_loss_2d, [PROB, PROB2])
add("losses.js_div_loss_2d[none]", g, KL.js_div_loss_2d, [PROB, PROB2], {"reduction": "none"})
add("losses.kl_div_loss_2d", g, KL.kl_div_loss_2d, [PROB, PROB2])
add("losses.kl_div_loss_2d[sum]", g, KL.kl_div_loss_2d, [PROB, PROB2], {"reduction": "sum"})
SIG_A = torch.rand(2, 256)
SIG_B = (SIG_A + 0.1 * torch.randn(2, 256)).clamp(0, 1)
add("losses.mutual_information_loss", g, KL.mutual_information_loss, [SIG_A, SIG_B])
add(
    "losses.mutual_information_loss[rect]",
    g,
    KL.mutual_information_loss,
    [SIG_A, SIG_B],
    {"kernel_function": KL.MIKernel.rectangular},
)
add(
    "losses.mutual_information_loss[tgauss,bins16]",
    g,
    KL.mutual_information_loss,
    [SIG_A, SIG_B],
    {"kernel_function": KL.MIKernel.truncated_gaussian, "num_bins": 16},
)
add(
    "losses.mutual_information_loss[mask]",
    g,
    KL.mutual_information_loss,
    [SIG_A, SIG_B, torch.rand(256) > 0.2, torch.rand(256) > 0.2],
    note="bool masks live (common mask, shape N)",
)
add("losses.mutual_information_loss_2d", g, KL.mutual_information_loss_2d, [GRAY, GRAY * 0.9 + 0.05])
add("losses.mutual_information_loss_3d", g, KL.mutual_information_loss_3d, [VOL, VOL * 0.9 + 0.05], tags=("3d",))
add("losses.normalized_mutual_information_loss", g, KL.normalized_mutual_information_loss, [SIG_A, SIG_B])
add(
    "losses.normalized_mutual_information_loss_2d",
    g,
    KL.normalized_mutual_information_loss_2d,
    [GRAY, GRAY * 0.9 + 0.05],
)
add(
    "losses.normalized_mutual_information_loss_3d",
    g,
    KL.normalized_mutual_information_loss_3d,
    [VOL, VOL * 0.9 + 0.05],
    tags=("3d",),
)
add("losses.MILossFromRef", g, KL.MILossFromRef(SIG_A), [SIG_B], note="reference signal fixed at construction")
add("losses.MILossFromRef[mask]", g, KL.MILossFromRef(SIG_A, mask=torch.rand(256) > 0.2), [SIG_B])
add("losses.MILossFromRef2D", g, KL.MILossFromRef2D(GRAY), [GRAY * 0.9 + 0.05])
add("losses.MILossFromRef3D", g, KL.MILossFromRef3D(VOL), [VOL * 0.9 + 0.05], tags=("3d",))
add("losses.NMILossFromRef", g, KL.NMILossFromRef(SIG_A), [SIG_B])
add("losses.NMILossFromRef2D", g, KL.NMILossFromRef2D(GRAY), [GRAY * 0.9 + 0.05])
add("losses.NMILossFromRef3D", g, KL.NMILossFromRef3D(VOL), [VOL * 0.9 + 0.05], tags=("3d",))
add("losses.MIKernel", g, None, [], skip="Enum of kernel functions")
add("losses.one_hot", g, lambda l: KL.one_hot(l, 4, l.device, torch.float32), [LABELS], note="num_classes baked")

# ------------------------------------------------------------------------------------------------ metrics
g = "metrics"
CLS_LOGITS = torch.randn(8, 5)
CLS_TARGET = torch.randint(0, 5, (8,))
add("metrics.accuracy", g, KM.accuracy, [CLS_LOGITS, CLS_TARGET], note="returns list of 0-d tensors")
add("metrics.accuracy[top1,top3]", g, KM.accuracy, [CLS_LOGITS, CLS_TARGET], {"topk": (1, 3)})
SEG_PRED = torch.randint(0, 4, (2, 16, 20))
add("metrics.confusion_matrix", g, KM.confusion_matrix, [SEG_PRED, LABELS], {"num_classes": 4})
add(
    "metrics.confusion_matrix[normalized]",
    g,
    KM.confusion_matrix,
    [SEG_PRED, LABELS],
    {"num_classes": 4, "normalized": True},
)
add("metrics.mean_iou", g, KM.mean_iou, [SEG_PRED, LABELS], {"num_classes": 4})
BOXES1 = torch.tensor([[40.0, 40, 60, 60], [30, 40, 50, 60], [10, 10, 20, 20]])
BOXES2 = torch.tensor([[40.0, 50, 60, 70], [30, 40, 40, 50]])
add("metrics.mean_iou_bbox", g, KM.mean_iou_bbox, [BOXES1, BOXES2])
add("metrics.mean_iou_bbox[xywh]", g, KM.mean_iou_bbox, [BOXES1, BOXES2], {"box_format": "xywh"})
add("metrics.mean_iou_bbox[cxcywh]", g, KM.mean_iou_bbox, [BOXES1, BOXES2], {"box_format": "cxcywh"})
add(
    "metrics.mean_average_precision",
    g,
    lambda pb, pl, ps, gb, gl: KM.mean_average_precision([pb], [pl], [ps], [gb], [gl], n_classes=3)[0],
    [BOXES1, torch.tensor([0, 1, 2]), torch.tensor([0.9, 0.8, 0.7]), BOXES2, torch.tensor([0, 1])],
    note="Python-heavy (per-class loops, dict return); lists wrapped",
)
add("metrics.psnr", g, KM.psnr, [IMG, IMG2], {"max_val": 1.0})
add("metrics.ssim", g, KM.ssim, [IMG, IMG2], {"window_size": 5})
add("metrics.ssim[valid]", g, KM.ssim, [IMG, IMG2], {"window_size": 5, "padding": "valid"})
add("metrics.SSIM", g, KM.SSIM(5), [IMG, IMG2])
add("metrics.ssim3d", g, KM.ssim3d, [VOL, torch.rand(1, 1, 8, 16, 16)], {"window_size": 3}, tags=("3d",))
add("metrics.SSIM3D", g, KM.SSIM3D(3), [VOL, torch.rand(1, 1, 8, 16, 16)], tags=("3d",))
FLOW = torch.randn(1, 16, 20, 2)
FLOW2 = FLOW + 0.1 * torch.randn(1, 16, 20, 2)
add("metrics.aepe", g, KM.aepe, [FLOW, FLOW2])
add("metrics.aepe[none]", g, KM.aepe, [FLOW, FLOW2], {"reduction": "none"})
add("metrics.average_endpoint_error", g, KM.average_endpoint_error, [FLOW, FLOW2], {"reduction": "sum"})
add("metrics.AEPE", g, KM.AEPE(), [FLOW, FLOW2])
DISP = torch.rand(1, 16, 20) * 10
DISP2 = DISP + torch.randn(1, 16, 20) * 2
VALID = torch.rand(1, 16, 20) > 0.2
add("metrics.mean_absolute_disparity_error", g, KM.mean_absolute_disparity_error, [DISP, DISP2])
add(
    "metrics.mean_absolute_disparity_error[mask]",
    g,
    KM.mean_absolute_disparity_error,
    [DISP, DISP2, VALID],
    note="bool mask live",
)
add("metrics.root_mean_squared_disparity_error", g, KM.root_mean_squared_disparity_error, [DISP, DISP2])
add(
    "metrics.root_mean_squared_disparity_error[mask,none]",
    g,
    KM.root_mean_squared_disparity_error,
    [DISP, DISP2, VALID],
    {"reduction": "none"},
)
add("metrics.mean_bad_pixel_error", g, KM.mean_bad_pixel_error, [DISP, DISP2], {"threshold": 1.5})
add(
    "metrics.mean_bad_pixel_error[mask]",
    g,
    lambda a, b, m: KM.mean_bad_pixel_error(a, b, 1.5, m),
    [DISP, DISP2, VALID],
    note="bool mask live",
)
R1 = K.geometry.axis_angle_to_rotation_matrix(torch.tensor([[0.1, 0.2, 0.3], [0.0, 0.5, 0.1]]))
R2 = K.geometry.axis_angle_to_rotation_matrix(torch.tensor([[0.15, 0.2, 0.25], [0.1, 0.4, 0.1]]))
add("metrics.angle_error_mat", g, KM.angle_error_mat, [R1[0], R2[0]])
add("metrics.angle_error_mat[batch]", g, KM.angle_error_mat, [R1, R2], tags=("batch>1",))
add("metrics.angle_error_vec", g, KM.angle_error_vec, [torch.tensor([1.0, 0.2, 0.1]), torch.tensor([0.9, 0.3, 0.0])])
add("metrics.angle_error_vec[batch]", g, KM.angle_error_vec, [torch.randn(4, 3), torch.randn(4, 3)], tags=("batch>1",))
add("metrics.translation_ate", g, KM.translation_ate, [torch.randn(3), torch.randn(3)])
add("metrics.translation_ate[batch]", g, KM.translation_ate, [torch.randn(4, 3), torch.randn(4, 3)], tags=("batch>1",))
P1 = torch.eye(4).repeat(2, 1, 1)
P1[:, :3, :3] = R1
P1[:, :3, 3] = torch.tensor([[1.0, 0.0, 0.5], [0.0, 1.0, 0.2]])
P2 = torch.eye(4).repeat(2, 1, 1)
P2[:, :3, :3] = R2
P2[:, :3, 3] = torch.tensor([[1.1, 0.1, 0.4], [0.2, 0.9, 0.2]])
add("metrics.pose_errors", g, KM.pose_errors, [P1[0], P2[0]], note="returns dict of tensors")
add("metrics.pose_errors[batch,no_fold]", g, KM.pose_errors, [P1, P2], {"fold_translation": False}, tags=("batch>1",))
add(
    "metrics.auc_from_errors",
    g,
    KM.auc_from_errors,
    [torch.rand(20) * 10],
    note="returns dict[float,float] of Python floats",
)
add("metrics.AverageMeter", g, None, [], skip="running-statistics helper, not a tensor op")

# ------------------------------------------------------------------------------------------------ utils
g = "utils"
add(
    "utils.create_meshgrid",
    g,
    lambda d: K.geometry.create_meshgrid(16, 24, device=d.device) + d * 0,
    [torch.zeros(1)],
    note="H,W baked; graph is fully constant (kornia.geometry.create_meshgrid)",
)
add(
    "utils.create_meshgrid[unnormalized]",
    g,
    lambda d: K.geometry.create_meshgrid(16, 24, normalized_coordinates=False, device=d.device) + d * 0,
    [torch.zeros(1)],
    note="constant graph",
)
add(
    "utils.create_meshgrid3d",
    g,
    lambda d: K.geometry.create_meshgrid3d(4, 8, 12, device=d.device) + d * 0,
    [torch.zeros(1)],
    note="constant graph",
    tags=("3d",),
)
add(
    "utils.create_meshgrid3d[unnormalized]",
    g,
    lambda d: K.geometry.create_meshgrid3d(4, 8, 12, normalized_coordinates=False, device=d.device) + d * 0,
    [torch.zeros(1)],
    note="constant graph",
    tags=("3d",),
)
add(
    "utils.eye_like",
    g,
    lambda t: K.core.ops.eye_like(3, t),
    [torch.rand(4, 3, 3)],
    note="n baked; batch from input (kornia.core.ops)",
)
add("utils.eye_like[shared]", g, lambda t: K.core.ops.eye_like(3, t, shared_memory=True), [torch.rand(4, 3, 3)])
add("utils.vec_like", g, lambda t: K.core.ops.vec_like(3, t), [torch.rand(4, 3, 3)], note="kornia.core.ops")
add(
    "utils.one_hot",
    g,
    None,
    [],
    skip="deprecated alias of kornia.losses.one_hot (kornia.utils is a deprecation shim, not an attribute of kornia)",
)
add(
    "utils.draw_line",
    g,
    lambda img, p1, p2, c: K.image.draw_line(img.clone(), p1, p2, c),
    [torch.zeros(3, 32, 40), torch.tensor([2.0, 3.0]), torch.tensor([35.0, 28.0]), torch.tensor([1.0, 0.5, 0.2])],
    note="kornia.image.draw_line; single image CxHxW",
)
add(
    "utils.draw_rectangle",
    g,
    lambda img, r: K.image.draw_rectangle(img.clone(), r),
    [torch.zeros(2, 3, 32, 40), torch.tensor([[[2.0, 3.0, 20.0, 25.0]], [[5.0, 5.0, 30.0, 30.0]]])],
    note="kornia.image.draw_rectangle, default color",
)
add(
    "utils.draw_rectangle[fill,color]",
    g,
    lambda img, r, c: K.image.draw_rectangle(img.clone(), r, c, fill=True),
    [
        torch.zeros(1, 3, 32, 40),
        torch.tensor([[[2.0, 3.0, 20.0, 25.0], [10.0, 10.0, 30.0, 20.0]]]),
        torch.tensor([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]]),
    ],
)
add(
    "utils.draw_point2d",
    g,
    lambda img, p, c: K.image.draw_point2d(img.clone(), p, c),
    [torch.zeros(3, 32, 40), torch.tensor([[2, 3], [10, 20], [30, 5]]), torch.tensor([1.0, 0.5, 0.2])],
    note="kornia.image.draw_point2d; points int64 (x,y)",
)
add(
    "utils.draw_convex_polygon",
    g,
    lambda img, poly, c: K.image.draw_convex_polygon(img.clone(), poly, c),
    [
        torch.zeros(1, 3, 32, 40),
        torch.tensor([[[4.0, 4.0], [30.0, 6.0], [28.0, 25.0], [6.0, 20.0]]]),
        torch.tensor([[0.5, 0.2, 0.9]]),
    ],
    note="kornia.image.draw_convex_polygon; polygons as BxNx2 tensor",
)
add(
    "utils.safe_inverse_with_mask",
    g,
    K.core.utils.safe_inverse_with_mask,
    [torch.eye(3)[None].repeat(3, 1, 1) + 0.1 * torch.rand(3, 3, 3)],
    note="returns (inv, mask)",
)
add(
    "utils.safe_solve_with_mask",
    g,
    K.core.utils.safe_solve_with_mask,
    [torch.rand(3, 3, 1), torch.eye(3)[None].repeat(3, 1, 1) + 0.1 * torch.rand(3, 3, 3)],
    note="returns (X, LU, mask)",
)
add(
    "utils.batched_forward",
    g,
    lambda x: K.core.utils.batched_forward(torch.nn.Identity(), x, torch.device("cpu"), batch_size=4),
    [torch.rand(10, 3)],
    note="micro-batch loop over Identity; batch_size baked",
)
add(
    "utils.torch_meshgrid",
    g,
    lambda a, b: torch.stack(_torch_meshgrid([a, b], indexing="ij")),
    [torch.arange(4.0), torch.arange(6.0)],
    note="thin wrapper over torch.meshgrid (kornia.utils shim)",
)
add("utils.ImageToTensor", g, None, [], skip="numpy input")
add("utils.image_to_tensor", g, None, [], skip="numpy input")
add("utils.image_list_to_tensor", g, None, [], skip="list of numpy inputs")
add("utils.tensor_to_image", g, None, [], skip="numpy output")
add("utils.image_to_string", g, None, [], skip="returns str")
add("utils.print_image", g, None, [], skip="prints; no tensor output")
add("utils.get_cuda_device_if_available", g, None, [], skip="returns torch.device")
add("utils.get_mps_device_if_available", g, None, [], skip="returns torch.device")
add("utils.get_cuda_or_mps_device_if_available", g, None, [], skip="returns torch.device")
add("utils.is_autocast_enabled", g, None, [], skip="returns Python bool")
add("utils.is_mps_tensor_safe", g, None, [], skip="returns Python bool")
add("utils.xla_is_available", g, None, [], skip="returns Python bool")
add("utils.dataclass_to_dict", g, None, [], skip="Python container helper")
add("utils.dict_to_dataclass", g, None, [], skip="Python container helper")
add("utils.deprecated", g, None, [], skip="decorator")
add("utils.map_location_to_cpu", g, None, [], skip="torch.load helper")
add("utils.get_sample_images", g, None, [], skip="downloads sample images; not a tensor op")
add("utils.load_pointcloud_ply", g, None, [], skip="file I/O")
add("utils.save_pointcloud_ply", g, None, [], skip="file I/O")

# ------------------------------------------------------------------------------------------------ morphology
g = "morphology"
add(
    "morphology.[no module classes]",
    g,
    None,
    [],
    skip="kornia.morphology exposes only the 7 functions already covered in the first pass",
)
add(
    "morphology.dilation[3d_kernel_variants]",
    g,
    K.morphology.dilation,
    [IMG, torch.ones(3, 5)],
    {"engine": "convolution"},
    note="engine=convolution variant (first pass used unfold)",
)
add("morphology.erosion[convolution]", g, K.morphology.erosion, [IMG, torch.ones(3, 5)], {"engine": "convolution"})
add(
    "morphology.dilation[structuring_element]",
    g,
    K.morphology.dilation,
    [IMG, torch.ones(3, 3), torch.zeros(3, 3)],
    note="structuring_element live tensor (non-flat)",
)
add(
    "morphology.dilation[border_constant]",
    g,
    K.morphology.dilation,
    [IMG, torch.ones(3, 3)],
    {"border_type": "constant", "border_value": 0.5},
)
add("morphology.gradient[batch]", g, K.morphology.gradient, [IMGB, torch.ones(3, 3)], tags=("batch>1",))

if __name__ == "__main__":
    run_cases(CASES, sys.argv[1], only=sys.argv[2:] or None)
