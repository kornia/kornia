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

import importlib
import inspect
import io
import pkgutil

import numpy as np
import pytest
import torch
from torch import nn

from kornia.color.gray import BgrToGrayscale, GrayscaleToRgb, RgbToGrayscale
from kornia.color.hls import HlsToRgb, RgbToHls
from kornia.color.hsv import HsvToRgb, RgbToHsv
from kornia.color.lab import LabToRgb, RgbToLab
from kornia.color.luv import LuvToRgb, RgbToLuv
from kornia.color.raw import CFA, RawToRgb, RgbToRaw
from kornia.color.rgb import (
    BgrToRgb,
    BgrToRgba,
    LinearRgbToRgb,
    RgbaToBgr,
    RgbaToRgb,
    RgbToBgr,
    RgbToLinearRgb,
    RgbToRgba,
)
from kornia.color.xyz import RgbToXyz, XyzToRgb
from kornia.color.ycbcr import RgbToYcbcr, YcbcrToRgb
from kornia.color.yuv import (
    RgbToYuv,
    RgbToYuv420,
    RgbToYuv422,
    Yuv420ToRgb,
    Yuv422ToRgb,
    YuvToRgb,
)
from kornia.core._compat import torch_version_lt
from kornia.core.mixin.onnx import ONNXExportMixin
from kornia.enhance.adjust import (
    AdjustHue,
    AdjustSaturation,
    AdjustSaturationWithGraySubtraction,
)
from kornia.filters.canny import Canny
from kornia.filters.dexined import DexiNed
from kornia.filters.motion import MotionBlur3D
from kornia.filters.sobel import SpatialGradient, SpatialGradient3d
from kornia.models.segmentation.base import SemanticSegmentation

# Every tensor here is created device-free and fed to onnxruntime as numpy, so the accelerator legs must not
# repeat the work.
pytestmark = pytest.mark.device_agnostic

onnx = pytest.importorskip("onnx")
ort = pytest.importorskip("onnxruntime")
pytest.importorskip("onnxscript")  # the dynamo exporter hard-requires it

if torch_version_lt(2, 5, 0):
    # Same floor as ``test_export_coverage.py``: ``torch.onnx.export(..., dynamo=True)`` returns the
    # ``ONNXProgram`` used below from torch 2.5, and 2.5.1 is a blocking PR leg.
    pytest.skip("the dynamo ONNX exporter needs torch >= 2.5", allow_module_level=True)

# The packages whose classes declare ``ONNX_EXPORTABLE`` / ``ONNX_DEFAULT_INPUTSHAPE`` (#5215).
_DECLARING_PACKAGES = ("kornia.color", "kornia.enhance", "kornia.filters", "kornia.models.segmentation")

# A class that keeps ``ONNX_EXPORTABLE = False`` needs a tracking issue: its case is a strict xfail that must
# fail inside the exporter, so it turns red once the class exports and the marker can go.
_TRACKING_ISSUES = {"Canny": "#5398: Canny fails Dynamo ONNX export"}

# Classes that export from torch 2.9 (2.9.1 and 2.14 checked) but not with the exporter of torch 2.5.1 (RgbToYuv422
# gives an invalid ``Range`` node, DexiNed fails on batch norm, MotionBlur3D on ``grid_sampler_3d``) or 2.6.0
# (HsvToRgb, AdjustHue, RgbToYuv422, DexiNed). These are limits of the old exporter, not of kornia, so the marker
# is not strict: 2.7 and 2.8 are unchecked.
_OLD_EXPORTER_FAILURES = {"RgbToYuv422", "DexiNed", "MotionBlur3D", "HsvToRgb", "AdjustHue"}


def _declaring_classes() -> set[str]:
    found = set()
    for package_name in _DECLARING_PACKAGES:
        package = importlib.import_module(package_name)
        for info in pkgutil.walk_packages(package.__path__, package_name + "."):
            module = importlib.import_module(info.name)
            for _, cls in inspect.getmembers(module, inspect.isclass):
                if cls.__module__ == info.name and {"ONNX_DEFAULT_INPUTSHAPE", "ONNX_EXPORTABLE"} & set(cls.__dict__):
                    found.add(cls.__qualname__)
    return found


_CLASSES = [
    BgrToGrayscale,
    GrayscaleToRgb,
    RgbToGrayscale,
    HlsToRgb,
    RgbToHls,
    HsvToRgb,
    RgbToHsv,
    LabToRgb,
    RgbToLab,
    LuvToRgb,
    RgbToLuv,
    RawToRgb,
    RgbToRaw,
    BgrToRgb,
    BgrToRgba,
    LinearRgbToRgb,
    RgbToBgr,
    RgbToLinearRgb,
    RgbToRgba,
    RgbaToBgr,
    RgbaToRgb,
    RgbToXyz,
    XyzToRgb,
    RgbToYcbcr,
    YcbcrToRgb,
    RgbToYuv,
    RgbToYuv420,
    RgbToYuv422,
    Yuv420ToRgb,
    Yuv422ToRgb,
    YuvToRgb,
    AdjustHue,
    AdjustSaturation,
    AdjustSaturationWithGraySubtraction,
    Canny,
    DexiNed,
    MotionBlur3D,
    SpatialGradient,
    SpatialGradient3d,
    SemanticSegmentation,
]

# Classes whose constructor needs arguments; the others are built with none. Built inside the test, so
# collection does not construct DexiNed.
_FACTORIES = {
    RawToRgb: lambda: RawToRgb(CFA.BG),
    RgbToRaw: lambda: RgbToRaw(CFA.BG),
    BgrToRgba: lambda: BgrToRgba(1.0),
    RgbToRgba: lambda: RgbToRgba(1.0),
    AdjustHue: lambda: AdjustHue(0.0),
    AdjustSaturation: lambda: AdjustSaturation(1.0),
    AdjustSaturationWithGraySubtraction: lambda: AdjustSaturationWithGraySubtraction(1.0),
    DexiNed: lambda: DexiNed(pretrained=False),
    MotionBlur3D: lambda: MotionBlur3D(3, 35.0, 0.5),
    SemanticSegmentation: lambda: SemanticSegmentation(
        model=nn.Conv2d(3, 3, kernel_size=1), pre_processor=nn.Identity(), post_processor=nn.Identity()
    ),
}


def _make_cases():
    cases = []
    for cls in _CLASSES:
        marks = []
        if cls.__dict__.get("ONNX_EXPORTABLE") is False:
            # ``raises`` keeps the xfail honest: an error outside the exporter, such as a missing input shape,
            # fails the case instead of satisfying the marker.
            reason = _TRACKING_ISSUES.get(cls.__name__, f"{cls.__name__} has no tracking issue")
            marks.append(pytest.mark.xfail(strict=True, raises=torch.onnx.OnnxExporterError, reason=reason))
        if cls.__name__ in _OLD_EXPORTER_FAILURES:
            marks.append(
                pytest.mark.xfail(torch_version_lt(2, 9, 0), reason="the dynamo exporter of torch < 2.9", strict=False)
            )
        cases.append(pytest.param(cls, id=cls.__name__, marks=marks))
    return cases


def _make_inputs(module: nn.Module) -> tuple[torch.Tensor, ...]:
    # A class without its own shape (Canny) takes the default that ``ONNXExportMixin.to_onnx`` would use.
    shape = getattr(module, "ONNX_DEFAULT_INPUTSHAPE", ONNXExportMixin.ONNX_DEFAULT_INPUTSHAPE)
    # Dynamic batch and channel axes -> 1; dynamic spatial axes -> 16, small and even for the subsampled planes.
    input_shape = tuple((1 if index < 2 else 16) if dim == -1 else dim for index, dim in enumerate(shape))

    # The two-input YUV decoders declare the shape of the luma plane; the chroma plane is subsampled.
    if isinstance(module, Yuv420ToRgb):
        batch, _, height, width = input_shape
        return torch.rand(*input_shape), torch.rand(batch, 2, height // 2, width // 2)
    if isinstance(module, Yuv422ToRgb):
        batch, _, height, width = input_shape
        return torch.rand(*input_shape), torch.rand(batch, 2, height, width // 2)
    return (torch.rand(*input_shape),)


def _export_and_run(module: nn.Module, inputs: tuple[torch.Tensor, ...]) -> None:
    module.eval()
    with torch.no_grad():
        eager = module(*inputs)
    eager = list(eager) if isinstance(eager, (tuple, list)) else [eager]

    # torch 2.5 cannot write a dynamo export to a file object; save the returned program instead.
    buffer = io.BytesIO()
    with torch.no_grad():
        program = torch.onnx.export(module, inputs, dynamo=True, opset_version=18, verbose=False)
    program.save(buffer)
    model = onnx.load_from_string(buffer.getvalue())
    onnx.checker.check_model(model)

    session = ort.InferenceSession(buffer.getvalue(), providers=["CPUExecutionProvider"])
    assert len(session.get_inputs()) == len(inputs), [i.name for i in session.get_inputs()]
    feeds = {info.name: tensor.numpy() for info, tensor in zip(session.get_inputs(), inputs, strict=True)}
    outputs = session.run(None, feeds)

    assert len(outputs) == len(eager)
    for expected, actual in zip(eager, outputs):
        np.testing.assert_allclose(expected.numpy(), actual, rtol=1e-4, atol=1e-4)


def test_cases_cover_every_class_declaring_the_onnx_attributes() -> None:
    assert {cls.__qualname__ for cls in _CLASSES} == _declaring_classes()
    assert {cls.__name__ for cls in _CLASSES if cls.__dict__.get("ONNX_EXPORTABLE") is False} == set(_TRACKING_ISSUES)


@pytest.mark.parametrize("cls", _make_cases())
def test_attribute_driven_onnx_export(cls: type) -> None:
    module = _FACTORIES.get(cls, cls)()
    _export_and_run(module, _make_inputs(module))
