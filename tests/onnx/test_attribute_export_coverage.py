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

import io

import numpy as np
import pytest
import torch
from torch import nn

pytestmark = pytest.mark.device_agnostic

onnx = pytest.importorskip("onnx")
ort = pytest.importorskip("onnxruntime")
pytest.importorskip("onnxscript")

if torch.__version__ < "2.5":
    pytest.skip("ONNX export coverage requires torch >= 2.5", allow_module_level=True)

from kornia.color.gray import BgrToGrayscale, GrayscaleToRgb, RgbToGrayscale
from kornia.color.hls import HlsToRgb, RgbToHls
from kornia.color.hsv import HsvToRgb, RgbToHsv
from kornia.color.lab import LabToRgb, RgbToLab
from kornia.color.luv import LuvToRgb, RgbToLuv
from kornia.color.raw import RawToRgb, RgbToRaw, CFA
from kornia.color.rgb import (
    BgrToRgb,
    BgrToRgba,
    LinearRgbToRgb,
    RgbToBgr,
    RgbToLinearRgb,
    RgbToRgba,
    RgbaToBgr,
    RgbaToRgb,
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


def _shape_from_attribute(shape: list[int], *, channel_override: int | None = None) -> tuple[int, ...]:
    result = [1 if dim == -1 else dim for dim in shape]

    if channel_override is not None and len(result) >= 2:
        result[1] = channel_override

    # Keep spatial tensors small while satisfying typical convolution/kernel constraints.
    for index in range(2, len(result)):
        if result[index] == 1:
            result[index] = 16

    return tuple(result)



def _make_cases():
    return [
        pytest.param(BgrToGrayscale(), id="BgrToGrayscale"),
        pytest.param(GrayscaleToRgb(), id="GrayscaleToRgb"),
        pytest.param(RgbToGrayscale(), id="RgbToGrayscale"),
        pytest.param(HlsToRgb(), id="HlsToRgb"),
        pytest.param(RgbToHls(), id="RgbToHls"),
        pytest.param(HsvToRgb(), id="HsvToRgb"),
        pytest.param(RgbToHsv(), id="RgbToHsv"),
        pytest.param(LabToRgb(), id="LabToRgb"),
        pytest.param(RgbToLab(), id="RgbToLab"),
        pytest.param(LuvToRgb(), id="LuvToRgb"),
        pytest.param(RgbToLuv(), id="RgbToLuv"),
        pytest.param(RawToRgb(CFA.BG), id="RawToRgb"),
        pytest.param(RgbToRaw(CFA.BG), id="RgbToRaw"),
        pytest.param(BgrToRgb(), id="BgrToRgb"),
        pytest.param(BgrToRgba(1.0), id="BgrToRgba"),
        pytest.param(LinearRgbToRgb(), id="LinearRgbToRgb"),
        pytest.param(RgbToBgr(), id="RgbToBgr"),
        pytest.param(RgbToLinearRgb(), id="RgbToLinearRgb"),
        pytest.param(RgbToRgba(1.0), id="RgbToRgba"),
        pytest.param(RgbaToBgr(), id="RgbaToBgr"),
        pytest.param(RgbaToRgb(), id="RgbaToRgb"),
        pytest.param(RgbToXyz(), id="RgbToXyz"),
        pytest.param(XyzToRgb(), id="XyzToRgb"),
        pytest.param(RgbToYcbcr(), id="RgbToYcbcr"),
        pytest.param(YcbcrToRgb(), id="YcbcrToRgb"),
        pytest.param(RgbToYuv(), id="RgbToYuv"),
        pytest.param(RgbToYuv420(), id="RgbToYuv420"),
        pytest.param(RgbToYuv422(), id="RgbToYuv422"),
        pytest.param(Yuv420ToRgb(), id="Yuv420ToRgb"),
        pytest.param(Yuv422ToRgb(), id="Yuv422ToRgb"),
        pytest.param(YuvToRgb(), id="YuvToRgb"),
        pytest.param(AdjustHue(0.0), id="AdjustHue"),
        pytest.param(AdjustSaturation(1.0), id="AdjustSaturation"),
        pytest.param(AdjustSaturationWithGraySubtraction(1.0), id="AdjustSaturationWithGraySubtraction"),
        pytest.param(
            Canny(),
            id="Canny",
            marks=pytest.mark.xfail(
                strict=True,
                reason="kornia issue #5398: Canny fails Dynamo ONNX export",
            ),
        ),
        pytest.param(DexiNed(pretrained=False), id="DexiNed"),
        pytest.param(MotionBlur3D(3, 35.0, 0.5), id="MotionBlur3D"),
        pytest.param(SpatialGradient(), id="SpatialGradient"),
        pytest.param(SpatialGradient3d(), id="SpatialGradient3d"),
        pytest.param(
            SemanticSegmentation(
                model=nn.Conv2d(3, 3, kernel_size=1),
                pre_processor=nn.Identity(),
                post_processor=nn.Identity(),
            ),
            id="SemanticSegmentation",
        ),
    ]


def _make_inputs(module: nn.Module) -> tuple[torch.Tensor, ...]:
    shape = getattr(module, "ONNX_DEFAULT_INPUTSHAPE", None)
    if shape is None:
        raise AssertionError(f"{type(module).__name__} has no ONNX_DEFAULT_INPUTSHAPE")

    input_shape = _shape_from_attribute(shape)

    name = type(module).__name__

    if name == "Yuv420ToRgb":
        y = torch.rand(*input_shape)
        uv = torch.rand(input_shape[0], 2, input_shape[2] // 2, input_shape[3] // 2)
        return y, uv

    if name == "Yuv422ToRgb":
        y = torch.rand(*input_shape)
        uv = torch.rand(input_shape[0], 2, input_shape[2], input_shape[3] // 2)
        return y, uv

    if name in {"SpatialGradient", "SpatialGradient3d"}:
        return (torch.rand(*input_shape),)

    return (torch.rand(*input_shape),)


def _export_and_run(module: nn.Module, inputs: tuple[torch.Tensor, ...]) -> None:
    module.eval()

    with torch.no_grad():
        eager = module(*inputs)

    if not isinstance(eager, (tuple, list)):
        eager = [eager]

    buffer = io.BytesIO()

    torch.onnx.export(
        module,
        inputs,
        buffer,
        dynamo=True,
        opset_version=18,
        verbose=False,
    )

    buffer.seek(0)
    model = onnx.load_model(buffer)
    onnx.checker.check_model(model)

    session = ort.InferenceSession(buffer.getvalue(), providers=["CPUExecutionProvider"])

    assert len(session.get_inputs()) == len(inputs)

    ort_inputs = {
        input_info.name: tensor.detach().cpu().numpy()
        for input_info, tensor in zip(session.get_inputs(), inputs)
    }

    outputs = session.run(None, ort_inputs)

    assert len(outputs) == len(eager)

    for expected, actual in zip(eager, outputs):
        np.testing.assert_allclose(
            expected.detach().cpu().numpy(),
            actual,
            rtol=1e-4,
            atol=1e-4,
        )


@pytest.mark.parametrize("module", _make_cases())
def test_attribute_driven_onnx_export(module: nn.Module) -> None:
    _export_and_run(module, _make_inputs(module))
