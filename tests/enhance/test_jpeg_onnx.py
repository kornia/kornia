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

import pytest
import torch

from kornia.core._compat import torch_version_ge
from kornia.enhance import JPEGCodecDifferentiable
from kornia.enhance.jpeg import _chroma_subsampling, _chroma_upsampling, _perform_padding
from kornia.geometry.transform import rescale

from testing.base import assert_close

# Every test in this module must stay device-free: the mark deselects the whole file on non-CPU devices.
pytestmark = pytest.mark.device_agnostic


def _export_codec(model: JPEGCodecDifferentiable, example: tuple[torch.Tensor, torch.Tensor], dynamo: bool) -> bytes:
    """Export ``model`` to ONNX with a dynamic batch, height and width."""
    buffer = io.BytesIO()
    if dynamo:
        from torch.export import Dim

        batch, height, width = Dim("batch"), Dim("height"), Dim("width")
        torch.onnx.export(
            model,
            example,
            buffer,
            dynamo=True,
            dynamic_shapes={"image_rgb": {0: batch, 2: height, 3: width}, "jpeg_quality": {0: batch}},
        )
    else:
        torch.onnx.export(
            model,
            example,
            buffer,
            dynamo=False,
            opset_version=17,
            input_names=["image", "quality"],
            output_names=["out"],
            dynamic_axes={
                "image": {0: "B", 2: "H", 3: "W"},
                "quality": {0: "B"},
                "out": {0: "B", 2: "H", 3: "W"},
            },
        )
    return buffer.getvalue()


@pytest.mark.timeout(240)
@pytest.mark.parametrize("export_kwargs", [{}, {"dynamo": False}], ids=["default", "legacy"])
def test_jpeg_codec_onnx_matches_eager_at_other_shapes(export_kwargs):
    """Traced once, the graph follows the batch, height and width, including sizes that are not multiples of 16."""
    pytest.importorskip("onnx")
    ort = pytest.importorskip("onnxruntime")
    # ``default`` is the dynamo exporter from torch 2.9 on, and the legacy one before.
    dynamo = export_kwargs.get("dynamo", torch_version_ge(2, 9, 0))
    if dynamo:
        pytest.importorskip("onnxscript")

    model = JPEGCodecDifferentiable().eval()
    example = (torch.rand(2, 3, 32, 48), torch.tensor([50.0, 80.0]))
    session = ort.InferenceSession(_export_codec(model, example, dynamo), providers=["CPUExecutionProvider"])
    input_names = [i.name for i in session.get_inputs()]

    generator = torch.Generator().manual_seed(0)
    # The traced size, other batches, other multiples of 16, odd sizes, and the smallest padded image.
    shapes = [
        (2, 3, 32, 48),
        (1, 3, 32, 48),
        (3, 3, 48, 64),
        (2, 3, 64, 32),
        (1, 3, 41, 70),
        (2, 3, 17, 17),
        (1, 3, 16, 16),
    ]
    for shape in shapes:
        image = torch.rand(*shape, generator=generator)
        quality = torch.rand(shape[0], generator=generator) * 90.0 + 5.0
        (actual,) = session.run(None, {input_names[0]: image.numpy(), input_names[1]: quality.numpy()})
        with torch.no_grad():
            expected = model(image, quality)
        assert actual.shape == tuple(expected.shape), f"traced at (32, 48), run at {shape}"
        assert_close(torch.from_numpy(actual), expected, rtol=1e-4, atol=1e-3)


@pytest.mark.parametrize("size", [(1, 1), (5, 7), (16, 16), (17, 33), (41, 70)])
def test_perform_padding_makes_the_size_a_multiple_of_16(size):
    image = torch.rand(1, 3, *size)
    padded, h_pad, w_pad = _perform_padding(image)
    assert padded.shape[-2:] == (size[0] + h_pad, size[1] + w_pad)
    assert padded.shape[-2] % 16 == 0
    assert padded.shape[-1] % 16 == 0
    assert 0 <= h_pad < 16
    assert 0 <= w_pad < 16
    assert_close(padded[..., : size[0], : size[1]], image, rtol=0.0, atol=0.0)


@pytest.mark.parametrize("size", [(16, 16), (32, 48), (64, 32)])
def test_chroma_resampling_matches_rescale(size):
    """The codec spells out its chroma resampling with explicit sizes; it must keep the values ``rescale`` gave."""
    ycbcr = torch.rand(2, 3, *size, generator=torch.Generator().manual_seed(0)) * 255.0
    _, cb, cr = _chroma_subsampling(ycbcr)
    for actual, channel in ((cb, ycbcr[:, 1]), (cr, ycbcr[:, 2])):
        expected = rescale(channel[:, None], factor=0.5, interpolation="bilinear", align_corners=False, antialias=True)
        assert_close(actual, expected[:, 0], rtol=0.0, atol=0.0)
        up = rescale(actual[:, None], factor=2.0, interpolation="bilinear", align_corners=False, antialias=False)
        assert_close(_chroma_upsampling(actual), up[:, 0], rtol=0.0, atol=0.0)
