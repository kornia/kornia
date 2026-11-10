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

import pytest
import torch

from kornia.core._compat import torch_version_ge
from kornia.models.dexined import DexiNed

from testing.base import assert_close

# Every test in this module must stay device-free: the mark deselects the whole file on non-CPU devices.
pytestmark = pytest.mark.device_agnostic


@pytest.mark.timeout(120)
def test_dexined_to_onnx(tmp_path):
    """Ensure `DexiNed.to_onnx` exports a valid ONNX model and saves the file."""
    onnx = pytest.importorskip("onnx")
    pytest.importorskip("onnxscript")

    model = DexiNed(pretrained=False)
    model.eval()

    onnx_path = tmp_path / "dexined.onnx"
    op = model.to_onnx(save=True, onnx_name=str(onnx_path), pseudo_shape=[1, 3, 32, 32])

    assert isinstance(op, onnx.ModelProto)
    assert onnx_path.exists(), "ONNX file was not written to disk"


@pytest.mark.timeout(120)
@pytest.mark.parametrize("export_kwargs", [{}, {"dynamo": False}], ids=["default", "legacy"])
def test_dexined_onnx_matches_eager_at_other_shapes(export_kwargs):
    """Batch, height and width are dynamic: other batches than the traced 2, H != W and odd sizes match eager."""
    pytest.importorskip("onnx")
    ort = pytest.importorskip("onnxruntime")
    if export_kwargs.get("dynamo", torch_version_ge(2, 9, 0)):
        pytest.importorskip("onnxscript")
    torch.manual_seed(0)
    model = DexiNed(pretrained=False).eval()
    op = model.to_onnx(save=False, pseudo_shape=[1, 3, 32, 32], **export_kwargs)
    session = ort.InferenceSession(op.SerializeToString(), providers=["CPUExecutionProvider"])

    generator = torch.Generator().manual_seed(0)
    for shape in [(1, 3, 32, 32), (3, 3, 48, 80), (1, 3, 33, 47), (2, 3, 65, 31)]:
        images = torch.rand(*shape, generator=generator)
        (edges,) = session.run(None, {session.get_inputs()[0].name: images.numpy()})
        with torch.no_grad():
            expected = model(images)
        assert_close(torch.from_numpy(edges), expected, rtol=1e-4, atol=1e-4)
