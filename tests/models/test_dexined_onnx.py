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

from kornia.models.dexined import DexiNed


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
