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

from testing.base import assert_close

# Every test in this module must stay device-free: the mark deselects the whole file on non-CPU devices.
pytestmark = pytest.mark.device_agnostic

onnx = pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")
if torch_version_ge(2, 9, 0):
    pytest.importorskip("onnxscript")  # the dynamo exporter, the default from torch 2.9

from kornia.core import ImageSequential  # noqa: E402
from kornia.onnx import ONNXModule, ONNXSequential, load  # noqa: E402

_RGB = {"input_shape": [-1, 3, -1, -1], "output_shape": [-1, 3, -1, -1]}


def _opsets(op):
    return {entry.version for entry in op.opset_import if entry.domain in ("", "ai.onnx")}


@pytest.fixture
def sigmoid():
    return ImageSequential(torch.nn.Sigmoid())


@pytest.fixture
def op(sigmoid):
    return sigmoid.to_onnx(save=False, **_RGB)


class TestONNXModule:
    @pytest.mark.parametrize(
        ("targets", "expected"),
        [
            ({"target_ir_version": 9}, ({18}, 9)),
            ({"target_opset_version": 17}, ({17}, None)),  # None: the IR version of the export
            ({"target_ir_version": 9, "target_opset_version": 17}, ({17}, 9)),
        ],
    )
    def test_target_versions(self, sigmoid, op, targets, expected):
        # IR 10 from the dynamo exporter, 8 from the legacy one (the default before torch 2.9): 9 converts both
        versions = (_opsets(op), op.ir_version)
        assert versions[0] == {18}
        assert versions[1] != 9
        opsets, ir_version = expected
        if ir_version is None:
            ir_version = op.ir_version

        module = ONNXModule(op, **targets)

        assert (_opsets(module.op), module.op.ir_version) == (opsets, ir_version)
        assert (_opsets(op), op.ir_version) == versions  # the caller's graph is converted on a copy
        x = torch.rand(2, 3, 4, 5)
        assert_close(torch.from_numpy(module(x.numpy())[0]), sigmoid(x))

    @pytest.mark.parametrize("form", ["Path", "~"])
    def test_takes_file_paths_as_path_objects_and_with_a_tilde(self, sigmoid, form, monkeypatch, tmp_path):
        import os

        path = tmp_path / "sigmoid.onnx"
        sigmoid.to_onnx(str(path), **_RGB)
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setenv("USERPROFILE", str(tmp_path))  # what os.path.expanduser reads on Windows
        model = {"Path": path, "~": os.path.join("~", "sigmoid.onnx")}[form]

        x = torch.rand(1, 3, 4, 5)
        for runtime in (ONNXModule(model), load(model), ONNXSequential(model)):
            assert_close(torch.from_numpy(runtime(x.numpy())[0]), sigmoid(x))

    def test_bytes_raise_type_error_naming_load_model_from_string(self, sigmoid, tmp_path):
        # ONNX Runtime reads bytes as a serialized model; kornia does not take them as a path either
        path = tmp_path / "sigmoid.onnx"
        sigmoid.to_onnx(str(path), **_RGB)
        for model in (path.read_bytes(), bytes(path)):
            for runtime in (ONNXModule, load, ONNXSequential):
                with pytest.raises(TypeError, match=r"got bytes\. For a serialized model, pass onnx\.load_model_from"):
                    runtime(model)

    @pytest.mark.parametrize("model", [1, None, {"model": "m.onnx"}])
    def test_other_argument_types_raise_type_error(self, model):
        message = r"an onnx.ModelProto, a str \(a file path, an 'hf://' name or a URL\), or an os.PathLike file"
        with pytest.raises(TypeError, match=message):
            ONNXModule(model)
        with pytest.raises(TypeError, match=message):
            ONNXSequential(model)
