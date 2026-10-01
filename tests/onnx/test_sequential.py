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

from kornia.onnx.sequential import ONNXSequential  # noqa: E402


class TestONNXSequential:
    @pytest.fixture
    def mock_model_proto(self):
        from onnx.helper import make_graph, make_model, make_node, make_tensor_value_info

        # Create a minimal ONNX model with an input and output
        input_info = make_tensor_value_info("input", onnx.TensorProto.FLOAT, [1, 2])
        output_info = make_tensor_value_info("output", onnx.TensorProto.FLOAT, [1, 2])
        node = make_node("Identity", ["input"], ["output"])
        graph = make_graph([node], "test_graph", [input_info], [output_info])
        op = onnx.OperatorSetIdProto()
        op.version = 17
        return make_model(graph, opset_imports=[op], ir_version=9)

    @pytest.fixture
    def onnx_sequential(self, mock_model_proto):
        return ONNXSequential(mock_model_proto)

    def test_init(self, onnx_sequential, mock_model_proto):
        assert len(onnx_sequential.operators) == 1
        assert onnx_sequential.operators[0] == mock_model_proto

    def test_load_op(self, onnx_sequential, mock_model_proto):
        # Test loading a ModelProto object
        model = onnx_sequential._load_op(mock_model_proto)
        assert model == mock_model_proto

    def test_combine_models(self, mock_model_proto):
        from unittest.mock import patch

        from onnx.helper import make_graph, make_model, make_node, make_tensor_value_info

        # The patch must wrap ONNXSequential() construction so merge_models is mocked
        # when _combine() actually calls it.
        with patch("onnx.compose.merge_models") as mock_merge_models:
            # Create a small ONNX model as the return value of merge_models
            input_info = make_tensor_value_info("input", onnx.TensorProto.FLOAT, [1, 2])
            output_info = make_tensor_value_info("output", onnx.TensorProto.FLOAT, [1, 2])
            node = make_node("Identity", ["input"], ["output"])
            graph = make_graph([node], "combined_graph", [input_info], [output_info])
            op = onnx.OperatorSetIdProto()
            opset_version = 17
            ir_version = 10
            op.version = opset_version
            combined_model = make_model(graph, opset_imports=[op], ir_version=ir_version)

            mock_merge_models.return_value = combined_model

            # Test combining multiple ONNX models with io_maps
            onnx_sequential = ONNXSequential(
                mock_model_proto,
                mock_model_proto,
                io_maps=[[("output", "input")]],  # list-of-list-of-tuples format
            )
            combined_op = onnx_sequential._combined_op

        assert isinstance(combined_op, onnx.ModelProto)

    def test_export_combined_model(self, onnx_sequential):
        from unittest.mock import patch

        with patch("onnx.save") as mock_save:
            # Test exporting the combined ONNX model
            onnx_sequential.export("exported_model.onnx")
            mock_save.assert_called_once_with(onnx_sequential._combined_op, "exported_model.onnx")

    def test_create_session(self, onnx_sequential):
        from unittest.mock import patch

        with patch("onnxruntime.InferenceSession") as mock_inference_session:
            # Test creating an ONNXRuntime session
            session = onnx_sequential.create_session()
            assert session == mock_inference_session()

    def test_set_get_session(self, onnx_sequential):
        from unittest.mock import MagicMock

        import onnxruntime as ort

        # Test setting and getting a custom session
        mock_session = MagicMock(spec=ort.InferenceSession)
        onnx_sequential.set_session(mock_session)
        assert onnx_sequential.get_session() == mock_session


def _opsets(op):
    # merge_models can repeat an opset import, so compare the set of default-domain versions.
    return {entry.version for entry in op.opset_import if entry.domain in ("", "ai.onnx")}


class TestONNXSequentialOfKorniaExports:
    """Chains of graphs exported by ``to_onnx``, run in onnxruntime against eager."""

    @pytest.fixture(autouse=True)
    def _exporter(self):
        pytest.importorskip("onnxruntime")
        if torch_version_ge(2, 9, 0):
            pytest.importorskip("onnxscript")  # the dynamo exporter, the default from torch 2.9

    def test_auto_version_conversion_accepts_kornia_exports(self):
        from kornia.core import ImageSequential
        from kornia.filters import GaussianBlur2d

        # Static shapes: the legacy exporter cannot export GaussianBlur2d with a dynamic dimension, even the batch
        # alone (#5222).
        shapes = {"input_shape": [2, 1, 16, 16], "output_shape": [2, 1, 16, 16]}
        blur = ImageSequential(GaussianBlur2d((3, 3), (1.5, 1.5)))
        sigmoid = ImageSequential(torch.nn.Sigmoid())
        ops = [blur.to_onnx(save=False, **shapes), sigmoid.to_onnx(save=False, **shapes)]

        seq = ONNXSequential(*ops, auto_ir_version_conversion=True)

        x = torch.rand(2, 1, 16, 16)
        assert_close(torch.from_numpy(seq(x.numpy())[0]), sigmoid(blur(x)))
        assert _opsets(seq._combined_op) == _opsets(ops[0]) == _opsets(ops[1])

    def test_default_target_opset_is_the_highest_among_the_models(self):
        from onnx import version_converter

        from kornia.core import ImageSequential

        op = ImageSequential(torch.nn.Sigmoid()).to_onnx(
            save=False, input_shape=[-1, 3, -1, -1], output_shape=[-1, 3, -1, -1]
        )
        (version,) = _opsets(op)
        newer = version_converter.convert_version(op, version + 1)

        seq = ONNXSequential(op, newer, auto_ir_version_conversion=True)

        # The older graph is brought up to the newer one's opset rather than both to a fixed one.
        assert _opsets(seq._combined_op) == {version + 1}
        x = torch.rand(1, 3, 4, 5)
        assert_close(torch.from_numpy(seq(x.numpy())[0]), x.sigmoid().sigmoid())
