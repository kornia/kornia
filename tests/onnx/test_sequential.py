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

from unittest.mock import MagicMock, patch

import onnx
import onnxruntime as ort
import pytest
from onnx.helper import make_graph, make_model, make_node, make_tensor_value_info

from kornia.onnx.sequential import ONNXSequential


class TestONNXSequential:
    @pytest.fixture
    def mock_model_proto(self):
        # Create a minimal ONNX model with an input and output
        input_info = make_tensor_value_info("input", onnx.TensorProto.FLOAT, [1, 2])
        output_info = make_tensor_value_info("output", onnx.TensorProto.FLOAT, [1, 2])
        node = make_node("Identity", ["input"], ["output"])
        graph = make_graph([node], "test_graph", [input_info], [output_info])
        op = onnx.OperatorSetIdProto()
        op.version = 17
        model = make_model(graph, opset_imports=[op])
        return model

    @pytest.fixture
    def onnx_sequential(self, mock_model_proto):
        # Return an ONNXSequential instance with mocked models
        return ONNXSequential(mock_model_proto, mock_model_proto, auto_ir_version_conversion=True, target_ir_version=10)

    def test_load_op_from_proto(self, mock_model_proto, onnx_sequential):
        # Test loading a model from an ONNX ModelProto object
        model = onnx_sequential._load_op(mock_model_proto)
        assert model == mock_model_proto

    @patch("onnx.compose.merge_models")
    def test_combine_models(self, mock_merge_models, mock_model_proto):
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
            auto_ir_version_conversion=True,
            target_ir_version=ir_version,
            target_opset_version=opset_version,
        )
        combined_op = onnx_sequential.combine([("output1", "input2")])

        assert isinstance(combined_op, onnx.ModelProto)

    @patch("onnx.save")
    def test_export_combined_model(self, mock_save, onnx_sequential):
        # Test exporting the combined ONNX model
        onnx_sequential.export("exported_model.onnx")
        mock_save.assert_called_once_with(onnx_sequential._combined_op, "exported_model.onnx")

    @patch("onnxruntime.InferenceSession")
    def test_create_session(self, mock_inference_session, onnx_sequential):
        # Test creating an ONNXRuntime session
        session = onnx_sequential.create_session()
        assert session == mock_inference_session()

    def test_set_get_session(self, onnx_sequential):
        # Test setting and getting a custom session
        mock_session = MagicMock(spec=ort.InferenceSession)
        onnx_sequential.set_session(mock_session)
        assert onnx_sequential.get_session() == mock_session
