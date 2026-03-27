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

onnx = pytest.importorskip("onnx")

from kornia.onnx.sequential import ONNXSequential


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
        model = make_model(graph, opset_imports=[op], ir_version=9)
        return model

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
            io_maps=[("output", "input")],  # Dummy io_maps
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
