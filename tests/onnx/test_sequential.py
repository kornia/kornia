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
        model = make_model(graph)
        return model

    @pytest.fixture
    def onnx_sequential(self, mock_model_proto):
        # Return an ONNXSequential instance with mocked models
        return ONNXSequential(mock_model_proto, mock_model_proto)

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
        combined_model = make_model(graph)

        mock_merge_models.return_value = combined_model

        # Test combining multiple ONNX models with io_maps
        onnx_sequential = ONNXSequential(mock_model_proto, mock_model_proto)
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
