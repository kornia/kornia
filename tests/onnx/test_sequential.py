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

    @pytest.mark.parametrize("auto_ir_version_conversion", [False, True])
    def test_chains_exports_from_another_kornia_release(self, auto_ir_version_conversion):
        # #6117: every to_onnx export carries kornia's "version", so a graph exported by another release refused to
        # chain: "Can't merge models with different values for the same model metadata property".
        from kornia.core import ImageSequential
        from kornia.onnx import add_metadata

        sigmoid, tanh = ImageSequential(torch.nn.Sigmoid()), ImageSequential(torch.nn.Tanh())
        fresh = sigmoid.to_onnx(save=False, **_RGB)
        older = add_metadata(tanh.to_onnx(save=False, **_RGB), [("version", "0.8.1")])

        seq = ONNXSequential(fresh, older, auto_ir_version_conversion=auto_ir_version_conversion)

        x = torch.rand(2, 3, 5, 7)
        assert_close(torch.from_numpy(seq(x.numpy())[0]), tanh(sigmoid(x)))
        assert _props(seq._combined_op) == [("source", "kornia"), ("version", "0.8.1")]

    def test_chains_exports_with_different_additional_metadata(self):
        from kornia.core import ImageSequential

        sigmoid, tanh = ImageSequential(torch.nn.Sigmoid()), ImageSequential(torch.nn.Tanh())
        a = sigmoid.to_onnx(save=False, additional_metadata=[("name", "sigmoid")], **_RGB)
        b = tanh.to_onnx(save=False, additional_metadata=[("name", "tanh")], **_RGB)

        seq = ONNXSequential(a, b)

        x = torch.rand(1, 3, 4, 6)
        assert_close(torch.from_numpy(seq(x.numpy())[0]), tanh(sigmoid(x)))
        assert dict(_props(seq._combined_op))["name"] == "tanh"
        assert dict(_props(a))["name"] == "sigmoid"  # the caller's graphs keep their own metadata

    @pytest.mark.parametrize("auto_ir_version_conversion", [False, True])
    def test_target_versions_are_applied_without_the_conversion_flag(self, auto_ir_version_conversion):
        # #6118: target_opset_version / target_ir_version were read only with auto_ir_version_conversion=True.
        from kornia.core import ImageSequential

        sigmoid = ImageSequential(torch.nn.Sigmoid())
        op = sigmoid.to_onnx(save=False, **_RGB)
        # IR 10 from the dynamo exporter, 8 from the legacy one (the default before torch 2.9): 9 converts both
        versions = (_opsets(op), op.ir_version)
        assert versions[0] == {18}
        assert versions[1] != 9

        seq = ONNXSequential(
            op,
            op,
            target_opset_version=17,
            target_ir_version=9,
            auto_ir_version_conversion=auto_ir_version_conversion,
        )

        for graph in [*seq.operators, seq._combined_op]:
            assert (_opsets(graph), graph.ir_version) == ({17}, 9)
        assert (_opsets(op), op.ir_version) == versions  # the caller's graph is converted on a copy
        x = torch.rand(2, 3, 4, 5)
        assert_close(torch.from_numpy(seq(x.numpy())[0]), sigmoid(sigmoid(x)))

    def test_one_target_version_without_the_flag_keeps_the_other(self):
        # as ONNXModule does: without auto_ir_version_conversion, only the versions given are changed
        from kornia.core import ImageSequential

        op = ImageSequential(torch.nn.Sigmoid()).to_onnx(save=False, **_RGB)
        assert op.ir_version != 9  # 10 from the dynamo exporter, 8 from the legacy one

        ir_only = ONNXSequential(op, op, target_ir_version=9)._combined_op
        opset_only = ONNXSequential(op, op, target_opset_version=17)._combined_op

        assert (_opsets(ir_only), ir_only.ir_version) == ({18}, 9)
        assert (_opsets(opset_only), opset_only.ir_version) == ({17}, op.ir_version)

    def test_takes_file_paths_as_path_objects(self, tmp_path):
        from kornia.core import ImageSequential

        sigmoid = ImageSequential(torch.nn.Sigmoid())
        path = tmp_path / "sigmoid.onnx"
        sigmoid.to_onnx(str(path), **_RGB)

        seq = ONNXSequential(path, path)

        x = torch.rand(1, 3, 4, 5)
        assert_close(torch.from_numpy(seq(x.numpy())[0]), sigmoid(sigmoid(x)))


_RGB = {"input_shape": [-1, 3, -1, -1], "output_shape": [-1, 3, -1, -1]}


def _props(op):
    return [(prop.key, prop.value) for prop in op.metadata_props]


def _vector(name, elem_type=None):
    from onnx.helper import make_tensor_value_info

    return make_tensor_value_info(name, onnx.TensorProto.FLOAT if elem_type is None else elem_type, [2])


def _model(nodes, inputs, outputs, initializers=(), metadata=()):
    from onnx.helper import make_graph, make_model, make_opsetid

    graph = make_graph(nodes, "g", inputs, outputs, initializer=list(initializers))
    model = make_model(graph, opset_imports=[make_opsetid("", 18)], ir_version=9)
    for key, value in metadata:
        model.metadata_props.add(key=key, value=value)
    return model


def _identity(*metadata, elem_type=None):
    from onnx.helper import make_node

    return _model(
        [make_node("Identity", ["input"], ["output"])],
        [_vector("input", elem_type)],
        [_vector("output", elem_type)],
        metadata=metadata,
    )


class TestONNXSequentialComposition:
    """Real merges of hand-built graphs: names, metadata and the checks at each joint."""

    @pytest.fixture(autouse=True)
    def _runtime(self):
        pytest.importorskip("onnxruntime")

    @pytest.fixture
    def three_graphs(self):
        from onnx.helper import make_node
        from onnx.numpy_helper import from_array

        two = from_array(torch.tensor([2.0, 2.0]).numpy(), "two")
        first = _model(
            [make_node("Neg", ["x"], ["neg"]), make_node("Abs", ["x"], ["abs"])],
            [_vector("x")],
            [_vector("neg"), _vector("abs")],
        )
        second = _model([make_node("Sub", ["p", "q"], ["diff"])], [_vector("p"), _vector("q")], [_vector("diff")])
        third = _model([make_node("Mul", ["d", "two"], ["y"])], [_vector("d")], [_vector("y")], initializers=[two])
        return first, second, third

    @pytest.mark.parametrize(
        ("joint", "expected"),
        [([("neg", "p"), ("abs", "q")], [-4.0, 0.0]), ([("neg", "q"), ("abs", "p")], [4.0, 0.0])],
    )
    def test_io_maps_connect_three_graphs(self, three_graphs, joint, expected):
        # 2 * (p - q) of neg(x) and abs(x): the pairs of each joint decide which output feeds which input.
        before = [graph.SerializeToString() for graph in three_graphs]

        seq = ONNXSequential(*three_graphs, io_maps=[joint, [("diff", "d")]])

        assert seq(torch.tensor([1.0, -2.0]).numpy())[0].tolist() == expected
        assert [value.name for value in seq._combined_op.graph.input] == ["K00-x"]
        assert [value.name for value in seq._combined_op.graph.output] == ["K02-y"]
        assert [graph.SerializeToString() for graph in three_graphs] == before

    @pytest.mark.parametrize("auto_ir_version_conversion", [False, True])
    def test_metadata_are_merged_with_the_later_value_of_each_key(self, auto_ir_version_conversion):
        # every pair of models disagrees on a key, the second and third one too
        graphs = [
            _identity(("a", "1"), ("b", "2")),
            _identity(("b", "3"), ("c", "4")),
            _identity(("a", "5"), ("c", "6")),
        ]
        before = [graph.SerializeToString() for graph in graphs]

        seq = ONNXSequential(*graphs, auto_ir_version_conversion=auto_ir_version_conversion)

        # each key keeps its first position and takes its last value, as add_metadata writes them
        assert _props(seq._combined_op) == [("a", "5"), ("b", "3"), ("c", "6")]
        assert [graph.SerializeToString() for graph in graphs] == before
        onnx.checker.check_model(seq._combined_op)
        assert seq(torch.tensor([1.0, 2.0]).numpy())[0].tolist() == [1.0, 2.0]

    def test_element_type_mismatch_at_a_joint_raises(self):
        # The merged graph drops the later model's declared input, so nothing compared the two types any more.
        graphs = [_identity(), _identity(elem_type=onnx.TensorProto.DOUBLE)]

        with pytest.raises(
            ValueError, match=r"output 'output' of model 0 is FLOAT.*input 'input' of model 1, which is DOUBLE"
        ):
            ONNXSequential(*graphs)

    def test_element_types_are_compared_through_io_maps(self, three_graphs):
        from onnx.helper import make_node

        first, _, third = three_graphs
        second = _model(
            [make_node("Cast", ["p"], ["diff"], to=onnx.TensorProto.FLOAT)],
            [_vector("p", onnx.TensorProto.INT64)],
            [_vector("diff")],
        )
        with pytest.raises(ValueError, match=r"output 'abs' of model 0 is FLOAT.*input 'p' of model 1, which is INT64"):
            ONNXSequential(first, second, third, io_maps=[[("abs", "p")], [("diff", "d")]])

    @pytest.mark.parametrize(
        ("models", "io_maps"),
        [
            (2, [("output", "input")]),  # the pairs of the only joint, not wrapped in a list
            (2, [("o1", "i1")]),  # the same with two-character names, which have a length of 2 like a pair
            (3, [("output", "input"), ("output", "input")]),  # one pair per joint instead of one list per joint
            (2, ["output"]),
            (2, [None]),
            (2, [[("output",)]]),
            (2, [[("output", "input", "input")]]),
            (2, [["output", "input"]]),
        ],
    )
    def test_malformed_io_maps_raise_a_corrective_message(self, models, io_maps):
        from kornia.core.exceptions import BaseError

        with pytest.raises(BaseError, match=r"io_maps\[0\] must be a list of \(output, input\) name pairs"):
            ONNXSequential(*[_identity() for _ in range(models)], io_maps=io_maps)

    def test_io_maps_without_an_entry_for_every_joint_raise(self):
        from kornia.core.exceptions import BaseError

        with pytest.raises(
            BaseError, match=r"one list of \(output, input\) name pairs per joint: 2 for 3 models, got 1"
        ):
            ONNXSequential(_identity(), _identity(), _identity(), io_maps=[[("output", "input")]])

    def test_io_maps_entries_past_the_last_joint_are_not_used(self):
        # they were never read, and a chain built with them stays as it was
        seq = ONNXSequential(_identity(), _identity(), io_maps=[[["output", "input"]], [("unused", "unused")]])

        assert seq(torch.tensor([1.0, 2.0]).numpy())[0].tolist() == [1.0, 2.0]
