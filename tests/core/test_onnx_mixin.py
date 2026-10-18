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

"""Export and runtime contract of the ONNX mixins, checked end to end through onnxruntime."""

import copy

import pytest
import torch
from torch import nn

from kornia.core import ImageSequential
from kornia.core._compat import torch_version_ge
from kornia.core.mixin.onnx import ONNXMixin, ONNXRuntimeMixin
from kornia.filters import GaussianBlur2d

from testing.base import BaseTester

onnx = pytest.importorskip("onnx")
ort = pytest.importorskip("onnxruntime")
if torch_version_ge(2, 9, 0):
    # From torch 2.9 ``torch.onnx.export`` defaults to the dynamo exporter, which needs onnxscript.
    pytest.importorskip("onnxscript")

# The opset ``to_onnx`` documents and requests.
DOCUMENTED_OPSET = 18

_RGB = {"input_shape": [-1, 3, -1, -1], "output_shape": [-1, 3, -1, -1]}


def _default_domain_opset(op):
    return next(entry.version for entry in op.opset_import if entry.domain in ("", "ai.onnx"))


def _input_dims(op):
    return [d.dim_param or d.dim_value for d in op.graph.input[0].type.tensor_type.shape.dim]


def _run(op, x):
    session = ort.InferenceSession(op.SerializeToString(), providers=["CPUExecutionProvider"])
    return torch.from_numpy(session.run(None, {session.get_inputs()[0].name: x.numpy()})[0])


@pytest.mark.device_agnostic
class TestONNXExportMixin(BaseTester):
    def test_train_mode_model_exports_the_eval_graph(self):
        torch.manual_seed(0)
        model = ImageSequential(nn.Conv2d(3, 3, 3, padding=1), nn.BatchNorm2d(3), nn.Dropout(0.5))
        model.train()
        model[1].eval()  # a mixed state, which must come back exactly
        op = model.to_onnx(save=False, **_RGB)

        assert "Dropout" not in {node.op_type for node in op.graph.node}
        assert [m.training for m in (model, *model)] == [True, True, False, True]

        x = torch.rand(2, 3, 8, 8)
        with torch.no_grad():
            expected = copy.deepcopy(model).eval()(x)
        self.assert_close(_run(op, x), expected)

    def test_failed_export_restores_the_training_mode(self, monkeypatch):
        model = ImageSequential(nn.Conv2d(3, 3, 1), nn.Dropout(0.5))
        model.train()
        model[0].eval()  # a mixed state: ``model.train(True)`` in place of an exact restore would lose it

        def _fail(*args, **kwargs):
            raise RuntimeError("export failed")

        monkeypatch.setattr(torch.onnx, "export", _fail)
        with pytest.raises(RuntimeError, match="export failed"):
            model.to_onnx(save=False, **_RGB)
        assert [m.training for m in (model, *model)] == [True, False, True]

    def test_explicit_training_mode_is_the_legacy_exporters_opt_out(self):
        # The documented opt-out from the eval-mode export: an explicit ``training=TRAINING`` reaches the legacy
        # exporter, which traces the training graph. The flags still come back exactly.
        model = ImageSequential(nn.Conv2d(3, 3, 1), nn.Dropout(0.5))
        model[0].eval()
        op = model.to_onnx(save=False, dynamo=False, training=torch.onnx.TrainingMode.TRAINING, **_RGB)

        assert "Dropout" in {node.op_type for node in op.graph.node}
        assert [m.training for m in (model, *model)] == [True, False, True]

    @pytest.mark.skipif(
        not torch_version_ge(2, 9, 0), reason="the torch.export-based exporter is the default from torch 2.9"
    )
    def test_default_exporter_from_torch_2_9_ignores_the_training_keyword(self):
        model = ImageSequential(nn.Conv2d(3, 3, 1), nn.Dropout(0.5))
        model.train()
        op = model.to_onnx(save=False, training=torch.onnx.TrainingMode.TRAINING, **_RGB)

        assert "Dropout" not in {node.op_type for node in op.graph.node}
        assert model.training

    @pytest.mark.parametrize(
        "channels, module",
        [(3, lambda: nn.Conv2d(3, 3, 1)), (1, lambda: GaussianBlur2d((3, 3), (1.5, 1.5)))],
        ids=["conv", "gaussian_blur"],
    )
    def test_saved_model_has_the_documented_opset(self, tmp_path, channels, module):
        """Test that the saved ONNX model uses the documented opset."""
        # The dynamo exporter builds opset 18 and down-converts a lower request only when every op has an adapter, which
        # a convolution has and the blur's ``Pad`` has not.
        path = tmp_path / "model.onnx"
        shape = [2, channels, 16, 16]
        op = ImageSequential(module()).to_onnx(onnx_name=str(path), input_shape=shape, output_shape=shape)
        assert _default_domain_opset(op) == DOCUMENTED_OPSET
        assert _default_domain_opset(onnx.load(str(path))) == DOCUMENTED_OPSET

    def test_default_input_shape_exports_a_fixed_channel_model(self):
        model = ImageSequential(nn.Conv2d(3, 3, 1))
        op = model.to_onnx(save=False)

        dims = _input_dims(op)
        assert dims[1] == 3
        assert all(isinstance(d, str) for d in (dims[0], dims[2], dims[3]))

        x = torch.rand(2, 3, 5, 7)
        with torch.no_grad():
            expected = model(x)
        self.assert_close(_run(op, x), expected)

    def test_pseudo_shape_alone_sets_the_channel_count(self):
        # Without input_shape, the default's fixed channel count follows an explicit pseudo shape.
        model = ImageSequential(nn.Conv2d(1, 1, 1))
        op = model.to_onnx(save=False, pseudo_shape=[1, 1, 32, 32])

        x = torch.rand(2, 1, 5, 7)
        with torch.no_grad():
            expected = model(x)
        self.assert_close(_run(op, x), expected)

        dims = _input_dims(op)
        assert dims[1] == 1
        assert all(isinstance(d, str) for d in (dims[0], dims[2], dims[3]))

    def test_input_shape_must_agree_with_an_explicit_pseudo_shape(self):
        with pytest.raises(ValueError, match="input_shape"):
            ImageSequential(nn.Sigmoid()).to_onnx(save=False, input_shape=[-1, 3, -1, -1], pseudo_shape=[1, 1, 32, 32])

    def test_input_shape_rank_must_match_the_pseudo_shape(self):
        with pytest.raises(ValueError, match="pseudo shape"):
            ImageSequential(nn.Identity()).to_onnx(save=False, input_shape=[-1, 3, -1, -1, -1])

    def test_missing_output_directory_is_reported_before_exporting(self, tmp_path, monkeypatch):
        def _must_not_export(*args, **kwargs):
            raise AssertionError("the export ran before the output directory was checked")

        monkeypatch.setattr(torch.onnx, "export", _must_not_export)
        with pytest.raises(FileNotFoundError, match="no_such_dir"):
            ImageSequential(nn.Identity()).to_onnx(onnx_name=str(tmp_path / "no_such_dir" / "x.onnx"), **_RGB)

    @pytest.mark.parametrize(
        "input_shape, pseudo_shape, traced",
        [
            ([-1, 3, -1, -1], [1, 3, 1, 32], [2, 3, 2, 32]),
            ([-1, 3, 16, -1], [0, 3, 16, 5], [2, 3, 16, 5]),
            ([1, 3, 16, 16], [1, 3, 16, 16], [1, 3, 16, 16]),
        ],
        ids=["sizes_of_one", "size_of_zero", "fixed_dimensions"],
    )
    def test_dynamic_dimensions_are_traced_at_two_or_more_6066(self, monkeypatch, input_shape, pseudo_shape, traced):
        # torch.export specializes an example size of 0 or 1. On torch 2.9, DexiNed exported from a batch of 1 kept
        # that batch inside its Resize nodes and failed in ONNX Runtime at any other batch size. Fixed dimensions keep
        # theirs.
        traced_shapes = []

        def _capture(model, args, *rest, **kwargs):
            traced_shapes.append(list(args.shape))
            raise RuntimeError("captured")

        monkeypatch.setattr(torch.onnx, "export", _capture)
        with pytest.raises(RuntimeError, match="captured"):
            ImageSequential(nn.Identity()).to_onnx(save=False, input_shape=input_shape, pseudo_shape=pseudo_shape)
        assert traced_shapes == [traced]


class _ChannelAffine(nn.Module):
    """A per-channel affine map, with its weights as parameters or as buffers.

    Its ops run in float16, float32 and float64 on onnxruntime's CPU provider.
    """

    def __init__(self, as_buffers: bool = False) -> None:
        super().__init__()
        weight = torch.linspace(0.5, 1.5, 3).view(1, 3, 1, 1)
        bias = torch.linspace(-0.25, 0.25, 3).view(1, 3, 1, 1)
        if as_buffers:
            self.register_buffer("weight", weight)
            self.register_buffer("bias", bias)
        else:
            self.weight = nn.Parameter(weight)
            self.bias = nn.Parameter(bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.weight + self.bias


class TestONNXExportMixinDtypeDevice(BaseTester):
    @pytest.mark.parametrize("as_buffers", [False, True], ids=["parameters", "buffers"])
    def test_dummy_input_follows_the_model_dtype_and_device(self, device, dtype, as_buffers):
        model = ImageSequential(_ChannelAffine(as_buffers)).to(device, dtype)
        assert len(list(model.parameters())) == (0 if as_buffers else 2)
        op = model.to_onnx(save=False, **_RGB)

        elem_types = {
            torch.float16: onnx.TensorProto.FLOAT16,
            torch.bfloat16: onnx.TensorProto.BFLOAT16,
            torch.float32: onnx.TensorProto.FLOAT,
            torch.float64: onnx.TensorProto.DOUBLE,
        }
        assert op.graph.input[0].type.tensor_type.elem_type == elem_types[dtype]
        if dtype == torch.bfloat16:
            return  # onnxruntime's CPU provider has no bfloat16 Mul, and numpy no bfloat16 to feed it

        x = torch.linspace(0, 1, 2 * 3 * 5 * 7, device=device, dtype=dtype).view(2, 3, 5, 7)
        with torch.no_grad():
            expected = model(x)
        self.assert_close(_run(op, x.cpu()), expected.cpu())


@pytest.mark.device_agnostic
class TestONNXRuntimeMixin(BaseTester):
    @pytest.fixture
    def op(self):
        return ImageSequential(nn.Sigmoid()).to_onnx(save=False, **_RGB)

    def test_session_options_are_used(self, op):
        from kornia.onnx import ONNXModule, ONNXSequential

        def _options():
            options = ort.SessionOptions()
            options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
            return options

        sessions = [
            ONNXRuntimeMixin()._create_session(op, session_options=_options()),
            ONNXSequential(op, session_options=_options()).get_session(),
            ONNXModule(op, session_options=_options()).get_session(),
        ]
        for session in sessions:
            level = session.get_session_options().graph_optimization_level
            assert level == ort.GraphOptimizationLevel.ORT_DISABLE_ALL

        x = torch.rand(1, 3, 4, 5)
        self.assert_close(torch.from_numpy(ONNXSequential(op, session_options=_options())(x.numpy())[0]), x.sigmoid())

    def test_runtime_rejects_wrong_input_count(self, op):
        from kornia.onnx import ONNXSequential

        runtime = ONNXSequential(op)
        x = torch.rand(1, 3, 4, 5).numpy()
        with pytest.raises(ValueError, match="Expected 1 ONNX inputs, got 0"):
            runtime()
        with pytest.raises(ValueError, match="Expected 1 ONNX inputs, got 2"):
            runtime(x, x)

    def test_runtime_feeds_inputs_in_session_order(self):
        from kornia.onnx import ONNXModule, ONNXSequential

        def _value(name):
            return onnx.helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, [1])

        node = onnx.helper.make_node("Sub", ["x", "y"], ["output"])
        graph = onnx.helper.make_graph([node], "sub", [_value("x"), _value("y")], [_value("output")])
        op = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 17)], ir_version=9)
        a, b = torch.tensor([2.0]).numpy(), torch.tensor([99.0]).numpy()
        for runtime in (ONNXSequential(op), ONNXModule(op)):
            assert runtime(a, b)[0].tolist() == [-97.0]
            assert runtime(b, a)[0].tolist() == [97.0]
            with pytest.raises(ValueError, match="Expected 2 ONNX inputs, got 3"):
                runtime(a, b, a)
            # torch tensors on the CPU and numpy arrays mix
            assert runtime(torch.tensor([2.0]), b)[0].tolist() == [-97.0]

    def test_runtime_takes_cpu_torch_tensors(self, op):
        import numpy as np

        from kornia.onnx import ONNXModule, ONNXSequential

        x = torch.rand(2, 3, 5, 4)
        inputs = [
            x,
            x.clone().requires_grad_(True),
            torch.rand(2, 3, 4, 5).transpose(-1, -2),  # not contiguous
        ]
        for runtime in (ONNXSequential(op), ONNXModule(op)):
            for value in inputs:
                (out,) = runtime(value)
                assert isinstance(out, np.ndarray)
                assert out.tolist() == runtime(np.ascontiguousarray(value.detach().numpy()))[0].tolist()
                self.assert_close(torch.from_numpy(out), value.detach().sigmoid())

    def test_runtime_rejects_tensors_off_the_cpu(self, op):
        from kornia.onnx import ONNXModule, ONNXSequential

        x = torch.empty(1, 3, 4, 5, device="meta")
        for runtime in (ONNXSequential(op), ONNXModule(op)):
            with pytest.raises(ValueError, match=r"input 0 is a torch tensor on meta.*Move it to the CPU"):
                runtime(x)

    @pytest.fixture
    def subtract(self):
        from kornia.onnx import ONNXModule

        def _value(name):
            return onnx.helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, [2])

        node = onnx.helper.make_node("Sub", ["x", "y"], ["output"])
        graph = onnx.helper.make_graph([node], "sub", [_value("x"), _value("y")], [_value("output")])
        return ONNXModule(onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 17)], ir_version=9))

    def test_runtime_names_the_input_off_the_cpu(self, subtract):
        with pytest.raises(ValueError, match=r"ONNX input 1 is a torch tensor on meta"):
            subtract(torch.tensor([3.0, 1.0]), torch.empty(2, device="meta"))

    def test_runtime_rejects_bfloat16_tensors_naming_the_input(self, subtract):
        # numpy has no bfloat16
        a = torch.tensor([3.0, 1.0])
        with pytest.raises(TypeError, match=r"ONNX input 1 is a torch.bfloat16 tensor, and numpy has no bfloat16"):
            subtract(a.numpy(), a.bfloat16())

    def test_runtime_rejects_sparse_tensors_naming_the_input(self, subtract):
        a = torch.tensor([3.0, 0.0])
        with pytest.raises(TypeError, match=r"ONNX input 0 is a torch.sparse_coo tensor.*\.to_dense\(\)"):
            subtract(a.to_sparse(), a)

    def test_runtime_reads_a_negative_bit_view(self, subtract):
        # .conj().imag returns a view with the negative bit set, which .numpy() refuses: [3, 1] - [-2, 1]
        negative_view = torch.tensor([1 + 2j, 3 - 1j]).conj().imag
        assert negative_view.is_neg()
        assert subtract(torch.tensor([3.0, 1.0]), negative_view)[0].tolist() == [5.0, 0.0]

    def test_add_metadata_overwrites_existing_keys(self, op):
        keys = [prop.key for prop in op.metadata_props]
        assert sorted(keys) == ["source", "version"]

        again = ONNXMixin()._add_metadata(copy.deepcopy(op), [("date", "20240909")])
        again = ONNXMixin()._add_metadata(again, [("date", "20260930")])

        assert sorted(prop.key for prop in again.metadata_props) == ["date", "source", "version"]
        assert {prop.key: prop.value for prop in again.metadata_props}["date"] == "20260930"
        onnx.checker.check_model(again)
