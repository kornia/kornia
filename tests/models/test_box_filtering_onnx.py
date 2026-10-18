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

"""ONNX export tests for BoxFiltering (#6077)."""

import pytest
import torch

from kornia.contrib.object_detection import BoxFiltering
from kornia.core._compat import torch_version_ge, torch_version_lt

from testing.base import DYNAMO_UNAVAILABLE_REASON, BaseTester, dynamo_is_available

# ONNX Runtime's CPU provider is exercised regardless of the selected device or dtype.
pytestmark = pytest.mark.device_agnostic

_EXPORTERS = [
    pytest.param(False, id="torchscript"),
    pytest.param(
        True,
        # conftest.py deselects IDs containing "dynamo" unless KORNIA_TEST_OPTIMIZER is set.
        id="torch_export",
        marks=[
            pytest.mark.skipif(not dynamo_is_available(), reason=DYNAMO_UNAVAILABLE_REASON),
            pytest.mark.skipif(torch_version_lt(2, 9, 0), reason="to_onnx uses dynamic_shapes from torch 2.9"),
            pytest.mark.xfail(
                torch_version_ge(2, 14) and torch_version_lt(2, 15, 0),
                reason=(
                    "torch 2.14's CPU ONNX exporter cannot translate aten.mul.Scalar in the class mask: "
                    "No decompositions registered for the real-valued input; "
                    "also affects test_sam_onnx.py::test_sam_encoder_to_onnx"
                ),
                raises=torch.onnx.OnnxExporterError,
                strict=torch.version.cuda is None,
            ),
        ],
    ),
]


def _detections(batch: int, count: int) -> torch.Tensor:
    return torch.rand(batch, count, 6, generator=torch.Generator().manual_seed(0))


def _session(exported):
    onnx = pytest.importorskip("onnx")
    ort = pytest.importorskip("onnxruntime")
    onnx.checker.check_model(exported)
    return ort.InferenceSession(exported.SerializeToString(), providers=["CPUExecutionProvider"])


class TestBoxFilteringONNX(BaseTester):
    @pytest.mark.timeout(120)
    @pytest.mark.parametrize("dynamo", _EXPORTERS)
    @pytest.mark.parametrize("pseudo_shape", [None, [5, 20, 6]], ids=["default_pseudo", "pseudo_batch_5"])
    def test_default_list_export_fixes_the_batch_6077(self, dynamo, pseudo_shape):
        pytest.importorskip("onnx")
        pytest.importorskip("onnxscript")
        ort = pytest.importorskip("onnxruntime")

        module = BoxFiltering(confidence_threshold=0.5)
        exported = module.to_onnx(save=False, pseudo_shape=pseudo_shape, dynamo=dynamo)

        dims = exported.graph.input[0].type.tensor_type.shape.dim
        assert dims[0].dim_value == 1
        assert dims[1].dim_param
        assert len(exported.graph.output) == 1
        assert len(exported.graph.output[0].type.tensor_type.shape.dim) == 2
        assert pseudo_shape is None or pseudo_shape == [5, 20, 6]  # Caller-owned shapes are not rewritten in place.

        session = _session(exported)
        for count in (7, 20, 33):  # The trace has 20 detections; the runtime count must remain dynamic.
            boxes = _detections(1, count)
            (actual,) = session.run(None, {"input": boxes.numpy()})
            self.assert_close(torch.from_numpy(actual), module(boxes)[0])

        boxes[..., 1] = 0.1
        (actual,) = session.run(None, {"input": boxes.numpy()})
        assert actual.shape == (0, 6)

        for batch in (3, 5, 7):
            with pytest.raises(ort.capi.onnxruntime_pybind11_state.InvalidArgument, match="Expected: 1"):
                session.run(None, {"input": _detections(batch, 20).numpy()})

    @pytest.mark.parametrize("input_shape", [[-1, -1, 6], [-1, 20, 6]])
    def test_list_export_rejects_an_explicit_dynamic_batch_6077(self, input_shape):
        with pytest.raises(ValueError, match="filter_as_zero=True"):
            BoxFiltering(confidence_threshold=0.5).to_onnx(save=False, input_shape=input_shape)

    @pytest.mark.timeout(120)
    @pytest.mark.parametrize("dynamo", _EXPORTERS)
    @pytest.mark.parametrize("pseudo_shape", [None, [3, 20, 6]])
    def test_list_export_preserves_an_explicit_fixed_batch_6077(self, dynamo, pseudo_shape):
        pytest.importorskip("onnx")
        pytest.importorskip("onnxscript")

        module = BoxFiltering(confidence_threshold=0.5)
        exported = module.to_onnx(save=False, input_shape=[3, -1, 6], pseudo_shape=pseudo_shape, dynamo=dynamo)
        assert exported.graph.input[0].type.tensor_type.shape.dim[0].dim_value == 3
        assert len(exported.graph.output) == 3

        boxes = _detections(3, 11)
        boxes[0, :, 1] = 0.1  # Empty, partially retained, and fully retained images in the same batch.
        boxes[2, :, 1] = 0.9
        actual = _session(exported).run(None, {"input": boxes.numpy()})
        expected = module(boxes)
        assert len(actual) == len(expected)
        for got, want in zip(actual, expected):
            self.assert_close(torch.from_numpy(got), want)

    def test_explicit_shapes_must_agree(self):
        with pytest.raises(ValueError, match="input_shape"):
            BoxFiltering(confidence_threshold=0.5).to_onnx(save=False, input_shape=[3, -1, 6], pseudo_shape=[5, 20, 6])

    @pytest.mark.timeout(120)
    @pytest.mark.parametrize("dynamo", _EXPORTERS)
    def test_filter_as_zero_keeps_a_dynamic_batch_6077(self, dynamo):
        pytest.importorskip("onnx")
        pytest.importorskip("onnxscript")

        module = BoxFiltering(confidence_threshold=0.5, filter_as_zero=True)
        exported = module.to_onnx(save=False, dynamo=dynamo)
        assert exported.graph.input[0].type.tensor_type.shape.dim[0].dim_param
        assert len(exported.graph.output) == 1

        session = _session(exported)
        # The trace has a batch of 5 and 20 detections.
        for batch, count in ((1, 20), (3, 7), (5, 20), (7, 33)):
            boxes = _detections(batch, count)
            (actual,) = session.run(None, {"input": boxes.numpy()})
            self.assert_close(torch.from_numpy(actual), module(boxes))
