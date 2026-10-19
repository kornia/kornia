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
from kornia.core._compat import torch_version_lt

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
        ],
    ),
]


def _detections(batch: int, count: int) -> torch.Tensor:
    """Create repeatable CPU detections for comparisons with ONNX Runtime."""
    return torch.rand(batch, count, 6, generator=torch.Generator().manual_seed(0))


def _session(exported):
    """Validate the exported graph and load it with ONNX Runtime's CPU provider."""
    onnx = pytest.importorskip("onnx")
    ort = pytest.importorskip("onnxruntime")
    onnx.checker.check_model(exported)
    return ort.InferenceSession(exported.SerializeToString(), providers=["CPUExecutionProvider"])


class TestBoxFilteringONNX(BaseTester):
    """Check batch contracts and runtime results for both BoxFiltering output modes."""

    @pytest.mark.timeout(120)
    @pytest.mark.parametrize("dynamo", _EXPORTERS)
    @pytest.mark.parametrize(
        ("input_shape", "pseudo_shape", "batch"),
        [(None, None, 1), (None, [3, 20, 6], 3), ([3, -1, 6], None, 3)],
        ids=["default", "pseudo_batch_3", "input_batch_3"],
    )
    def test_list_export_fixes_the_batch_6077(self, dynamo, input_shape, pseudo_shape, batch):
        """Fix the batch in the graph input, keep detection counts dynamic, and reject every other batch."""
        pytest.importorskip("onnx")
        pytest.importorskip("onnxscript")
        ort = pytest.importorskip("onnxruntime")

        module = BoxFiltering(confidence_threshold=0.5)
        exported = module.to_onnx(save=False, input_shape=input_shape, pseudo_shape=pseudo_shape, dynamo=dynamo)

        dims = exported.graph.input[0].type.tensor_type.shape.dim
        assert dims[0].dim_value == batch
        assert dims[1].dim_param
        assert len(exported.graph.output) == batch
        assert [d.dim_value for d in exported.graph.output[0].type.tensor_type.shape.dim][1:] == [6]

        session = _session(exported)
        # The trace has 20 detections; the runtime count must remain dynamic.
        for count, empty_first in ((7, True), (33, False)):
            boxes = _detections(batch, count)
            if empty_first:
                boxes[0, :, 1] = 0.1  # No detection kept.
            if batch > 1:
                boxes[-1, :, 1] = 0.9  # Every detection kept.
            actual = session.run(None, {"input": boxes.numpy()})
            expected = module(boxes)
            assert len(actual) == len(expected) == batch
            for got, want in zip(actual, expected):
                self.assert_close(torch.from_numpy(got), want)

        for other in (2, 5):
            with pytest.raises(ort.capi.onnxruntime_pybind11_state.InvalidArgument, match=f"Expected: {batch}"):
                session.run(None, {"input": _detections(other, 20).numpy()})

    @pytest.mark.parametrize("input_shape", [[-1, -1, 6], [-1, 20, 6]])
    def test_list_export_rejects_a_dynamic_batch_6077(self, input_shape):
        """Point callers asking for a dynamic batch to the tensor-output mode."""
        with pytest.raises(ValueError, match="filter_as_zero=True"):
            BoxFiltering(confidence_threshold=0.5).to_onnx(save=False, input_shape=input_shape)

    def test_explicit_shapes_must_agree(self):
        """Reject a pseudo batch that conflicts with an explicitly fixed input batch."""
        with pytest.raises(ValueError, match="input_shape"):
            BoxFiltering(confidence_threshold=0.5).to_onnx(save=False, input_shape=[3, -1, 6], pseudo_shape=[5, 20, 6])

    @pytest.mark.timeout(120)
    @pytest.mark.parametrize("dynamo", _EXPORTERS)
    def test_filter_as_zero_keeps_a_dynamic_batch_6077(self, dynamo):
        """Match eager tensor outputs at batches both smaller and larger than the traced batch."""
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

    @pytest.mark.timeout(120)
    @pytest.mark.parametrize("dynamo", _EXPORTERS)
    def test_filter_without_threshold_exports_one_input_6112(self, dynamo):
        """A filter without a threshold exports one input and filters at the eager default, 0.0."""
        pytest.importorskip("onnx")
        pytest.importorskip("onnxscript")

        module = BoxFiltering(filter_as_zero=True)
        exported = module.to_onnx(save=False, dynamo=dynamo)
        assert len(exported.graph.input) == 1

        session = _session(exported)
        boxes = torch.zeros(2, 50, 6)
        boxes[:, :, 1] = torch.linspace(0.01, 0.99, 50)
        boxes[:, :, 2:] = 1.0
        (actual,) = session.run(None, {"input": boxes.numpy()})
        assert int((torch.from_numpy(actual)[:, :, 1] > 0).sum()) == int((module(boxes)[:, :, 1] > 0).sum()) == 100

    @pytest.mark.timeout(120)
    @pytest.mark.parametrize("dynamo", _EXPORTERS)
    def test_filter_with_threshold_keeps_it_as_a_constant_6112(self, dynamo):
        """A thresholded filter still filters at its threshold, from one input."""
        pytest.importorskip("onnx")
        pytest.importorskip("onnxscript")

        module = BoxFiltering(confidence_threshold=0.5, filter_as_zero=True)
        exported = module.to_onnx(save=False, dynamo=dynamo)
        assert len(exported.graph.input) == 1

        boxes = torch.zeros(2, 50, 6)
        boxes[:, :, 1] = torch.linspace(0.01, 0.99, 50)
        (actual,) = _session(exported).run(None, {"input": boxes.numpy()})
        self.assert_close(torch.from_numpy(actual), module(boxes))
        assert int((torch.from_numpy(actual)[:, :, 1] > 0).sum()) == 50
