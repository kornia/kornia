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

"""ONNX export tests for ObjectDetector's per-image outputs (#6186)."""

import pytest
import torch

from kornia.contrib.object_detection import ObjectDetector, _returns_detections_per_image
from kornia.core._compat import torch_version_lt
from kornia.models.processors import ResizePreProcessor

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


class _Scored(torch.nn.Module):
    """A detector with one query per image, so every post-processor sees a batch of rows."""

    def forward(self, images):
        score = images.mean((1, 2, 3))
        boxes = images.new_tensor([0.5, 0.5, 0.25, 0.25]).expand(images.shape[0], 1, 4)
        return torch.stack([score, -score], -1)[:, None], boxes


def _detections(logits, boxes):
    scores, labels = logits.max(-1, keepdim=True)
    return torch.cat([labels.to(boxes), scores, boxes], -1)


class _ListPerImage(torch.nn.Module):
    """The #6162 case: one tensor per image, returned as a list."""

    def forward(self, logits, boxes, sizes):
        return list(_detections(logits, boxes).unbind(0))


class _TuplePerImage(torch.nn.Module):
    """The #6186 case: the same per-image tensors, returned as a tuple built by indexing."""

    def forward(self, logits, boxes, sizes):
        d = _detections(logits, boxes)
        return tuple([d[i] for i in range(d.shape[0])])


class _BatchedPair(torch.nn.Module):
    """A tuple whose length does not follow the batch, so it must not be read as per-image."""

    def forward(self, logits, boxes, sizes):
        return logits, boxes


def _export(post_processor, **kwargs):
    detector = ObjectDetector(_Scored(), ResizePreProcessor(32, 32), post_processor).eval()
    return detector.to_onnx(image_size=32, save=False, **kwargs)


def _session(exported):
    """Validate the exported graph and load it with ONNX Runtime's CPU provider."""
    onnx = pytest.importorskip("onnx")
    ort = pytest.importorskip("onnxruntime")
    onnx.checker.check_model(exported)
    return ort.InferenceSession(exported.SerializeToString(), providers=["CPUExecutionProvider"])


class TestObjectDetectorPerImageONNX(BaseTester):
    """Pin the batch contract of an ObjectDetector that returns one detection tensor per image."""

    @pytest.mark.parametrize("post_processor", [_TuplePerImage(), _BatchedPair()], ids=["tuple", "batched_pair"])
    def test_the_sequence_length_decides_6186(self, post_processor):
        """A tuple is per-image only when one entry follows each image of the batch."""
        detector = ObjectDetector(_Scored(), ResizePreProcessor(32, 32), post_processor).eval()
        assert _returns_detections_per_image(detector, torch.rand(1, 3, 32, 32)) is isinstance(
            post_processor, _TuplePerImage
        )

    @pytest.mark.timeout(120)
    @pytest.mark.parametrize("dynamo", _EXPORTERS)
    def test_per_image_tuple_exports_a_fixed_batch_6186(self, dynamo):
        """Export the tuple case like the list case: one output per image of a fixed traced batch."""
        pytest.importorskip("onnx")
        pytest.importorskip("onnxscript")
        ort = pytest.importorskip("onnxruntime")

        exported = _export(_TuplePerImage(), dynamo=dynamo)

        dims = exported.graph.input[0].type.tensor_type.shape.dim
        assert dims[0].dim_value == 1
        assert len(exported.graph.output) == 1

        session = _session(exported)
        images = torch.rand(1, 3, 32, 32, generator=torch.Generator().manual_seed(0))
        (actual,) = session.run(None, {"input": images.numpy()})
        expected = ObjectDetector(_Scored(), ResizePreProcessor(32, 32), _ListPerImage()).eval()(images)[0]
        self.assert_close(torch.from_numpy(actual), expected)

        for other in (2, 3):
            with pytest.raises(ort.capi.onnxruntime_pybind11_state.InvalidArgument, match="Expected: 1"):
                session.run(None, {"input": torch.rand(other, 3, 32, 32).numpy()})

    @pytest.mark.timeout(120)
    def test_batched_tuple_keeps_a_dynamic_batch_6186(self):
        """A batched pair is not per-image, so it keeps exporting for a dynamic batch."""
        pytest.importorskip("onnx")
        pytest.importorskip("onnxruntime")

        exported = _export(_BatchedPair(), dynamo=False)
        dims = exported.graph.input[0].type.tensor_type.shape.dim
        assert dims[0].dim_param
        assert len(exported.graph.output) == 2

        session = _session(exported)
        images = torch.rand(3, 3, 32, 32, generator=torch.Generator().manual_seed(0))
        outputs = session.run(None, {"input": images.numpy()})
        logits, boxes = ObjectDetector(_Scored(), ResizePreProcessor(32, 32), _BatchedPair()).eval()(images)
        self.assert_close(torch.from_numpy(outputs[0]), logits)
        self.assert_close(torch.from_numpy(outputs[1]), boxes)

    def test_dynamic_batch_is_rejected_for_a_per_image_tuple_6186(self):
        """Point a caller asking for a dynamic batch to a post-processor that returns one tensor."""
        with pytest.raises(ValueError, match="confidence_filtering=False"):
            _export(_TuplePerImage(), dynamo=False, dynamic_axes={"input": {0: "batch"}})
