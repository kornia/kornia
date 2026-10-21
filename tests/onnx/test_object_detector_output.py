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

import numpy as np
import pytest
import torch
from torch import nn

from kornia.contrib.object_detection import ObjectDetector, _returns_detections_per_image
from kornia.core import ImageModule
from kornia.core._compat import torch_version_lt
from kornia.models.processors import ResizePreProcessor

from testing.base import DYNAMO_UNAVAILABLE_REASON, BaseTester, dynamo_is_available


class _Scored(nn.Module):
    def forward(self, images):
        signal = images.mean((-2, -1))
        return signal[:, None, :2], torch.cat((signal, signal[:, :1]), dim=-1)[:, None]


class _Post(nn.Module):
    def __init__(self, output_kind):
        super().__init__()
        self.output_kind = output_kind

    def forward(self, logits, boxes, sizes):
        if self.output_kind == "batched_pair":
            return logits, boxes
        detections = torch.cat((logits, boxes), dim=-1)
        if self.output_kind == "one_tuple":
            return (detections,)
        if self.output_kind == "list":
            return [detections[i] for i in range(detections.shape[0])]
        return tuple(detections[i] for i in range(detections.shape[0]))


def _detector(output_kind):
    return ObjectDetector(_Scored(), ResizePreProcessor(16, 16), _Post(output_kind)).eval()


def _session(exported):
    ort = pytest.importorskip("onnxruntime")
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    return ort.InferenceSession(exported.SerializeToString(), options, providers=["CPUExecutionProvider"])


class TestObjectDetectorOutputSequence(BaseTester):
    @pytest.mark.parametrize("batch", [1, 2])
    # A 1-tuple of batched detections has the length of a batch of one: only the other batch size tells it apart.
    @pytest.mark.parametrize(
        "output_kind, expected", [("list", True), ("tuple", True), ("batched_pair", False), ("one_tuple", False)]
    )
    def test_probe(self, batch, output_kind, expected):
        detector = _detector(output_kind)
        assert _returns_detections_per_image(detector, torch.rand(batch, 3, 16, 16)) is expected

    @pytest.mark.parametrize("dynamo", [False, True])
    @pytest.mark.parametrize("output_kind", ["list", "tuple"])
    def test_fixed_batch_export(self, dynamo, output_kind):
        pytest.importorskip("onnx")
        if dynamo:
            pytest.importorskip("onnxscript")
        detector = _detector(output_kind)
        exported = detector.to_onnx(image_size=16, save=False, dynamo=dynamo)
        session = _session(exported)
        images = torch.rand(1, 3, 16, 16)

        assert exported.graph.input[0].type.tensor_type.shape.dim[0].dim_value == 1
        outputs = session.run(None, {"input": images.numpy()})
        assert len(outputs) == 1
        np.testing.assert_allclose(outputs[0], detector(images)[0].numpy(), rtol=1e-5, atol=1e-6)

        ort = pytest.importorskip("onnxruntime")
        with pytest.raises(ort.capi.onnxruntime_pybind11_state.InvalidArgument):
            session.run(None, {"input": torch.rand(3, 3, 16, 16).numpy()})

    @pytest.mark.parametrize(
        "export_kwargs",
        [
            pytest.param({"dynamo": False, "dynamic_axes": {"input": {0: "batch"}}}, id="torchscript"),
            pytest.param(
                {"dynamo": True, "dynamic_shapes": ({0: "batch"},)},
                # conftest.py deselects IDs containing "dynamo" unless KORNIA_TEST_OPTIMIZER is set.
                id="torch_export",
                marks=[
                    pytest.mark.skipif(not dynamo_is_available(), reason=DYNAMO_UNAVAILABLE_REASON),
                    pytest.mark.skipif(torch_version_lt(2, 9, 0), reason="to_onnx uses dynamic_shapes from torch 2.9"),
                ],
            ),
        ],
    )
    @pytest.mark.parametrize("output_kind", ["list", "tuple"])
    def test_dynamic_per_image_batch_rejected(self, output_kind, export_kwargs, tmp_path):
        pytest.importorskip("onnx")
        if export_kwargs["dynamo"]:
            pytest.importorskip("onnxscript")
        output_path = tmp_path / "detector.onnx"
        with pytest.raises(ValueError, match="post-processor that returns one tensor"):
            _detector(output_kind).to_onnx(onnx_name=str(output_path), image_size=16, **export_kwargs)
        assert not output_path.exists()

    @pytest.mark.parametrize("dynamo", [False, True])
    def test_batched_pair_stays_dynamic(self, dynamo):
        pytest.importorskip("onnx")
        if dynamo:
            pytest.importorskip("onnxscript")
        detector = _detector("batched_pair")
        # Bound the dynamo dimension: torch 2.5 otherwise falls back to a static TorchScript graph.
        batch_options = (
            {"dynamic_shapes": ({0: torch.export.Dim("batch", min=1, max=8)},)}
            if dynamo
            else {"dynamic_axes": {"input": {0: "batch"}}}
        )
        exported = detector.to_onnx(
            image_size=16,
            save=False,
            dynamo=dynamo,
            output_names=["logits", "boxes"],
            **batch_options,
        )
        session = _session(exported)
        assert exported.graph.input[0].type.tensor_type.shape.dim[0].dim_param
        for batch in (1, 3, 5):
            images = torch.rand(batch, 3, 16, 16)
            outputs = session.run(None, {"input": images.numpy()})
            assert len(outputs) == 2
            for actual, expected in zip(outputs, detector(images)):
                np.testing.assert_allclose(actual, expected.numpy(), rtol=1e-5, atol=1e-6)

    def test_tuple_probe_preserves_rng_and_training_mode(self):
        class RandomScored(_Scored):
            def forward(self, images):
                logits, boxes = super().forward(images)
                return logits + torch.rand_like(logits), boxes

        detector = ObjectDetector(RandomScored(), ResizePreProcessor(16, 16), _Post("tuple")).train()
        images = torch.rand(1, 3, 16, 16)
        state = torch.get_rng_state().clone()
        assert _returns_detections_per_image(detector, images)
        assert torch.equal(state, torch.get_rng_state())
        assert all(module.training for module in detector.modules())

    def test_tuple_probe_preserves_image_module_cache(self):
        class CachedPre(ImageModule):
            def forward(self, images):
                return images, images.new_ones((images.shape[0], 2))

        pre = CachedPre()
        detector = ObjectDetector(_Scored(), pre, _Post("tuple"))
        pre(torch.rand(1, 3, 8, 8))
        cache = pre.__dict__["_output_image"]

        assert _returns_detections_per_image(detector, torch.rand(1, 3, 16, 16))
        assert pre.__dict__["_output_image"] is cache

    def test_tuple_probe_accepts_contiguous_only_preprocessor(self):
        class FlattenPre(nn.Module):
            def forward(self, images):
                images = images.view(-1).mul(0.5).view_as(images)
                return images, images.new_ones((images.shape[0], 2))

        detector = ObjectDetector(_Scored(), FlattenPre(), _Post("tuple"))
        images = torch.rand(1, 3, 16, 16)
        assert len(detector(images)) == 1
        assert _returns_detections_per_image(detector, images)
