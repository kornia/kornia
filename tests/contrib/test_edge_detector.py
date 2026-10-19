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

import kornia
from kornia.contrib.edge_detection import EdgeDetectorBuilder
from kornia.models.dexined import DexiNed

from testing.base import BaseTester


class TestEdgeDetector(BaseTester):
    def test_builder_feeds_dexined_bgr_minus_bgr_mean(self, device, dtype):
        # DexiNed was trained on ``cv2.imread`` images (BGR, 0-255) minus ``mean_bgr``: xavysp/DexiNed@d944b70
        # datasets.py:255, 300, with ``--mean_pixel_values`` [103.939, 116.779, 123.68] in B, G, R order
        # (main.py:319-320).
        # A 64 px network input keeps the random DexiNed fast in half precision on the CPU.
        det = EdgeDetectorBuilder.build(pretrained=False, image_size=64).to(device, dtype)
        # DexiNed stays at ``model[1]``, so the state-dict keys of a saved ``EdgeDetector`` stay ``model.1.*``.
        assert isinstance(det.model[1], DexiNed)
        dexined = next(m for m in det.model.modules() if isinstance(m, DexiNed))

        captured = {}

        def capture(name):
            def hook(module, args):
                captured[name] = args[0].clone()

            return hook

        handles = [
            det.model.register_forward_pre_hook(capture("seq_in")),
            dexined.register_forward_pre_hook(capture("dexined_in")),
        ]
        try:
            det(torch.rand(1, 3, 64, 96, device=device, dtype=dtype))
        finally:
            for handle in handles:
                handle.remove()

        # Compared in float64 on the CPU (MPS has no float64).
        seq_in = captured["seq_in"].cpu().double()  # the resized RGB image in [0, 1]
        mean_bgr = torch.tensor([103.939, 116.779, 123.68], dtype=torch.float64).view(1, 3, 1, 1)
        expected = 255 * seq_in.flip(1) - mean_bgr
        # Normalize computes ``(x - m / 255) / (1 / 255)`` in ``dtype`` from float32 statistics: about four roundings,
        # each up to half an ulp of the coarser of ``dtype`` and float32 on the 0-255 scale, so 4 * 255 * eps bounds
        # the error (bfloat16: 8.0, below the 19.7 between the R and B means that a mean in the wrong order leaves).
        atol = 4 * 255 * max(torch.finfo(dtype).eps, torch.finfo(torch.float32).eps)
        self.assert_close(captured["dexined_in"].cpu().double(), expected, rtol=0.0, atol=atol)

    @pytest.mark.device_agnostic
    def test_edge_detector_onnx_matches_eager(self):
        pytest.importorskip("onnx")
        ort = pytest.importorskip("onnxruntime")
        pytest.importorskip("onnxscript")

        det = EdgeDetectorBuilder.build(pretrained=False)
        # ``image_size=None`` exports ``det.model`` (the BGR step, DexiNed and the sigmoid) with a dynamic batch,
        # height and width.
        op = det.to_onnx(image_size=None, include_pre_and_post_processor=False, save=False)
        session = ort.InferenceSession(op.SerializeToString(), providers=["CPUExecutionProvider"])

        img = torch.rand(2, 3, 64, 96)
        exported = torch.from_numpy(session.run(None, {session.get_inputs()[0].name: img.numpy()})[0])
        with torch.no_grad():
            eager = det.model(img)
        self.assert_close(exported, eager, rtol=0.0, atol=1e-4)

    @pytest.mark.slow
    def test_smoke(self, device, dtype):
        img = torch.rand(2, 3, 64, 64, device=device, dtype=dtype)
        net = kornia.contrib.EdgeDetectorBuilder.build(pretrained=False).to(device, dtype)
        out = net(img)
        # ResizePostProcessor returns a list, so we need to handle that
        if isinstance(out, list):
            assert len(out) == 2
            assert all(item.shape == (1, 1, 64, 64) for item in out)
        else:
            assert out.shape == (2, 1, 64, 64)

    @pytest.mark.slow
    @pytest.mark.skip(reason="issue with `ClassVar[list[int]]`")
    def test_jit(self, device, dtype):
        op = kornia.contrib.EdgeDetectorBuilder.build(pretrained=False).to(device, dtype)
        op_jit = torch.jit.script(op)
        assert op_jit is not None
