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

from kornia.contrib.visual_prompter import VisualPrompter
from kornia.models.sam import SamConfig
from kornia.utils._compat import torch_version

from testing.base import BaseTester


class TestVisualPrompter(BaseTester):
    @pytest.mark.slow
    def test_smoke(self, device, dtype):
        data = torch.rand(3, 77, 128, device=device, dtype=dtype)
        prompter = VisualPrompter(SamConfig("vit_b"), device, dtype)

        prompter.set_image(data)
        assert prompter.is_image_set

        prompter.reset_image()
        assert not prompter.is_image_set

    @pytest.mark.slow
    @pytest.mark.parametrize("batch_size", [1, 4])
    @pytest.mark.parametrize("N", [2, 5])
    @pytest.mark.parametrize("multimask_output", [True, False])
    def test_cardinality(self, device, batch_size, N, multimask_output):
        # SAM: don't supports float64
        dtype = torch.float32
        data = torch.rand(3, 77, 128, device=device, dtype=dtype)
        prompter = VisualPrompter(SamConfig("vit_b"), device, dtype)

        keypoints = torch.randint(0, min(data.shape[-2:]), (batch_size, N, 2), device=device).to(dtype=dtype)
        labels = torch.randint(0, 1, (batch_size, N), device=device).to(dtype=dtype)

        prompter.set_image(data)

        out = prompter.predict(keypoints, labels, multimask_output=multimask_output)

        C = 3 if multimask_output else 1
        assert out.logits.shape == (batch_size, C, 256, 256)
        assert out.scores.shape == (batch_size, C)

    def test_exception(self):
        prompter = VisualPrompter(SamConfig("vit_b"))

        data = torch.rand(1, 3, 1, 2)

        # Wrong shape for the image
        with pytest.raises(TypeError) as errinfo:
            prompter.set_image(data, [], False)
        assert "shape must be [['3', 'H', 'W']]. Got torch.Size([1, 3, 1, 2])" in str(errinfo)

        # predict without set an image
        with pytest.raises(Exception) as errinfo:
            prompter.predict()
        assert "An image must be set with `self.set_image(...)`" in str(errinfo)

        # Valid masks
        with pytest.raises(TypeError) as errinfo:
            prompter._valid_masks(data)
        assert "shape must be [['K', '1', '256', '256']]. Got torch.Size([1, 3, 1, 2])" in str(errinfo)

        # Valid boxes
        with pytest.raises(TypeError) as errinfo:
            prompter._valid_boxes(data)
        assert "shape must be [['K', '4']]. Got torch.Size([1, 3, 1, 2])" in str(errinfo)

        # Valid keypoints
        with pytest.raises(TypeError) as errinfo:
            prompter._valid_keypoints(data, None)
        assert "shape must be [['K', 'N', '2']]. Got torch.Size([1, 3, 1, 2])" in str(errinfo)

        with pytest.raises(TypeError) as errinfo:
            prompter._valid_keypoints(torch.rand(1, 1, 2), data)
        assert "shape must be [['K', 'N']]. Got torch.Size([1, 3, 1, 2])" in str(errinfo)

        with pytest.raises(Exception) as errinfo:
            prompter._valid_keypoints(torch.rand(1, 1, 2), torch.rand(2, 1))
        assert "The keypoints and labels should have the same batch size" in str(errinfo)

    @pytest.mark.skip(reason="Unnecessary test")
    def test_gradcheck(self, device): ...

    @pytest.mark.skip(reason="Unnecessary test")
    def test_module(self): ...

    @pytest.mark.skipif(torch_version() in {"2.1.2", "2.0.1"}, reason="Not working well")
    def test_dynamo(self, device, torch_optimizer):
        dtype = torch.float32
        batch_size = 1
        N = 2
        data = torch.rand(3, 77, 128, device=device, dtype=dtype)
        keypoints = torch.randint(0, min(data.shape[-2:]), (batch_size, N, 2), device=device, dtype=dtype)
        labels = torch.randint(0, 1, (batch_size, N), device=device, dtype=dtype)

        prompter = VisualPrompter(SamConfig("vit_b"), device, dtype)
        prompter.set_image(data)

        expected = prompter.predict(keypoints=keypoints, keypoints_labels=labels)
        prompter.reset_image()

        prompter.compile()
        prompter.set_image(data)
        actual = prompter.predict(keypoints=keypoints, keypoints_labels=labels)

        # TODO (joao): explore the reason for the discrepancy between cuda/cpu
        rtol = None
        atol = None
        if "cuda" in device.type:
            rtol = 1e-3
            atol = 1e-3

        self.assert_close(expected.logits, actual.logits, rtol=rtol, atol=atol)
        self.assert_close(expected.scores, actual.scores, rtol=rtol, atol=atol)
