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

from testing.base import BaseTester


class TestVisualPrompter(BaseTester):
    @pytest.mark.slow
    def test_smoke(self, device, dtype):
        if dtype not in (torch.float32, torch.float16):
            pytest.skip("VisualPrompter (SAM) primarily supports float32 and float16")

        prompter = VisualPrompter(device=device, dtype=dtype)
        assert prompter is not None
        assert not prompter.is_image_set

    @pytest.mark.slow
    def test_batching_pipeline(self, device, dtype):
        if dtype not in (torch.float32, torch.float16):
            pytest.skip("VisualPrompter (SAM) primarily supports float32 and float16")
        prompter = VisualPrompter(device=device, dtype=dtype)

        batch_size = 2
        image = torch.rand(batch_size, 3, 256, 256).to(device=device, dtype=dtype)

        prompter.set_image(image)
        assert prompter.is_image_set
        assert prompter.image_embeddings.shape[0] == batch_size

        boxes_tensor = torch.tensor(
            [[[10.0, 10.0, 50.0, 50.0]], [[20.0, 20.0, 80.0, 80.0]]],
            device=device,
            dtype=dtype,
        )

        results = prompter.predict(boxes=boxes_tensor)

        assert results.logits.shape == (batch_size, 3, 256, 256)

    def test_exception(self, device, dtype):

        prompter = VisualPrompter(device=device, dtype=dtype)

        image = torch.rand(1, 2, 3, 256, 256).to(device=device, dtype=dtype)
        with pytest.raises(Exception):
            prompter.set_image(image)

    def test_gradcheck(self, device):
        pytest.skip("Gradcheck is not currently applicable for VisualPrompter inference.")

    def test_cardinality(self, device, dtype):
        pytest.skip("Cardinality is covered by the test_batching_pipeline.")

    def test_dynamo(self, device, dtype, torch_optimizer):
        pytest.skip("Dynamo compilation currently broken for VisualPrompter. See #FIXME in source.")
