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

from kornia.models.qwen25.qwen2_vl import (
    Qwen2VLPatchMerger,
    Qwen2VLVisionTransformer,
)


class TestQwen2VL:
    @pytest.mark.parametrize("batch_size", [1, 2])
    def test_smoke(self, batch_size, device, dtype):
        #  smaller model → stable
        model = Qwen2VLVisionTransformer(embed_dim=64, depth=2, num_heads=4).to(device=device, dtype=dtype)

        x = torch.randn(batch_size, 3, 224, 224, device=device, dtype=dtype)

        output = model(x)

        assert output.shape == (batch_size, 256, 64)

    def test_gradients(self, device):
        model = Qwen2VLVisionTransformer(embed_dim=64, depth=1, num_heads=4).to(device)

        x = torch.randn(1, 3, 224, 224, device=device, requires_grad=True)
        output = model(x)

        loss = output.mean()
        loss.backward()

        assert x.grad is not None

    def test_patch_merger(self, device):
        merger = Qwen2VLPatchMerger(dim=64).to(device)

        x = torch.randn(1, 3, 224, 224, device=device)
        output = merger(x)

        assert output.shape == (1, 256, 64)

    def test_batch_consistency(self, device):
        """Ensure outputs are consistent between batch and single input."""
        model = Qwen2VLVisionTransformer(embed_dim=64, depth=2, num_heads=4).to(device)

        x = torch.randn(2, 3, 224, 224, device=device)

        out_batch = model(x)
        out_single = model(x[:1])

        assert torch.allclose(out_batch[0], out_single[0], atol=1e-5)
