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

        x = torch.randn(batch_size, 3, 112, 112, device=device, dtype=dtype)

        output = model(x)

        assert output.shape == (batch_size, 64, 64)

    def test_gradients(self, device):
        model = Qwen2VLVisionTransformer(embed_dim=64, depth=1, num_heads=4).to(device)

        x = torch.randn(1, 3, 112, 112, device=device, requires_grad=True)
        output = model(x)

        loss = output.mean()
        loss.backward()

        assert x.grad is not None

    def test_patch_merger(self, device):
        merger = Qwen2VLPatchMerger(dim=64).to(device)

        x = torch.randn(1, 3, 112, 112, device=device)
        output = merger(x)

        assert output.shape == (1, 64, 64)

    def test_batch_consistency(self, device):
        """Ensure outputs are consistent between batch and single input."""
        model = Qwen2VLVisionTransformer(embed_dim=64, depth=2, num_heads=4).to(device)

        x = torch.randn(2, 3, 112, 112, device=device)

        out_batch = model(x)
        out_single = model(x[:1])

        assert torch.allclose(out_batch[0], out_single[0], atol=1e-5)

    def test_wart_qwen2vl_scaffold_has_no_positional_information_6003(self, device):
        # The scaffold has no positional information (#6003 tracks the Qwen2.5-VL vision tower): swapping two 14-px
        # patches of the image only swaps the two output tokens. The reference tower's 2-D rotary embedding would
        # change every token.
        model = Qwen2VLVisionTransformer(embed_dim=64, depth=2, num_heads=4).to(device).eval()
        images = torch.randn(1, 3, 56, 56, device=device)  # a 4 x 4 grid of 14-px patches
        swapped = images.clone()
        # patch (0, 0), token 0, and patch (2, 3), token 2 * 4 + 3 = 11
        swapped[..., 0:14, 0:14] = images[..., 28:42, 42:56]
        swapped[..., 28:42, 42:56] = images[..., 0:14, 0:14]
        permutation = list(range(16))
        permutation[0], permutation[11] = 11, 0

        with torch.no_grad():
            output = model(images)
            output_swapped = model(swapped)

        torch.testing.assert_close(output_swapped, output[:, permutation], rtol=0.0, atol=1e-5)
