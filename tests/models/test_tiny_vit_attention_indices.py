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

import copy

import pytest
import torch

from kornia.models.tiny_vit import Attention

from testing.base import BaseTester

# Checkpoint bias entries are ordered by absolute row displacement, then column displacement.
_RECTANGULAR_INDICES = [
    [0, 1, 2, 3, 4, 5],
    [1, 0, 1, 4, 3, 4],
    [2, 1, 0, 5, 4, 3],
    [3, 4, 5, 0, 1, 2],
    [4, 3, 4, 1, 0, 1],
    [5, 4, 3, 2, 1, 0],
]


class TestTinyViTAttentionIndices(BaseTester):
    def test_rectangular_checkpoint_indices(self, device):
        with torch.device(device):
            indices, count = Attention.build_attention_bias((2, 3))
        expected = torch.tensor(_RECTANGULAR_INDICES, device=device, dtype=torch.int64)
        assert count == 6
        assert indices.dtype == torch.int64
        self.assert_close(indices, expected)

    @pytest.mark.parametrize("resolution", [(1, 3), (3, 1)])
    def test_single_axis_checkpoint_indices(self, device, resolution):
        with torch.device(device):
            indices, count = Attention.build_attention_bias(resolution)
        expected = torch.tensor([[0, 1, 2], [1, 0, 1], [2, 1, 0]], device=device, dtype=torch.int64)
        assert count == 3
        self.assert_close(indices, expected)

    @pytest.mark.applies_to(
        when=lambda p: not (p["device_name"] == "mps" and p["dtype_name"] == "float64"),
        reason="MPS does not support float64",
    )
    def test_checkpoint_bias_values_reach_the_same_token_pairs(self, device, dtype):
        model = Attention(4, 2, num_heads=2, attn_ratio=1.0, resolution=(2, 3)).to(device=device, dtype=dtype)
        with torch.no_grad():
            # Distinct nonzero entries expose a permutation that zero-initialized attention biases would hide.
            model.attention_biases.copy_(torch.arange(12, device=device, dtype=dtype).reshape(2, 6) / 10)
        reference = copy.deepcopy(model)
        reference.attention_bias_idxs = torch.tensor(_RECTANGULAR_INDICES, device=device, dtype=torch.int64)
        data = torch.rand(2, 6, 4, device=device, dtype=dtype)
        self.assert_close(model(data), reference(data))
