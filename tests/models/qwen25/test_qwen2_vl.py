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

from kornia.models.qwen25 import Qwen2VLVisionTransformer


class TestQwen2VL:
    @pytest.mark.parametrize("batch_size", [1, 2])
    def test_smoke(self, batch_size, device, dtype):
        model = Qwen2VLVisionTransformer().to(device=device, dtype=dtype)
        input = torch.randn(batch_size, 3, 224, 224, device=device, dtype=dtype)

        output = model(input)

        assert output.shape[0] == batch_size
        # Get expected output dimension from model (merger output)
        expected_dim = model.merger.mlp[-1].out_features
        assert output.shape[2] == expected_dim
