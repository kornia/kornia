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
from torch import Tensor

from kornia.models.processors.naflex import NaFlex
from kornia.models.siglip2 import SigLip2VisionConfig, SigLip2VisionEmbeddings

from testing.base import BaseTester


class TestNaFlex(BaseTester):
    @pytest.fixture
    def model(self):
        # 1. Create the base model
        config = SigLip2VisionConfig(image_size=224, patch_size=16)
        base = SigLip2VisionEmbeddings(config)

        # 2. Pass specific parts to NaFlex (New Mentor API)
        return NaFlex(
            patch_embedding_fcn=base.patch_embedding,
            position_embedding=base.position_embedding,
        )

    def test_smoke(self, model, device, dtype):
        input_data = torch.randn(1, 3, 224, 224, device=device, dtype=dtype)
        out = model(input_data)
        assert isinstance(out, Tensor)

    def test_cardinality(self, model, device, dtype):
        # 224x320 image -> 14x20 patches -> 280 total
        input_data = torch.randn(1, 3, 224, 320, device=device, dtype=dtype)
        out = model(input_data)
        assert out.shape[1] == 280

    def test_exception(self, device, dtype):
        # Test with a fake function and tensor
        def fake_patch_fcn(x):
            return torch.randn(1, 100, 768, device=device, dtype=dtype)

        # Create a bad position embedding (200 is not a perfect square)
        bad_pos_embed = torch.randn(200, 768, device=device, dtype=dtype)
        wrapper_bad = NaFlex(fake_patch_fcn, bad_pos_embed)
        input_data = torch.randn(1, 3, 224, 224, device=device, dtype=dtype)

        with pytest.raises(ValueError, match="Original positional embedding is not a square grid"):
            wrapper_bad(input_data)
