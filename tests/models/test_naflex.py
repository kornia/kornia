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
# ... (Copy license from above) ...

import pytest
import torch
from torch import Tensor

from kornia.models.siglip2 import SigLip2VisionConfig, SigLip2VisionEmbeddings
from kornia.models.siglip2.naflex import NaFlex

from testing.base import BaseTester


class TestNaFlex(BaseTester):
    @pytest.fixture
    def model(self):
        config = SigLip2VisionConfig(image_size=224, patch_size=16)
        base = SigLip2VisionEmbeddings(config)
        return NaFlex(base, embedding_attr="position_embedding")

    def test_smoke(self, model, device, dtype):
        # Does it run without crashing?
        input_data = torch.randn(1, 3, 224, 224, device=device, dtype=dtype)
        out = model(input_data)
        assert isinstance(out, Tensor)

    def test_cardinality(self, model, device, dtype):
        # Does NaFlex actually work on wide images?
        # 224 height // 16 = 14 patches
        # 320 width // 16 = 20 patches
        # Total = 14 * 20 = 280 patches
        input_data = torch.randn(1, 3, 224, 320, device=device, dtype=dtype)
        out = model(input_data)

        # Expected shape: (Batch, Patches, Hidden)
        assert out.shape[1] == 280

    def test_exception(self, device, dtype):
        # Does it fail gracefully if we wrap a bad object?
        class BadModel:
            pass

        wrapper = NaFlex(BadModel(), embedding_attr="fake_attr")
        input_data = torch.randn(1, 3, 224, 224, device=device, dtype=dtype)

        with pytest.raises(AttributeError):
            wrapper(input_data)
