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

from kornia.color import sepia_from_rgb


@pytest.mark.parametrize("B", [1, 8, 32])
@pytest.mark.parametrize("C", [3])
@pytest.mark.parametrize("H", [128, 256, 512])
@pytest.mark.parametrize("W", [128, 256, 512])
def test_sepia_from_rgb(benchmark, device, dtype, torch_optimizer, B, C, H, W):
    data = torch.rand(B, C, H, W, device=device, dtype=dtype)

    op = torch_optimizer(sepia_from_rgb)

    actual = benchmark(op, data)

    assert actual.shape == (B, 3, H, W)
