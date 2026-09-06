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

from kornia.contrib.super_resolution import (
    RRDBNetBuilder,
    SmallSRBuilder,
    SuperResolution,
    SuperResolutionConfig,
)

from testing.base import BaseTester


class TestSuperResolutionBuilders(BaseTester):
    """Pin the public super-resolution entry points.

    Regression test for #4291: ``SuperResolution`` did not implement ``ModelBase``'s
    abstract ``from_config`` and defined no ``__init__``, so every builder raised at
    construction. Nothing in ``tests/`` called the builders, which is why CI never
    noticed.
    """

    def test_small_sr_builder_builds_and_runs(self, device, dtype):
        model = SmallSRBuilder.build(pretrained=False).to(device, dtype)
        out = model(torch.rand(1, 3, 16, 16, device=device, dtype=dtype))
        assert out.shape[0] == 1
        assert out.shape[1] == 3

    def test_rrdbnet_builder_builds_and_runs(self, device, dtype):
        model = RRDBNetBuilder.build(pretrained=False).to(device, dtype)
        out = model(torch.rand(1, 3, 16, 16, device=device, dtype=dtype))
        assert out.shape[0] == 1
        assert out.shape[1] == 3

    @pytest.mark.parametrize("model_name", ["small_sr", "RealESRNet_x4plus"])
    def test_from_config_dispatches_to_both_families(self, device, dtype, model_name):
        config = SuperResolutionConfig(model_name=model_name, pretrained=False)
        model = SuperResolution.from_config(config).to(device, dtype)
        out = model(torch.rand(1, 3, 16, 16, device=device, dtype=dtype))
        assert out.shape[0] == 1
        assert out.shape[1] == 3

    def test_from_config_rejects_unknown_model_name(self):
        with pytest.raises(ValueError):
            SuperResolution.from_config(SuperResolutionConfig(model_name="not_a_model", pretrained=False))
