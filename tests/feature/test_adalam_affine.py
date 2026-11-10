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

from __future__ import annotations

import pytest
import torch

from kornia.feature.adalam import get_adalam_default_config
from kornia.feature.adalam.ransac import ransac

from testing.base import BaseTester


class TestAdalamSharedAffine(BaseTester):
    @pytest.mark.parametrize("refit", [False, True])
    def test_ragged_neighborhoods(self, device, dtype, refit):
        if dtype not in (torch.float32, torch.float64):
            pytest.skip("AdaLAM operates on full-precision coordinates")
        generator = torch.Generator().manual_seed(19)
        sizes = [40, 25]
        source = torch.rand(sum(sizes), 2, generator=generator).to(device=device, dtype=dtype) - 0.5
        source[0] = 0
        source[sizes[0]] = 0
        groups = torch.arange(2, device=device).repeat_interleave(torch.tensor(sizes, device=device))
        expected = torch.tensor([[[1.1, -0.2], [0.1, 0.9]], [[0.8, 0.1], [-0.1, 1.2]]], device=device, dtype=dtype)
        target = (expected[groups] @ source[..., None]).squeeze(-1)
        # Independent nonzero localization noise exercises AdaLAM's original adaptive selection.
        noise = 0.002 * torch.randn(target.shape, generator=generator).to(device=device, dtype=dtype)
        noise[0] = 0
        noise[sizes[0]] = 0
        target += noise
        outliers = torch.tensor([35, 36, 37, 38, 39, 60, 61, 62, 63, 64], device=device)
        target[outliers] += 10
        config = get_adalam_default_config()
        config["device"] = device
        indices, models, confidence, counts = ransac(
            source, target, torch.tensor(sizes, device=device), config, refit=refit
        )
        assert not torch.isin(indices, outliers).any()
        assert (counts >= torch.tensor([25, 15], device=device)).all()
        assert (confidence >= config["min_confidence"]).all()
        self.assert_close(models, expected, atol=0.03, rtol=0.03)

    @pytest.mark.parametrize("refit", [False, True])
    def test_rank_deficient_neighborhood(self, device, dtype, refit):
        if dtype not in (torch.float32, torch.float64):
            pytest.skip("AdaLAM operates on full-precision coordinates")
        source = torch.zeros(20, 2, device=device, dtype=dtype)
        source[:, 0] = torch.linspace(-0.5, 0.5, 20, device=device, dtype=dtype)
        target = source * 1.1
        config = get_adalam_default_config()
        config["device"] = device
        _, _, confidence, counts = ransac(source, target, torch.tensor([20], device=device), config, refit=refit)
        accepted = (confidence >= config["min_confidence"]) & (counts * (1 - 1 / confidence) >= config["min_inliers"])
        assert not accepted.any()
