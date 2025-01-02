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

import torch
from torch import nn
from torch.distributions import Normal

from kornia.augmentation.random_generator import DistributionWithMapper

from testing.base import assert_close


class TestDistMapper:
    def test_mapper(self):
        _ = torch.manual_seed(0)
        dist = DistributionWithMapper(Normal(0.0, 1.0), map_fn=nn.Sigmoid())
        out = dist.rsample((8,))
        exp = torch.tensor([0.8236, 0.4272, 0.1017, 0.6384, 0.2527, 0.1980, 0.5995, 0.6980])
        assert_close(out, exp, rtol=1e-4, atol=1e-4)
