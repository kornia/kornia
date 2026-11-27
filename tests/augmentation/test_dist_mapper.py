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
import pickle

import pytest
import torch
from torch import nn
from torch.distributions import Normal

from kornia.augmentation.random_generator import DistributionWithMapper

from testing.base import BaseTester, assert_close


class TestDistMapper(BaseTester):
    def test_mapper(self):
        _ = torch.manual_seed(0)
        dist = DistributionWithMapper(Normal(0.0, 1.0), map_fn=nn.Sigmoid())
        out = dist.rsample((8,))
        exp = torch.tensor([0.8236, 0.4272, 0.1017, 0.6384, 0.2527, 0.1980, 0.5995, 0.6980])
        assert_close(out, exp, rtol=1e-4, atol=1e-4)

    @pytest.mark.parametrize("batch_shape", [(), (2,)])
    def test_distribution_attributes(self, batch_shape, device, dtype):
        source = Normal(
            torch.zeros(batch_shape, device=device, dtype=dtype), torch.ones((), device=device, dtype=dtype)
        )
        dist = DistributionWithMapper(source, map_fn=nn.Sigmoid())
        assert dist.loc is source.loc
        assert dist.scale is source.scale
        assert dist.batch_shape == source.batch_shape
        assert dist.event_shape == source.event_shape

    @pytest.mark.parametrize("initialized", [True, False])
    def test_missing_attribute(self, initialized):
        dist = DistributionWithMapper(Normal(0.0, 1.0)) if initialized else object.__new__(DistributionWithMapper)
        with pytest.raises(AttributeError):
            _ = dist.missing_attribute
        assert not hasattr(dist, "missing_attribute")

    @pytest.mark.parametrize("copy_fn", [copy.copy, copy.deepcopy])
    def test_copy(self, copy_fn, device, dtype):
        source = Normal(torch.zeros(2, device=device, dtype=dtype), torch.ones(2, device=device, dtype=dtype))
        dist = DistributionWithMapper(source, map_fn=nn.Sigmoid())
        copied = copy_fn(dist)
        assert isinstance(copied, DistributionWithMapper)
        assert isinstance(copied.map_fn, nn.Sigmoid)
        torch.manual_seed(0)
        expected = dist.sample((4,))
        torch.manual_seed(0)
        self.assert_close(copied.sample((4,)), expected, rtol=0, atol=0)

    def test_pickle(self, device, dtype):
        source = Normal(torch.zeros(2, device=device, dtype=dtype), torch.ones(2, device=device, dtype=dtype))
        dist = DistributionWithMapper(source, map_fn=nn.Sigmoid())
        restored = pickle.loads(pickle.dumps(dist))  # noqa: S301
        assert isinstance(restored, DistributionWithMapper)
        assert isinstance(restored.map_fn, nn.Sigmoid)
        self.assert_close(restored.dist.loc, source.loc, rtol=0, atol=0)
        self.assert_close(restored.dist.scale, source.scale, rtol=0, atol=0)
        assert restored.batch_shape == source.batch_shape

    def test_deepcopy_keeps_the_wrapper_when_the_distribution_defines_deepcopy(self):
        class CopyingNormal(Normal):
            def __deepcopy__(self, memo):
                return CopyingNormal(self.loc.clone(), self.scale.clone())

        copied = copy.deepcopy(DistributionWithMapper(CopyingNormal(0.0, 1.0), map_fn=nn.Sigmoid()))
        assert isinstance(copied, DistributionWithMapper)
        assert isinstance(copied.map_fn, nn.Sigmoid)
        assert isinstance(copied.dist, CopyingNormal)
