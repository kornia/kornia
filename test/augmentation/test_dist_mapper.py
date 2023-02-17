import torch
import torch.nn as nn
from torch.distributions import Normal

from kornia.augmentation.random_generator import DistributionWithMapper
from kornia.testing import assert_close


class TestDistMapper:
    def test_mapper(self):
        _ = torch.manual_seed(0)
        dist = DistributionWithMapper(Normal(0.0, 1.0), map_fn=nn.Sigmoid())
        out = dist.rsample((8,))
        exp = torch.tensor([0.8236, 0.4272, 0.1017, 0.6384, 0.2527, 0.1980, 0.5995, 0.6980])
        assert_close(out, exp, rtol=1e-4, atol=1e-4)
