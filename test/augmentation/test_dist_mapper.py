import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.testing import assert_allclose

from kornia.augmentation.random_generator import DistributionWithMapper


class TestDistMapper:

    def test_mapper(self,):
        _ = torch.manual_seed(0)
        dist = DistributionWithMapper(Normal(0., 1.,), value_mapper=nn.Sigmoid())
        out = dist.rsample((8,))
        exp = torch.tensor([
            0.8236, 0.4272, 0.1017, 0.6384, 0.2527, 0.1980, 0.5995, 0.6980
        ])
        assert_allclose(out, exp, rtol=1e-4, atol=1e-4)
