import torch
import torch.nn.functional as F
from torch.testing import assert_allclose

import kornia.augmentation as K
from kornia.grad_estimator import STEFunction, StraightThroughEstimator


class TestSTE:

    def test_smoke(self,):
        StraightThroughEstimator(K.Normalize(0.5, 0.5))

    def test_function(self, device, dtype):
        input = torch.randn(4, requires_grad=True, device=device, dtype=dtype)
        output = torch.sign(input)
        loss = output.mean()
        loss.backward()
        assert_allclose(input.grad, torch.tensor([0., 0., 0., 0.], device=device, dtype=dtype))

        out_est = STEFunction.apply(input, output, F.hardtanh)
        loss = out_est.mean()
        loss.backward()
        assert_allclose(input.grad, torch.tensor([0.2500, 0.2500, 0.2500, 0.2500], device=device, dtype=dtype))

    def test_module(self, device, dtype):
        input = torch.randn(1, 1, 4, 4, requires_grad=True, device=device, dtype=dtype)
        estimator = StraightThroughEstimator(K.RandomPosterize(3, p=1.), grad_fn=F.hardtanh)
        out = estimator(input)
        loss = out.mean()
        loss.backward()
        o = torch.tensor(
            [[[[0.0625, 0.0625, 0.0625, 0.0625],
               [0.0625, 0.0625, 0.0625, 0.0625],
               [0.0625, 0.0625, 0.0625, 0.0625],
               [0.0625, 0.0625, 0.0625, 0.0625]]]], device=device, dtype=dtype)
        assert_allclose(input.grad, o)
