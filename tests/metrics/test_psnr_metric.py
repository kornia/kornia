import torch

import kornia

from testing.base import BaseTester


class TestPsnr(BaseTester):
    def test_metric(self, device, dtype):
        sample = torch.ones(1, device=device, dtype=dtype)
        expected = torch.tensor(20.0, device=device, dtype=dtype)
        actual = kornia.metrics.psnr(sample, 1.2 * sample, 2.0)
        self.assert_close(actual, expected)
