import pytest
import torch

import kornia

from testing.base import BaseTester


class TestTverskyLoss(BaseTester):
    def test_smoke(self, device, dtype):
        num_classes = 3
        logits = torch.rand(2, num_classes, 3, 2, device=device, dtype=dtype)
        labels = torch.rand(2, 3, 2) * num_classes
        labels = labels.to(device).long()

        criterion = kornia.losses.TverskyLoss(alpha=0.5, beta=0.5)
        assert criterion(logits, labels) is not None

    def test_exception(self):
        criterion = kornia.losses.TverskyLoss(alpha=0.5, beta=0.5)

        with pytest.raises(TypeError) as errinfo:
            criterion("not a tensor", torch.rand(1))
        assert "pred type is not a torch.Tensor. Got" in str(errinfo)

        with pytest.raises(ValueError) as errinfo:
            criterion(torch.rand(1), torch.rand(1))
        assert "Invalid pred shape, we expect BxNxHxW. Got:" in str(errinfo)

        with pytest.raises(ValueError) as errinfo:
            criterion(torch.rand(1, 1, 1, 1), torch.rand(1, 1, 1, 2))
        assert "pred and target shapes must be the same. Got:" in str(errinfo)

        with pytest.raises(ValueError) as errinfo:
            criterion(torch.rand(1, 1, 1, 1), torch.rand(1, 1, 1, 1, device="meta"))
        assert "pred and target must be in the same device. Got:" in str(errinfo)

    def test_all_zeros(self, device, dtype):
        num_classes = 3
        logits = torch.zeros(2, num_classes, 1, 2, device=device, dtype=dtype)
        logits[:, 0] = 10.0
        logits[:, 1] = 1.0
        logits[:, 2] = 1.0
        labels = torch.zeros(2, 1, 2, device=device, dtype=torch.int64)

        criterion = kornia.losses.TverskyLoss(alpha=0.5, beta=0.5)
        loss = criterion(logits, labels)
        self.assert_close(loss, torch.zeros_like(loss), atol=1e-3, rtol=1e-3)

    def test_gradcheck(self, device):
        num_classes = 3
        alpha, beta = 0.5, 0.5  # for tversky loss
        logits = torch.rand(2, num_classes, 3, 2, device=device, dtype=torch.float64)
        labels = torch.randint(0, num_classes, (2, 3, 2), device=device)

        self.gradcheck(
            kornia.losses.tversky_loss, (logits, labels, alpha, beta), dtypes=[torch.float64, torch.int64, None, None]
        )

    def test_dynamo(self, device, dtype, torch_optimizer):
        num_classes = 3
        params = (0.5, 0.05)
        logits = torch.rand(2, num_classes, 3, 2, device=device, dtype=dtype)
        labels = torch.rand(2, 3, 2) * num_classes
        labels = labels.to(device).long()

        op = kornia.losses.tversky_loss
        op_optimized = torch_optimizer(op)

        actual = op_optimized(logits, labels, *params)
        expected = op(logits, labels, *params)
        self.assert_close(actual, expected)

    def test_module(self, device, dtype):
        num_classes = 3
        params = (0.5, 0.5)
        logits = torch.rand(2, num_classes, 3, 2, device=device, dtype=dtype)
        labels = torch.rand(2, 3, 2) * num_classes
        labels = labels.to(device).long()

        op = kornia.losses.tversky_loss
        op_module = kornia.losses.TverskyLoss(*params)

        actual = op_module(logits, labels)
        expected = op(logits, labels, *params)
        self.assert_close(actual, expected)
