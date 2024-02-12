import pytest
import torch

import kornia

from testing.base import BaseTester


class TestDiceLoss(BaseTester):
    def test_smoke(self, device, dtype):
        num_classes = 3
        logits = torch.rand(2, num_classes, 3, 2, device=device, dtype=dtype)
        labels = torch.rand(2, 3, 2) * num_classes
        labels = labels.to(device).long()

        criterion = kornia.losses.DiceLoss()
        assert criterion(logits, labels) is not None

    def test_all_zeros(self, device, dtype):
        num_classes = 3
        logits = torch.zeros(2, num_classes, 1, 2, device=device, dtype=dtype)
        logits[:, 0] = 10.0
        logits[:, 1] = 1.0
        logits[:, 2] = 1.0
        labels = torch.zeros(2, 1, 2, device=device, dtype=torch.int64)

        criterion = kornia.losses.DiceLoss()
        loss = criterion(logits, labels)
        self.assert_close(loss, torch.zeros_like(loss), rtol=1e-3, atol=1e-3)

    def test_exception(self):
        with pytest.raises(ValueError) as errinf:
            kornia.losses.DiceLoss()(torch.rand(1, 1, 1), torch.rand(1, 1, 1))
        assert "Invalid pred shape, we expect BxNxHxW. Got:" in str(errinf)

        with pytest.raises(ValueError) as errinf:
            kornia.losses.DiceLoss()(torch.rand(1, 1, 1, 1), torch.rand(1, 1, 1, 2))
        assert "pred and target shapes must be the same. Got: " in str(errinf)

        with pytest.raises(ValueError) as errinf:
            kornia.losses.DiceLoss()(torch.rand(1, 1, 1, 1), torch.rand(1, 1, 1, 1, device="meta"))
        assert "pred and target must be in the same device. Got:" in str(errinf)

    def test_averaging_micro(self, device, dtype):
        num_classes = 2
        eps = 1e-8

        logits = torch.zeros(1, num_classes, 4, 1, device=device, dtype=dtype)
        logits[:, 0, 0:3] = 10.0
        logits[:, 0, 3:4] = 1.0
        logits[:, 1, 0:3] = 1.0
        logits[:, 1, 3:4] = 10.0

        labels = torch.zeros(2, 4, 1, device=device, dtype=torch.int64)

        exp_1_0 = torch.exp(torch.tensor([1.0], device=device, dtype=dtype))
        exp_10_0 = torch.exp(torch.tensor([10.0], device=device, dtype=dtype))

        expected_intersection = (3.0 * exp_10_0 + 1.0 * exp_1_0) / (exp_1_0 + exp_10_0)
        expected_cardinality = 8.0  # for micro averaging cardinality is equal 2 * H * W
        expected_loss = 1.0 - 2.0 * expected_intersection / (expected_cardinality + eps)
        expected_loss = expected_loss.squeeze()

        criterion = kornia.losses.DiceLoss(average="micro", eps=eps)
        loss = criterion(logits, labels)
        self.assert_close(loss, expected_loss, rtol=1e-3, atol=1e-3)

    def test_averaging_macro(self, device, dtype):
        num_classes = 2
        eps = 1e-8

        logits = torch.zeros(1, num_classes, 4, 1, device=device, dtype=dtype)
        logits[:, 0, 0:3] = 10.0
        logits[:, 0, 3:4] = 1.0
        logits[:, 1, 0:3] = 1.0
        logits[:, 1, 3:4] = 10.0

        labels = torch.zeros(2, 4, 1, device=device, dtype=torch.int64)

        exp_1_0 = torch.exp(torch.tensor([1.0], device=device, dtype=dtype))
        exp_10_0 = torch.exp(torch.tensor([10.0], device=device, dtype=dtype))

        expected_intersection_1 = (3.0 * exp_10_0 + exp_1_0) / (exp_1_0 + exp_10_0)
        expected_intersection_2 = 0.0  # all labels are 0 so the intersection for the second class is empty
        expected_cardinality_1 = 4.0 + (3.0 * exp_10_0 + 1.0 * exp_1_0) / (exp_1_0 + exp_10_0)
        expected_cardinality_2 = 0.0 + (1.0 * exp_10_0 + 3.0 * exp_1_0) / (exp_1_0 + exp_10_0)

        expected_loss_1 = 1.0 - 2.0 * expected_intersection_1 / (expected_cardinality_1 + eps)
        expected_loss_2 = 1.0 - 2.0 * expected_intersection_2 / (expected_cardinality_2 + eps)
        expected_loss = (expected_loss_1 + expected_loss_2) / 2.0
        expected_loss = expected_loss.squeeze()

        criterion = kornia.losses.DiceLoss(average="macro", eps=eps)
        loss = criterion(logits, labels)
        self.assert_close(loss, expected_loss, rtol=1e-3, atol=1e-3)

    def test_gradcheck(self, device):
        num_classes = 3
        logits = torch.rand(2, num_classes, 3, 2, device=device, dtype=torch.float64)
        labels = torch.randint(0, num_classes, (2, 3, 2), device=device)
        self.gradcheck(kornia.losses.dice_loss, (logits, labels), dtypes=[torch.float64, torch.int64])

    def test_dynamo(self, device, dtype, torch_optimizer):
        num_classes = 3
        logits = torch.rand(2, num_classes, 1, 2, device=device, dtype=dtype)
        labels = torch.rand(2, 1, 2) * num_classes
        labels = labels.to(device).long()

        op = kornia.losses.dice_loss
        op_optimized = torch_optimizer(op)

        self.assert_close(op(logits, labels), op_optimized(logits, labels))

    def test_module(self, device, dtype):
        num_classes = 3
        logits = torch.rand(2, num_classes, 1, 2, device=device, dtype=dtype)
        labels = torch.rand(2, 1, 2) * num_classes
        labels = labels.to(device).long()

        op = kornia.losses.dice_loss
        op_module = kornia.losses.DiceLoss()

        self.assert_close(op(logits, labels), op_module(logits, labels))
