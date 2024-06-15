import pytest
import torch

import kornia

from testing.base import BaseTester


class TestHausdorffLoss(BaseTester):
    @pytest.mark.parametrize("reduction", ["mean", "none", "sum"])
    @pytest.mark.parametrize(
        "hd,shape", [[kornia.losses.HausdorffERLoss, (10, 10)], [kornia.losses.HausdorffERLoss3D, (10, 10, 10)]]
    )
    def test_smoke_none(self, hd, shape, reduction, device, dtype):
        num_classes = 3
        logits = torch.rand(2, num_classes, *shape, dtype=dtype, device=device)
        labels = (torch.rand(2, 1, *shape, dtype=dtype, device=device) * (num_classes - 1)).long()
        loss = hd(reduction=reduction)

        loss(logits, labels)

    def test_exception_2d(self):
        with pytest.raises(ValueError) as errinf:
            kornia.losses.HausdorffERLoss()((torch.rand(1, 2, 1) > 0.5) * 1, (torch.rand(1, 1, 1, 2) > 0.5) * 1)
        assert "Only 2D images supported. Got " in str(errinf)

        with pytest.raises(ValueError) as errinf:
            kornia.losses.HausdorffERLoss()(
                (torch.rand(1, 2, 1, 1) > 0.5) * 1, torch.tensor([[[[1]]]], dtype=torch.float32)
            )
        assert "Expect long type target value in range" in str(errinf)

        with pytest.raises(ValueError) as errinf:
            kornia.losses.HausdorffERLoss()((torch.rand(1, 2, 1, 1) > 0.5) * 1, (torch.rand(1, 1, 1, 2) > 0.5) * 1)
        assert "Prediction and target need to be of same size, and target should not be one-hot." in str(errinf)

    def test_exception_3d(self):
        with pytest.raises(ValueError) as errinf:
            kornia.losses.HausdorffERLoss3D()((torch.rand(1, 2, 1) > 0.5) * 1, (torch.rand(1, 1, 1, 2) > 0.5) * 1)
        assert "Only 3D images supported. Got " in str(errinf)

        with pytest.raises(ValueError) as errinf:
            kornia.losses.HausdorffERLoss3D()(
                (torch.rand(1, 2, 1, 1, 1) > 0.5) * 1, torch.tensor([[[[[5]]]]], dtype=torch.float32)
            )
        assert "Invalid target value" in str(errinf)

    def test_numeric(self, device, dtype):
        if dtype == torch.float64:
            pytest.xfail("Sometimes failing on float64")
        num_classes = 3
        shape = (50, 50)
        hd = kornia.losses.HausdorffERLoss
        logits = torch.rand(2, num_classes, *shape, dtype=dtype, device=device)
        labels = (torch.rand(2, 1, *shape, dtype=dtype, device=device) * (num_classes - 1)).long()
        loss = hd(k=10)

        expected = torch.tensor(0.025, device=device, dtype=dtype)

        actual = loss(logits, labels)
        self.assert_close(actual, expected, rtol=0.005, atol=0.005)

    def test_numeric_3d(self, device, dtype):
        num_classes = 3
        shape = (50, 50, 50)
        hd = kornia.losses.HausdorffERLoss3D
        logits = torch.rand(2, num_classes, *shape, dtype=dtype, device=device)
        labels = (torch.rand(2, 1, *shape, dtype=dtype, device=device) * (num_classes - 1)).long()
        loss = hd(k=10)

        expected = torch.tensor(0.011, device=device, dtype=dtype)
        actual = loss(logits, labels)
        self.assert_close(actual, expected, rtol=1e-2, atol=1e-2)

    @pytest.mark.parametrize(
        "hd,shape", [[kornia.losses.HausdorffERLoss, (5, 5)], [kornia.losses.HausdorffERLoss3D, (5, 5, 5)]]
    )
    def test_gradcheck(self, hd, shape, device):
        num_classes = 3
        logits = torch.rand(2, num_classes, *shape, device=device)
        labels = (torch.rand(2, 1, *shape, device=device) * (num_classes - 1)).long()
        loss = hd(k=2)

        self.gradcheck(loss, (logits, labels), dtypes=[torch.float64, torch.int64])
