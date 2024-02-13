import pytest
import torch

import kornia

from testing.base import BaseTester


class TestCharbonnierLoss(BaseTester):
    @pytest.mark.parametrize("reduction", ["mean", "sum", "none", None])
    @pytest.mark.parametrize("shape", [(1, 2, 9, 9), (2, 4, 3, 6)])
    def test_smoke(self, device, dtype, reduction, shape):
        img1 = torch.rand(shape, device=device, dtype=dtype)
        img2 = torch.rand(shape, device=device, dtype=dtype)

        assert kornia.losses.charbonnier_loss(img1, img2, reduction) is not None

    def test_exception(self, device, dtype):
        img = torch.rand(3, 3, 3, device=device, dtype=dtype)

        # wrong reduction
        with pytest.raises(Exception) as execinfo:
            kornia.losses.charbonnier_loss(img, img, reduction="test")
        assert "Given type of reduction is not supported. Got: test" in str(execinfo)

        # Check if both are tensors
        with pytest.raises(TypeError) as errinfo:
            kornia.losses.charbonnier_loss(1.0, img)
        assert "Not a Tensor type. Got:" in str(errinfo)

        with pytest.raises(TypeError) as errinfo:
            kornia.losses.charbonnier_loss(img, 1.0)
        assert "Not a Tensor type. Got:" in str(errinfo)

        # Check if same shape
        img_b = torch.rand(1, 1, 3, 3, 4, device=device, dtype=dtype)
        with pytest.raises(TypeError) as errinfo:
            kornia.losses.charbonnier_loss(img, img_b, 3)
        assert "Not same shape for tensors. Got:" in str(errinfo)

    @pytest.mark.parametrize("shape", [(1, 3, 5, 5), (2, 5, 5)])
    def test_cardinality(self, shape, device, dtype):
        img = torch.rand(shape, device=device, dtype=dtype)

        actual = kornia.losses.charbonnier_loss(img, img, reduction="none")
        assert actual.shape == shape

        actual = kornia.losses.charbonnier_loss(img, img, reduction="sum")
        assert actual.shape == ()

        actual = kornia.losses.charbonnier_loss(img, img, reduction="mean")
        assert actual.shape == ()

    def test_gradcheck(self, device):
        img1 = torch.rand(2, 3, 3, 3, device=device, dtype=torch.float64)
        img2 = torch.rand(2, 3, 3, 3, device=device, dtype=torch.float64)

        self.gradcheck(kornia.losses.charbonnier_loss, (img1, img2))

    def test_dynamo(self, device, dtype, torch_optimizer):
        img1 = torch.rand(2, 3, 3, 3, device=device, dtype=dtype)
        img2 = torch.rand(2, 3, 3, 3, device=device, dtype=dtype)

        op = kornia.losses.charbonnier_loss
        op_optimized = torch_optimizer(op)

        self.assert_close(op(img1, img2), op_optimized(img1, img2))

    def test_module(self, device, dtype):
        img1 = torch.rand(2, 3, 3, 3, device=device, dtype=dtype)
        img2 = torch.rand(2, 3, 3, 3, device=device, dtype=dtype)

        op = kornia.losses.charbonnier_loss
        op_module = kornia.losses.CharbonnierLoss()

        self.assert_close(op(img1, img2), op_module(img1, img2))

    @pytest.mark.parametrize("reduction", ["mean", "sum"])
    @pytest.mark.parametrize("shape", [(1, 2, 9, 9), (2, 4, 3, 6)])
    def test_perfect_prediction(self, device, dtype, reduction, shape):
        # Sanity test
        img = torch.rand(shape, device=device, dtype=dtype)
        actual = kornia.losses.charbonnier_loss(img, img, reduction=reduction)
        expected = torch.tensor(0.0, device=device, dtype=dtype)
        self.assert_close(actual, expected)

        # Check loss computation
        img1 = torch.ones(shape, device=device, dtype=dtype)
        img2 = torch.zeros(shape, device=device, dtype=dtype)

        actual = kornia.losses.charbonnier_loss(img1, img2, reduction=reduction)

        if reduction == "mean":
            expected = torch.tensor(0.41421356237, device=device, dtype=dtype)
        elif reduction == "sum":
            expected = (torch.ones_like(img1, device=device, dtype=dtype) * 0.41421356237).sum()

        self.assert_close(actual, expected)
