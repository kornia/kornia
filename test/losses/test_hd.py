import numpy as np
import pytest
import torch
from scipy.ndimage import convolve
from torch import nn
from torch.autograd import gradcheck

import kornia
import kornia.testing as utils
from kornia.testing import assert_close


class HausdorffERLossNumpy(nn.Module):
    """Binary Hausdorff loss based on morphological erosion.

    Taken from https://github.com/PatRyg99/HausdorffLoss/blob/master/hausdorff_loss.py
    """

    def __init__(self, alpha=2.0, erosions=10, **kwargs):
        super().__init__()
        self.alpha = alpha
        self.erosions = erosions
        self.prepare_kernels()

    def prepare_kernels(self):
        cross = np.array([[[0, 1, 0], [1, 1, 1], [0, 1, 0]]])
        bound = np.array([[[0, 0, 0], [0, 1, 0], [0, 0, 0]]])

        self.kernel2D = cross * 0.2
        # NOTE: updated from np.array([bound, cross, bound]) * (1 / 7)
        self.kernel3D = np.array([bound, cross, bound]).squeeze()[None] * (1 / 7)

    @torch.no_grad()
    def perform_erosion(self, pred: np.ndarray, target: np.ndarray) -> np.ndarray:
        bound = (pred - target) ** 2

        if bound.ndim == 5:
            kernel = self.kernel3D
        elif bound.ndim == 4:
            kernel = self.kernel2D
        else:
            raise ValueError(f"Dimension {bound.ndim} is nor supported.")

        eroted = np.zeros_like(bound)

        for batch in range(len(bound)):
            for k in range(self.erosions):
                # compute convolution with kernel
                dilation = convolve(bound[batch], kernel, mode="constant", cval=0.0)

                # apply soft thresholding at 0.5 and normalize
                erosion = dilation - 0.5
                erosion[erosion < 0] = 0

                if erosion.ptp() != 0:
                    erosion = (erosion - erosion.min()) / erosion.ptp()

                # save erosion and add to loss
                bound[batch] = erosion
                eroted[batch] += erosion * (k + 1) ** self.alpha

        return eroted

    def forward_one(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Uses one binary channel: 1 - fg, 0 - bg
        pred: (b, 1, x, y, z) or (b, 1, x, y)
        target: (b, 1, x, y, z) or (b, 1, x, y)
        """
        assert pred.size(1) == target.size(1) == 1
        # pred = torch.sigmoid(pred)

        eroted = torch.from_numpy(self.perform_erosion(pred.cpu().numpy(), target.cpu().numpy())).to(
            dtype=pred.dtype, device=pred.device
        )

        loss = eroted.mean()

        return loss

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Uses one binary channel: 1 - fg, 0 - bg
        pred: (b, 1, x, y, z) or (b, 1, x, y)
        target: (b, 1, x, y, z) or (b, 1, x, y)
        """
        assert pred.dim() in (4, 5), "Only 2D and 3D supported"
        assert pred.dim() == target.dim() and target.size(1) == 1, "Prediction and target need to be of same dimension"
        return torch.stack(
            [
                self.forward_one(
                    pred[:, i : i + 1],
                    torch.where(
                        target == i,
                        torch.tensor(1, device=target.device, dtype=target.dtype),
                        torch.tensor(0, device=target.device, dtype=target.dtype),
                    ),
                )
                for i in range(pred.size(1))
            ]
        ).mean()


class TestHausdorffLoss:
    @pytest.mark.parametrize("reduction", ['mean', 'none', 'sum'])
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
        assert 'Only 2D images supported. Got ' in str(errinf)

        with pytest.raises(ValueError) as errinf:
            kornia.losses.HausdorffERLoss()(
                (torch.rand(1, 2, 1, 1) > 0.5) * 1, torch.tensor([[[[1]]]], dtype=torch.float32)
            )
        assert 'Expect long type target value in range' in str(errinf)

        with pytest.raises(ValueError) as errinf:
            kornia.losses.HausdorffERLoss()((torch.rand(1, 2, 1, 1) > 0.5) * 1, (torch.rand(1, 1, 1, 2) > 0.5) * 1)
        assert 'Prediction and target need to be of same size, and target should not be one-hot.' in str(errinf)

    def test_exception_3d(self):
        with pytest.raises(ValueError) as errinf:
            kornia.losses.HausdorffERLoss3D()((torch.rand(1, 2, 1) > 0.5) * 1, (torch.rand(1, 1, 1, 2) > 0.5) * 1)
        assert 'Only 3D images supported. Got ' in str(errinf)

        with pytest.raises(ValueError) as errinf:
            kornia.losses.HausdorffERLoss3D()(
                (torch.rand(1, 2, 1, 1, 1) > 0.5) * 1, torch.tensor([[[[[5]]]]], dtype=torch.float32)
            )
        assert 'Invalid target value' in str(errinf)

    @pytest.mark.parametrize(
        "hd,shape", [[kornia.losses.HausdorffERLoss, (50, 50)], [kornia.losses.HausdorffERLoss3D, (50, 50, 50)]]
    )
    def test_numeric(self, hd, shape, device, dtype):
        num_classes = 3
        logits = torch.rand(2, num_classes, *shape, dtype=dtype, device=device)
        labels = (torch.rand(2, 1, *shape, dtype=dtype, device=device) * (num_classes - 1)).long()
        loss = hd(k=10)
        loss_np = HausdorffERLossNumpy(erosions=10)

        expected = loss_np(logits, labels)
        actual = loss(logits, labels)
        assert_close(actual, expected)

    @pytest.mark.parametrize(
        "hd,shape", [[kornia.losses.HausdorffERLoss, (5, 5)], [kornia.losses.HausdorffERLoss3D, (5, 5, 5)]]
    )
    def test_gradcheck(self, hd, shape, device):
        num_classes = 3
        logits = torch.rand(2, num_classes, *shape, device=device)
        labels = (torch.rand(2, 1, *shape, device=device) * (num_classes - 1)).long()
        loss = hd(k=2)

        logits = utils.tensor_to_gradcheck_var(logits)  # to var
        assert gradcheck(loss, (logits, labels), raise_exception=True, fast_mode=True)
