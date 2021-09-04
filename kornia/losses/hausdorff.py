from typing import Callable

import torch
import torch.nn as nn


class _HausdorffERLossBase(nn.Module):
    """Base class for binary Hausdorff loss based on morphological erosion.

    This is an Hausdorff Distance (HD) Loss that based on morphological erosion,which provided
    a differentiable approximation of Hausdorff distance as stated in :cite:`karimi2019reducing`.
    The code is refactored on top of `here <https://github.com/PatRyg99/HausdorffLoss/
        blob/master/hausdorff_loss.py>`__.

    Args:
        alpha: controls the erosion rate in each iteration. Default: 2.0.
        erosions: the number of iterations of erosion. Default: 10.
    """
    conv: Callable
    avg_pool: Callable

    def __init__(self, alpha: float = 2.0, erosions: int = 10) -> None:
        super().__init__()
        self.alpha = alpha
        self.erosions = erosions
        self.register_buffer("kernel", self.get_kernel())

    def get_kernel(self) -> torch.Tensor:
        raise NotImplementedError

    def perform_erosion(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        bound = (pred - target) ** 2

        kernel = torch.as_tensor(self.kernel, device=pred.device, dtype=pred.dtype)
        eroted = torch.zeros_like(bound)

        for k in range(self.erosions):
            # Same padding, assuming kernel is odd and square (cube) shaped.
            padding = (kernel.size(-1) - 1) // 2
            # compute convolution with kernel
            dilation = self.conv(bound, weight=kernel, padding=padding, groups=1)
            # apply soft thresholding at 0.5 and normalize
            erosion = dilation - 0.5
            erosion[erosion < 0] = 0

            # image-wise differences for 2D images
            erosion_max = self.avg_pool(erosion)
            erosion_min = - self.avg_pool(- erosion)
            # No normalization needed if `max - min = 0`
            to_norm = (erosion_max.squeeze() - erosion_min.squeeze()) != 0
            erosion[to_norm] = (erosion[to_norm] - erosion_min[to_norm]) / (
                erosion_max[to_norm] - erosion_min[to_norm])

            # save erosion and add to loss
            eroted = eroted + erosion * (k + 1) ** self.alpha
            bound = erosion

        return eroted

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute Hausdorff loss.

        Args:
            pred: predicted tensor with a shape of :math:`(B, C, H, W)` or :math:`(B, C, D, H, W)`.
                Each channel is as binary as: 1 -> fg, 0 -> bg.
            target: target tensor with a shape of :math:`(B, 1, H, W)` or :math:`(B, C, D, H, W)`.
        """
        assert pred.shape[2:] == target.shape[2:] and pred.size(0) == target.size(0) and target.size(1) == 1, (
            "Prediction and target need to be of same size, and target should not be one-hot."
            f"Got {pred.shape} and {target.shape}."
        )
        assert pred.size(1) <= target.max()
        return torch.stack([
            self.perform_erosion(pred[:, i:i + 1], torch.where(target == i, 1, 0)).mean()
            for i in range(pred.size(1))
        ]).mean()


class HausdorffERLoss(_HausdorffERLossBase):
    """Binary Hausdorff loss based on morphological erosion.

    This is an Hausdorff Distance (HD) Loss that based on morphological erosion,which provided
    a differentiable approximation of Hausdorff distance as stated in :cite:`karimi2019reducing`.
    The code is refactored on top of `here <https://github.com/PatRyg99/HausdorffLoss/
        blob/master/hausdorff_loss.py>`__.

    Args:
        alpha: controls the erosion rate in each iteration. Default: 2.0.
        erosions: the number of iterations of erosion. Default: 10.
    
    Examples:
        >>> hdloss = HausdorffERLoss()
        >>> input = torch.randn(5, 3, 20, 20)
        >>> target = torch.randn(5, 1, 20, 20).long() * 3
        >>> res = hdloss(input, target)
    """
    conv = torch.conv2d
    avg_pool = nn.AdaptiveAvgPool2d(1)

    def get_kernel(self) -> None:
        cross = torch.tensor([[[0, 1, 0], [1, 1, 1], [0, 1, 0]]])
        kernel = cross * 0.2
        return kernel[None]

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute Hausdorff loss.

        Args:
            pred: predicted tensor with a shape of :math:`(B, C, H, W)`.
                Each channel is as binary as: 1 -> fg, 0 -> bg.
            target: target tensor with a shape of :math:`(B, 1, H, W)`.
        """
        assert pred.dim() == 4, f"Only 2D images supported. Got {pred.dim()}."
        return super().forward(pred, target)


class HausdorffERLoss3D(_HausdorffERLossBase):
    """Binary 3D Hausdorff loss based on morphological erosion.

    This is a 3D Hausdorff Distance (HD) Loss that based on morphological erosion,which provided
    a differentiable approximation of Hausdorff distance as stated in :cite:`karimi2019reducing`.
    The code is refactored on top of `here <https://github.com/PatRyg99/HausdorffLoss/
        blob/master/hausdorff_loss.py>`__.

    Args:
        alpha: controls the erosion rate in each iteration. Default: 2.0.
        erosions: the number of iterations of erosion. Default: 10.

    Examples:
        >>> hdloss = HausdorffERLoss3D()
        >>> input = torch.randn(5, 3, 20, 20, 20)
        >>> target = torch.randn(5, 1, 20, 20, 20).long() * 3
        >>> res = hdloss(input, target)
    """

    conv = torch.conv3d
    avg_pool = nn.AdaptiveAvgPool3d(1)

    def get_kernel(self) -> None:
        # Kernel from cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        cross = torch.tensor([[[0, 1, 0], [1, 1, 1], [0, 1, 0]]])
        bound = torch.tensor([[[0, 0, 0], [0, 1, 0], [0, 0, 0]]])
        kernel = torch.stack([bound, cross, bound], dim=1) * (1 / 7)
        return kernel[None]

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute 3D Hausdorff loss.

        Args:
            pred: predicted tensor with a shape of :math:`(B, C, D, H, W)`.
                Each channel is as binary as: 1 -> fg, 0 -> bg.
            target: target tensor with a shape of :math:`(B, 1, D, H, W)`.
        """
        assert pred.dim() == 5, f"Only 3D images supported. Got {pred.dim()}."
        return super().forward(pred, target)
