from __future__ import annotations

from typing import Callable

import torch
from torch import nn

from kornia.core import Module, Tensor, as_tensor, stack, tensor, where, zeros_like


class _HausdorffERLossBase(Module):
    """Base class for binary Hausdorff loss based on morphological erosion.

    This is an Hausdorff Distance (HD) Loss that based on morphological erosion,which provided
    a differentiable approximation of Hausdorff distance as stated in :cite:`karimi2019reducing`.
    The code is refactored on top of `here <https://github.com/PatRyg99/HausdorffLoss/
        blob/master/hausdorff_loss.py>`__.

    Args:
        alpha: controls the erosion rate in each iteration.
        k: the number of iterations of erosion.
        reduction: Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'.
            'none': no reduction will be applied, 'mean': the weighted mean of the output is taken,
            'sum': the output will be summed.

    Returns:
        Estimated Hausdorff Loss.
    """

    conv: Callable[..., Tensor]
    max_pool: Callable[..., Tensor]

    def __init__(self, alpha: float = 2.0, k: int = 10, reduction: str = "mean") -> None:
        super().__init__()
        self.alpha = alpha
        self.k = k
        self.reduction = reduction
        self.register_buffer("kernel", self.get_kernel())

    def get_kernel(self) -> Tensor:
        """Get kernel for image morphology convolution."""
        raise NotImplementedError

    def perform_erosion(self, pred: Tensor, target: Tensor) -> Tensor:
        bound = (pred - target) ** 2

        kernel = as_tensor(self.kernel, device=pred.device, dtype=pred.dtype)
        eroded = zeros_like(bound, device=pred.device, dtype=pred.dtype)
        mask = torch.ones_like(bound, device=pred.device, dtype=torch.bool)

        # Same padding, assuming kernel is odd and square (cube) shaped.
        padding = (kernel.size(-1) - 1) // 2
        for k in range(self.k):
            # compute convolution with kernel
            dilation = self.conv(bound, weight=kernel, padding=padding, groups=1)
            # apply soft thresholding at 0.5 and normalize
            erosion = dilation - 0.5
            erosion[erosion < 0] = 0

            # image-wise differences for 2D images
            erosion_max = self.max_pool(erosion)
            erosion_min = -self.max_pool(-erosion)
            # No normalization needed if `max - min = 0`
            _to_norm = (erosion_max - erosion_min) != 0
            to_norm = _to_norm.squeeze()
            if to_norm.any():
                # NOTE: avoid in-place ops like below, which will not pass gradcheck:
                #       erosion[to_norm] = (erosion[to_norm] - erosion_min[to_norm]) / (
                #           erosion_max[to_norm] - erosion_min[to_norm])
                _erosion_to_fill = (erosion - erosion_min) / (erosion_max - erosion_min)
                erosion = where(mask * _to_norm, _erosion_to_fill, erosion)

            # save erosion and add to loss
            eroded = eroded + erosion * (k + 1) ** self.alpha
            bound = erosion

        return eroded

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        """Compute Hausdorff loss.

        Args:
            pred: predicted tensor with a shape of :math:`(B, C, H, W)` or :math:`(B, C, D, H, W)`.
                Each channel is as binary as: 1 -> fg, 0 -> bg.
            target: target tensor with a shape of :math:`(B, 1, H, W)` or :math:`(B, C, D, H, W)`.

        Returns:
            Estimated Hausdorff Loss.
        """
        if not (pred.shape[2:] == target.shape[2:] and pred.size(0) == target.size(0) and target.size(1) == 1):
            raise ValueError(
                "Prediction and target need to be of same size, and target should not be one-hot."
                f"Got {pred.shape} and {target.shape}."
            )

        if pred.size(1) < target.max().item():
            raise ValueError("Invalid target value.")

        out = stack(
            [
                self.perform_erosion(
                    pred[:, i : i + 1],
                    where(
                        target == i,
                        tensor(1, device=target.device, dtype=target.dtype),
                        tensor(0, device=target.device, dtype=target.dtype),
                    ),
                )
                for i in range(pred.size(1))
            ]
        )

        if self.reduction == "mean":
            out = out.mean()
        elif self.reduction == "sum":
            out = out.sum()
        elif self.reduction == "none":
            pass
        else:
            raise NotImplementedError(f"reduction `{self.reduction}` has not been implemented yet.")

        return out


class HausdorffERLoss(_HausdorffERLossBase):
    r"""Binary Hausdorff loss based on morphological erosion.

    Hausdorff Distance loss measures the maximum distance of a predicted segmentation boundary to
    the nearest ground-truth edge pixel. For two segmentation point sets X and Y ,
    the one-sided HD from X to Y is defined as:

    .. math::

        hd(X,Y) = \max_{x \in X} \min_{y \in Y}||x - y||_2

    Furthermore, the bidirectional HD is:

    .. math::

        HD(X,Y) = max(hd(X, Y), hd(Y, X))

    This is an Hausdorff Distance (HD) Loss that based on morphological erosion, which provided
    a differentiable approximation of Hausdorff distance as stated in :cite:`karimi2019reducing`.
    The code is refactored on top of `here <https://github.com/PatRyg99/HausdorffLoss/
    blob/master/hausdorff_loss.py>`__.

    Args:
        alpha: controls the erosion rate in each iteration.
        k: the number of iterations of erosion.
        reduction: Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'.
            'none': no reduction will be applied, 'mean': the weighted mean of the output is taken,
            'sum': the output will be summed.

    Examples:
        >>> hdloss = HausdorffERLoss()
        >>> input = torch.randn(5, 3, 20, 20)
        >>> target = (torch.rand(5, 1, 20, 20) * 2).long()
        >>> res = hdloss(input, target)
    """

    conv = torch.conv2d
    max_pool = nn.AdaptiveMaxPool2d(1)

    def get_kernel(self) -> Tensor:
        """Get kernel for image morphology convolution."""
        cross = tensor([[[0, 1, 0], [1, 1, 1], [0, 1, 0]]])
        kernel = cross * 0.2
        return kernel[None]

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        """Compute Hausdorff loss.

        Args:
            pred: predicted tensor with a shape of :math:`(B, C, H, W)`.
                Each channel is as binary as: 1 -> fg, 0 -> bg.
            target: target tensor with a shape of :math:`(B, 1, H, W)`.

        Returns:
            Estimated Hausdorff Loss.
        """
        if pred.dim() != 4:
            raise ValueError(f"Only 2D images supported. Got {pred.dim()}.")

        if not (target.max() < pred.size(1) and target.min() >= 0 and target.dtype == torch.long):
            raise ValueError(
                f"Expect long type target value in range (0, {pred.size(1)}). ({target.min()}, {target.max()})"
            )
        return super().forward(pred, target)


class HausdorffERLoss3D(_HausdorffERLossBase):
    r"""Binary 3D Hausdorff loss based on morphological erosion.

    Hausdorff Distance loss measures the maximum distance of a predicted segmentation boundary to
    the nearest ground-truth edge pixel. For two segmentation point sets X and Y ,
    the one-sided HD from X to Y is defined as:

    .. math::

        hd(X,Y) = \max_{x \in X} \min_{y \in Y}||x - y||_2

    Furthermore, the bidirectional HD is:

    .. math::

        HD(X,Y) = max(hd(X, Y), hd(Y, X))

    This is a 3D Hausdorff Distance (HD) Loss that based on morphological erosion, which provided
    a differentiable approximation of Hausdorff distance as stated in :cite:`karimi2019reducing`.
    The code is refactored on top of `here <https://github.com/PatRyg99/HausdorffLoss/
    blob/master/hausdorff_loss.py>`__.

    Args:
        alpha: controls the erosion rate in each iteration.
        k: the number of iterations of erosion.
        reduction: Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'.
            'none': no reduction will be applied, 'mean': the weighted mean of the output is taken,
            'sum': the output will be summed.

    Examples:
        >>> hdloss = HausdorffERLoss3D()
        >>> input = torch.randn(5, 3, 20, 20, 20)
        >>> target = (torch.rand(5, 1, 20, 20, 20) * 2).long()
        >>> res = hdloss(input, target)
    """

    conv = torch.conv3d
    max_pool = nn.AdaptiveMaxPool3d(1)

    def get_kernel(self) -> Tensor:
        """Get kernel for image morphology convolution."""
        cross = tensor([[[0, 1, 0], [1, 1, 1], [0, 1, 0]]])
        bound = tensor([[[0, 0, 0], [0, 1, 0], [0, 0, 0]]])
        # NOTE: The original repo claimed it shaped as (3, 1, 3, 3)
        #    which Jian suspect it is wrongly implemented.
        # https://github.com/PatRyg99/HausdorffLoss/blob/9f580acd421af648e74b45d46555ccb7a876c27c/hausdorff_loss.py#L94
        kernel = stack([bound, cross, bound], 1) * (1 / 7)
        return kernel[None]

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        """Compute 3D Hausdorff loss.

        Args:
            pred: predicted tensor with a shape of :math:`(B, C, D, H, W)`.
                Each channel is as binary as: 1 -> fg, 0 -> bg.
            target: target tensor with a shape of :math:`(B, 1, D, H, W)`.

        Returns:
            Estimated Hausdorff Loss.
        """
        if pred.dim() != 5:
            raise ValueError(f"Only 3D images supported. Got {pred.dim()}.")

        return super().forward(pred, target)
