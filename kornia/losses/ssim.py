from __future__ import annotations

import torch
from torch import nn

from kornia import metrics


def ssim_loss(
    img1: torch.Tensor,
    img2: torch.Tensor,
    window_size: int,
    max_val: float = 1.0,
    eps: float = 1e-12,
    reduction: str = "mean",
    padding: str = "same",
) -> torch.Tensor:
    r"""Function that computes a loss based on the SSIM measurement.

    The loss, or the Structural dissimilarity (DSSIM) is described as:

    .. math::

      \text{loss}(x, y) = \frac{1 - \text{SSIM}(x, y)}{2}

    See :meth:`~kornia.losses.ssim` for details about SSIM.

    Args:
        img1: the first input image with shape :math:`(B, C, H, W)`.
        img2: the second input image with shape :math:`(B, C, H, W)`.
        window_size: the size of the gaussian kernel to smooth the images.
        max_val: the dynamic range of the images.
        eps: Small value for numerically stability when dividing.
        reduction : Specifies the reduction to apply to the
         output: ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
         ``'mean'``: the sum of the output will be divided by the number of elements
         in the output, ``'sum'``: the output will be summed.
        padding: ``'same'`` | ``'valid'``. Whether to only use the "valid" convolution
         area to compute SSIM to match the MATLAB implementation of original SSIM paper.

    Returns:
        The loss based on the ssim index.

    Examples:
        >>> input1 = torch.rand(1, 4, 5, 5)
        >>> input2 = torch.rand(1, 4, 5, 5)
        >>> loss = ssim_loss(input1, input2, 5)
    """
    # compute the ssim map
    ssim_map: torch.Tensor = metrics.ssim(img1, img2, window_size, max_val, eps, padding)

    # compute and reduce the loss
    loss = torch.clamp((1.0 - ssim_map) / 2, min=0, max=1)

    if reduction == "mean":
        loss = torch.mean(loss)
    elif reduction == "sum":
        loss = torch.sum(loss)
    elif reduction == "none":
        pass
    else:
        raise NotImplementedError("Invalid reduction option.")

    return loss


class SSIMLoss(nn.Module):
    r"""Create a criterion that computes a loss based on the SSIM measurement.

    The loss, or the Structural dissimilarity (DSSIM) is described as:

    .. math::

      \text{loss}(x, y) = \frac{1 - \text{SSIM}(x, y)}{2}

    See :meth:`~kornia.losses.ssim_loss` for details about SSIM.

    Args:
        window_size: the size of the gaussian kernel to smooth the images.
        max_val: the dynamic range of the images.
        eps: Small value for numerically stability when dividing.
        reduction : Specifies the reduction to apply to the
         output: ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
         ``'mean'``: the sum of the output will be divided by the number of elements
         in the output, ``'sum'``: the output will be summed.
        padding: ``'same'`` | ``'valid'``. Whether to only use the "valid" convolution
         area to compute SSIM to match the MATLAB implementation of original SSIM paper.

    Returns:
        The loss based on the ssim index.

    Examples:
        >>> input1 = torch.rand(1, 4, 5, 5)
        >>> input2 = torch.rand(1, 4, 5, 5)
        >>> criterion = SSIMLoss(5)
        >>> loss = criterion(input1, input2)
    """

    def __init__(
        self, window_size: int, max_val: float = 1.0, eps: float = 1e-12, reduction: str = "mean", padding: str = "same"
    ) -> None:
        super().__init__()
        self.window_size: int = window_size
        self.max_val: float = max_val
        self.eps: float = eps
        self.reduction: str = reduction
        self.padding: str = padding

    def forward(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        return ssim_loss(img1, img2, self.window_size, self.max_val, self.eps, self.reduction, self.padding)
