import torch
import torch.nn as nn
from torch.nn.functional import mse_loss


class PSNRLoss(nn.Module):
    r"""Creates a criterion that calculates the PSNR between 2 images. Given an m x n image, the PSNR is:

    .. math::

        \text{PSNR} = 10 \log_{10} \bigg(\frac{\text{MAX}_I^2}{MSE(I,T)}\bigg)

    where

    .. math::

        \text{MSE}(I,T) = \frac{1}{mn}\sum_{i=0}^{m-1}\sum_{j=0}^{n-1} [I(i,j) - T(i,j)]^2

    and :math:`\text{MAX}_I` is the maximum possible input value
    (e.g for floating point images :math:`\text{MAX}_I=1`).


    Arguments:
        max_val (float): Maximum value of input

    Shape:
        - input: :math:`(*)`
        - approximation: :math:`(*)` same shape as input
        - output: :math:`()` a scalar

    Examples:
        >>> psnr_loss(torch.ones(1), 1.2*torch.ones(1), 2) # 10 * log(4/((1.2-1)**2)) / log(10)
        tensor(20.0000)

    Reference:
        https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio#Definition
    """

    def __init__(self, max_val: float) -> None:
        super(PSNRLoss, self).__init__()
        self.max_val: float = max_val

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:  # type: ignore
        return psnr_loss(input, target, self.max_val)


def psnr_loss(input: torch.Tensor, target: torch.Tensor, max_val: float) -> torch.Tensor:
    r"""Function that computes PSNR

    See :class:`~kornia.losses.PSNRLoss` for details.
    """
    if not torch.is_tensor(input) or not torch.is_tensor(target):
        raise TypeError(f"Expected 2 torch tensors but got {type(input)} and {type(target)}")

    if input.shape != target.shape:
        raise TypeError(f"Expected tensors of equal shapes, but got {input.shape} and {target.shape}")
    mse_val = mse_loss(input, target, reduction='mean')
    max_val_tensor: torch.Tensor = torch.tensor(max_val).to(input.device).to(input.dtype)
    return 10 * torch.log10(max_val_tensor * max_val_tensor / mse_val)
