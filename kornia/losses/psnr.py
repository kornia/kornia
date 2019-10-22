import torch
import torch.nn as nn
from torch.nn.functional import mse_loss


class PSNR(nn.Module):
    r"""Creates a criterion that calculates the PSNR between 2 images.

    Arguments:
        max_val (float): Maximum value os signal

    Shape:
        - signal: :math:`(*)`
        - approximation: :math:`(*)` same shape as signal
        - output: :math:`()` a scalar

    Examples:
        >>> kornia.losses.psnr(torch.ones(1), 1.2*torch.ones(1), 2)
        tensor(20.0000) # 10 * log(4/((1.2-1)**2)) / log(10)
    """

    def __init__(self, max_val):
        self.max_val = max_val


    def forward(self, signal: torch.tensor, approximation: torch.tensor):
        return psnr(signal, approximation, self.max_val)


def psnr (signal: torch.tensor, approximation: torch.tensor, max_val):
    if not torch.is_tensor(signal) or not torch.is_tensor(approximation):
        raise TypeError(f"Expected 2 torch tensors but got {type(signal)} and {type(approximation)}")
    if signal.shape != approximation.shape:
        raise TypeError(f"Expected tensors of equal shapes, but got {signal.shape} and {approximation.shape}")
    mse_val = mse_loss(signal, approximation, reduction = 'mean')
    return 10 * torch.log(max_val*max_val/mse_val)/torch.log(10)
