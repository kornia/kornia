import torch
import torch.nn as nn

import kornia


__all__ = [
    "AddWeighted",
]


class AddWeighted(nn.Module):
    r"""Calculates the weighted sum of two Tensors.

    The function calculates the weighted sum of two Tensors as follows:

    .. math::
        out = src1 * alpha + src2 * beta + gamma

    Args:
        src1 (torch.Tensor): Tensor.
        alpha (float): weight of the src1 elements.
        src2 (torch.Tensor): Tensor of same size and channel number as src1.
        beta (float): weight of the src2 elements.
        gamma (float): scalar added to each sum.

    Returns:
        torch.Tensor: Weighted Tensor.
    """

    def __init__(self, alpha: float, beta: float, gamma: float) -> None:
        super(AddWeighted, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def forward(self, src1: torch.Tensor, src2: torch.Tensor) -> torch.Tensor:  # type: ignore
        return kornia.enhance.add_weighted(src1, self.alpha, src2, self.beta, self.gamma)
