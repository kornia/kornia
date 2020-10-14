import torch
import torch.nn as nn


__all__ = [
    "add_weighted",
    "AddWeighted",
]


def add_weighted(src1: torch.Tensor, alpha: float,
                 src2: torch.Tensor, beta: float,
                 gamma: float) -> torch.Tensor:
    r"""Blend two Tensors.

    See :class:`~kornia.color.AddWeighted` for details.
    """
    if not isinstance(src1, torch.Tensor):
        raise TypeError("src1 should be a tensor. Got {}".format(type(src1)))

    if not isinstance(src2, torch.Tensor):
        raise TypeError("src2 should be a tensor. Got {}".format(type(src2)))

    if not isinstance(alpha, float):
        raise TypeError("alpha should be a float. Got {}".format(type(alpha)))

    if not isinstance(beta, float):
        raise TypeError("beta should be a float. Got {}".format(type(beta)))

    if not isinstance(gamma, float):
        raise TypeError("gamma should be a float. Got {}".format(type(gamma)))

    return src1 * alpha + src2 * beta + gamma


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
        return add_weighted(src1, self.alpha, src2, self.beta, self.gamma)
