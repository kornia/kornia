import torch
import torch.nn as nn

__all__ = ["add_weighted", "AddWeighted"]


def add_weighted(src1: torch.Tensor, alpha: float, src2: torch.Tensor, beta: float, gamma: float) -> torch.Tensor:
    r"""Calculates the weighted sum of two Tensors.

    .. image:: _static/img/add_weighted.png

    The function calculates the weighted sum of two Tensors as follows:

    .. math::
        out = src1 * alpha + src2 * beta + gamma

    Args:
        src1: Tensor of shape :math:`(B, C, H, W)`.
        alpha: weight of the src1 elements.
        src2: Tensor of same size and channel number as src1 :math:`(B, C, H, W)`.
        beta: weight of the src2 elements.
        gamma: scalar added to each sum.

    Returns:
        Weighted Tensor of shape :math:`(B, C, H, W)`.

    Example:
        >>> input1 = torch.rand(1, 1, 5, 5)
        >>> input2 = torch.rand(1, 1, 5, 5)
        >>> output = add_weighted(input1, 0.5, input2, 0.5, 1.0)
        >>> output.shape
        torch.Size([1, 1, 5, 5])

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
        alpha: weight of the src1 elements.
        beta: weight of the src2 elements.
        gamma: scalar added to each sum.

    Shape:
        - Input1: Tensor of shape :math:`(B, C, H, W)`.
        - Input2: Tensor of shape :math:`(B, C, H, W)`.
        - Output: Weighted tensor of shape :math:`(B, C, H, W)`.

    Example:
        >>> input1 = torch.rand(1, 1, 5, 5)
        >>> input2 = torch.rand(1, 1, 5, 5)
        >>> output = AddWeighted(0.5, 0.5, 1.0)(input1, input2)
        >>> output.shape
        torch.Size([1, 1, 5, 5])

    """

    def __init__(self, alpha: float, beta: float, gamma: float) -> None:
        super(AddWeighted, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def forward(self, src1: torch.Tensor, src2: torch.Tensor) -> torch.Tensor:  # type: ignore
        return add_weighted(src1, self.alpha, src2, self.beta, self.gamma)
