from typing import Tuple
import torch
import torch.nn as nn


def integral(input: torch.Tensor) -> torch.Tensor:
    r"""Calculates integral of the input tensor.

    Args:
        image: the input tensor with shape :math:`(B,C,H,W)` with shape :math:`(B,C,H,W)`.

    Returns:
        Integral tensor for the input tensor

    Examples:
        >>> input = torch.randn(2,2,5,5)
        >>> output = integral(input)
        >>> output.shape
        torch.Size([2, 2, 5, 5])
    """
    if not isinstance(input, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(input)}")
    S = torch.cumsum(input, dim=-1)
    S = torch.cumsum(S, dim=-2)
    return S


class Integral(nn.Module):
    r"""Calculates integral of the input tensor.

    Args:
        image: the input tensor with shape :math:`(B,C,H,W)`.

    Returns:
        Integral tensor for the input tensor with shape :math:`(B,C,H,W)`.
    Shape:
        - Input: :math:`(B, C, H, W)`
        - Output: :math:`(B, C, H, W)`
    Examples:
        >>> input = torch.randn(2,2,5,5)
        >>> integral_calc = Integral()
        >>> output = integral_calc(input)
        >>> output.shape
        torch.Size([2, 2, 5, 5])
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return integral(input)
