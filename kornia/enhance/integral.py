from typing import Optional, Tuple

from kornia.core import Module, Tensor
from kornia.testing import KORNIA_CHECK_SHAPE


def integral_tensor(input: Tensor, dim: Optional[Tuple[int]] = None) -> Tensor:
    r"""Calculates integral of the input tensor.

    Args:
        input: the input tensor with shape :math:`(B,C,H,W)`.

    Returns:
        Integral tensor for the input tensor.

    Examples:
        >>> input = torch.randn(2,2,5,5)
        >>> output = integral(input)
        >>> output.shape
        torch.Size([2, 2, 5, 5])
    """
    KORNIA_CHECK_SHAPE(input, ["B", "C", "H", "W"])
    if dim is None:
        dim = (-1,)
    S: Tensor = input
    for i in dim:
        S = S.cumsum(i)
    return S


def integral_image(image: Tensor) -> Tensor:
    r"""Calculates integral of the input image tensor.

    Args:
      image: the image to be summed.
    """
    KORNIA_CHECK_SHAPE(input, ["B", "C", "H", "W"])
    return integral_tensor(image, (-2, -1))


class IntegralTensor(Module):
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

    def forward(self, input: Tensor) -> Tensor:
        return integral_tensor(input)


class IntegralImage(Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, input: Tensor) -> Tensor:
        return integral_image(input)
