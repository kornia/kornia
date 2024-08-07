from typing import Optional, Tuple

from kornia.core import ImageModule as Module
from kornia.core import Tensor
from kornia.core.check import KORNIA_CHECK, KORNIA_CHECK_SHAPE


def integral_tensor(input: Tensor, dim: Optional[Tuple[int, ...]] = None) -> Tensor:
    """Calculates integral of the input tensor.

    The algorithm computes the integral image by summing over the specified dimensions.

    In case dim is specified, the contained dimensions must be unique and sorted in ascending order
    and not exceed the number of dimensions of the input tensor.

    Args:
        input: the input tensor with shape :math:`(*, D)`. Where D is the number of dimensions.
        dim: the dimension to be summed.

    Returns:
        Integral tensor for the input tensor with shape :math:`(*, D)`.

    Examples:
        >>> input = torch.ones(3, 5)
        >>> output = integral_tensor(input, (-2, -1))
        >>> output
        tensor([[ 1.,  2.,  3.,  4.,  5.],
                [ 2.,  4.,  6.,  8., 10.],
                [ 3.,  6.,  9., 12., 15.]])
    """
    KORNIA_CHECK_SHAPE(input, ["*", "D"])

    if dim is None:
        dim = (-1,)

    KORNIA_CHECK(len(dim) > 0, "dim must be a non-empty tuple.")
    KORNIA_CHECK(len(dim) <= len(input.shape), "dim must be a tuple of length <= input.shape.")

    output = input
    for i in dim:
        output = output.cumsum(i)
    return output


def integral_image(image: Tensor) -> Tensor:
    r"""Calculates integral of the input image tensor.

    This particular version sums over the last two dimensions.

    Args:
        image: the input image tensor with shape :math:`(*, H, W)`.

    Returns:
        Integral tensor for the input image tensor with shape :math:`(*, H, W)`.

    Examples:
        >>> input = torch.ones(1, 5, 5)
        >>> output = integral_image(input)
        >>> output
        tensor([[[ 1.,  2.,  3.,  4.,  5.],
                 [ 2.,  4.,  6.,  8., 10.],
                 [ 3.,  6.,  9., 12., 15.],
                 [ 4.,  8., 12., 16., 20.],
                 [ 5., 10., 15., 20., 25.]]])
    """
    KORNIA_CHECK_SHAPE(image, ["*", "H", "W"])

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
        >>> input = torch.ones(3, 5)
        >>> dim = (-2, -1)
        >>> output = IntegralTensor(dim)(input)
        >>> output
        tensor([[ 1.,  2.,  3.,  4.,  5.],
                [ 2.,  4.,  6.,  8., 10.],
                [ 3.,  6.,  9., 12., 15.]])
    """

    def __init__(self, dim: Optional[Tuple[int, ...]] = None) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, input: Tensor) -> Tensor:
        return integral_tensor(input, self.dim)


class IntegralImage(Module):
    """Calculates integral of the input image tensor.

    This particular version sums over the last two dimensions.

    Args:
        image: the input image tensor with shape :math:`(B,C,H,W)`.

    Returns:
        Integral tensor for the input image tensor with shape :math:`(B,C,H,W)`.

    Shape:
        - Input: :math:`(B, C, H, W)`
        - Output: :math:`(B, C, H, W)`

    Examples:
        >>> input = torch.ones(1, 5, 5)
        >>> output = IntegralImage()(input)
        >>> output
        tensor([[[ 1.,  2.,  3.,  4.,  5.],
                 [ 2.,  4.,  6.,  8., 10.],
                 [ 3.,  6.,  9., 12., 15.],
                 [ 4.,  8., 12., 16., 20.],
                 [ 5., 10., 15., 20., 25.]]])
    """

    def forward(self, input: Tensor) -> Tensor:
        return integral_image(input)
