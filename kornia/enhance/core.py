from typing import Union

from kornia.core import ImageModule as Module
from kornia.core import Tensor, tensor
from kornia.core.check import KORNIA_CHECK, KORNIA_CHECK_IS_TENSOR


def add_weighted(
    src1: Tensor, alpha: Union[float, Tensor], src2: Tensor, beta: Union[float, Tensor], gamma: Union[float, Tensor]
) -> Tensor:
    r"""Calculate the weighted sum of two Tensors.

    .. image:: _static/img/add_weighted.png

    The function calculates the weighted sum of two Tensors as follows:

    .. math::
        out = src1 * alpha + src2 * beta + gamma

    Args:
        src1: Tensor with an arbitrary shape, equal to shape of src2.
        alpha: weight of the src1 elements as Union[float, Tensor].
        src2: Tensor with an arbitrary shape, equal to shape of src1.
        beta: weight of the src2 elements as Union[float, Tensor].
        gamma: scalar added to each sum as Union[float, Tensor].

    Returns:
        Weighted Tensor with shape equal to src1 and src2 shapes.

    Example:
        >>> input1 = torch.rand(1, 1, 5, 5)
        >>> input2 = torch.rand(1, 1, 5, 5)
        >>> output = add_weighted(input1, 0.5, input2, 0.5, 1.0)
        >>> output.shape
        torch.Size([1, 1, 5, 5])

    Notes:
        Tensor alpha/beta/gamma have to be with shape broadcastable to src1 and src2 shapes.
    """
    KORNIA_CHECK_IS_TENSOR(src1)
    KORNIA_CHECK_IS_TENSOR(src2)
    KORNIA_CHECK(src1.shape == src2.shape, f"src1 and src2 have different shapes. Got {src1.shape} and {src2.shape}")

    if isinstance(alpha, Tensor):
        KORNIA_CHECK(src1.shape == alpha.shape, "alpha has a different shape than src.")
    else:
        alpha = tensor(alpha, dtype=src1.dtype, device=src1.device)

    if isinstance(beta, Tensor):
        KORNIA_CHECK(src1.shape == beta.shape, "beta has a different shape than src.")
    else:
        beta = tensor(beta, dtype=src1.dtype, device=src1.device)

    if isinstance(gamma, Tensor):
        KORNIA_CHECK(src1.shape == gamma.shape, "gamma has a different shape than src.")
    else:
        gamma = tensor(gamma, dtype=src1.dtype, device=src1.device)

    return src1 * alpha + src2 * beta + gamma


class AddWeighted(Module):
    r"""Calculate the weighted sum of two Tensors.

    The function calculates the weighted sum of two Tensors as follows:

    .. math::
        out = src1 * alpha + src2 * beta + gamma

    Args:
        alpha: weight of the src1 elements as Union[float, Tensor].
        beta: weight of the src2 elements as Union[float, Tensor].
        gamma: scalar added to each sum as Union[float, Tensor].

    Shape:
        - Input1: Tensor with an arbitrary shape, equal to shape of Input2.
        - Input2: Tensor with an arbitrary shape, equal to shape of Input1.
        - Output: Weighted tensor with shape equal to src1 and src2 shapes.

    Example:
        >>> input1 = torch.rand(1, 1, 5, 5)
        >>> input2 = torch.rand(1, 1, 5, 5)
        >>> output = AddWeighted(0.5, 0.5, 1.0)(input1, input2)
        >>> output.shape
        torch.Size([1, 1, 5, 5])

    Notes:
        Tensor alpha/beta/gamma have to be with shape broadcastable to src1 and src2 shapes.
    """

    def __init__(self, alpha: Union[float, Tensor], beta: Union[float, Tensor], gamma: Union[float, Tensor]) -> None:
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def forward(self, src1: Tensor, src2: Tensor) -> Tensor:
        return add_weighted(src1, self.alpha, src2, self.beta, self.gamma)
