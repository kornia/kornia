import torch
from torch import Tensor, nn

from kornia.core.check import KORNIA_CHECK, KORNIA_CHECK_IS_TENSOR, KORNIA_CHECK_SHAPE


def aepe(input: torch.Tensor, target: torch.Tensor, reduction: str = "mean") -> torch.Tensor:
    r"""Create a function that calculates the average endpoint error (AEPE) between 2 flow maps.

    AEPE is the endpoint error between two 2D vectors (e.g., optical flow).
    Given a h x w x 2 optical flow map, the AEPE is:

    .. math::

        \text{AEPE}=\frac{1}{hw}\sum_{i=1, j=1}^{h, w}\sqrt{(I_{i,j,1}-T_{i,j,1})^{2}+(I_{i,j,2}-T_{i,j,2})^{2}}

    Args:
        input: the input flow map with shape :math:`(*, 2)`.
        target: the target flow map with shape :math:`(*, 2)`.
        reduction : Specifies the reduction to apply to the
         output: ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
         ``'mean'``: the sum of the output will be divided by the number of elements
         in the output, ``'sum'``: the output will be summed.

    Return:
        the computed AEPE as a scalar.

    Examples:
        >>> ones = torch.ones(4, 4, 2)
        >>> aepe(ones, 1.2 * ones)
        tensor(0.2828)

    Reference:
        https://link.springer.com/content/pdf/10.1007/s11263-010-0390-2.pdf
    """
    KORNIA_CHECK_IS_TENSOR(input)
    KORNIA_CHECK_IS_TENSOR(target)
    KORNIA_CHECK_SHAPE(input, ["*", "2"])
    KORNIA_CHECK_SHAPE(target, ["*", "2"])
    KORNIA_CHECK(
        input.shape == target.shape, f"input and target shapes must be the same. Got: {input.shape} and {target.shape}"
    )

    epe: Tensor = ((input[..., 0] - target[..., 0]) ** 2 + (input[..., 1] - target[..., 1]) ** 2).sqrt()

    if reduction == "mean":
        epe = epe.mean()
    elif reduction == "sum":
        epe = epe.sum()
    elif reduction == "none":
        pass
    else:
        raise NotImplementedError("Invalid reduction option.")

    return epe


class AEPE(nn.Module):
    r"""Computes the average endpoint error (AEPE) between 2 flow maps.

    EPE is the endpoint error between two 2D vectors (e.g., optical flow).
    Given a h x w x 2 optical flow map, the AEPE is:

    .. math::

        \text{AEPE}=\frac{1}{hw}\sum_{i=1, j=1}^{h, w}\sqrt{(I_{i,j,1}-T_{i,j,1})^{2}+(I_{i,j,2}-T_{i,j,2})^{2}}

    Args:
        reduction : Specifies the reduction to apply to the
         output: ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
         ``'mean'``: the sum of the output will be divided by the number of elements
         in the output, ``'sum'``: the output will be summed.

    Shape:
        - input: :math:`(*, 2)`.
        - target :math:`(*, 2)`.
        - output: :math:`(1)`.

    Examples:
        >>> input1 = torch.rand(1, 4, 5, 2)
        >>> input2 = torch.rand(1, 4, 5, 2)
        >>> epe = AEPE(reduction="mean")
        >>> epe = epe(input1, input2)
    """

    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        self.reduction: str = reduction

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return aepe(input, target, self.reduction)


average_endpoint_error = aepe
