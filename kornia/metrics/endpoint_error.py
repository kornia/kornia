import torch
from torch import Tensor

from kornia.core.check import KORNIA_CHECK, KORNIA_CHECK_IS_TENSOR, KORNIA_CHECK_SHAPE


def epe(input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    r"""Create a function that calculates the endpoint error (EPE) between 2 flow maps.

    EPE is the endpoint error between two 2D vecotrs (e.g., oprical flow).
    Given a h x w x 2 optical flow map, the EPE is:

    .. math::

        \text{EPE}=\frac{1}{hw}\sum_{i=1, j=1}^{h, w}(I_{i,j,1}-T_{i,j,1})^{2}+(I_{i,j,2}-T_{i,j,2})^{2}

    Args:
        input: the input flow map with shape :math:`(*, 2)`.
        target: the target flow map with shape :math:`(*, 2)`.

    Return:
        the computed EPE as a scalar.

    Examples:
        >>> ones = torch.ones(4, 4, 2)
        >>> epe(ones, 1.2 * ones)
        tensor(0.0800)

    Reference:
        https://link.springer.com/content/pdf/10.1007/s11263-010-0390-2.pdf
    """
    KORNIA_CHECK_IS_TENSOR(input)
    KORNIA_CHECK_IS_TENSOR(target)
    KORNIA_CHECK_SHAPE(input, ["*", "2"])
    KORNIA_CHECK_SHAPE(target, ["*", "2"])
    KORNIA_CHECK(input.shape == target.shape,
                 f"input and target shapes must be the same. Got: {input.shape} and {target.shape}")

    epe: Tensor = ((input[..., 0] - target[..., 0]) ** 2 + (input[..., 1] - target[..., 1]) ** 2).mean()

    return epe
