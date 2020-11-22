from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import kornia
from kornia.filters import get_gaussian_kernel2d, filter2D


def _compute_zero_padding(kernel_size: int) -> int:
    """Computes zero padding."""
    return (kernel_size - 1) // 2


class SSIM(kornia.nn.SSIM):
    r"""Creates a criterion that measures the Structural Similarity (SSIM)
    index between each element in the input `x` and target `y`.

    The index can be described as:

    .. math::

      \text{SSIM}(x, y) = \frac{(2\mu_x\mu_y+c_1)(2\sigma_{xy}+c_2)}
      {(\mu_x^2+\mu_y^2+c_1)(\sigma_x^2+\sigma_y^2+c_2)}

    where:
      - :math:`c_1=(k_1 L)^2` and :math:`c_2=(k_2 L)^2` are two variables to
        stabilize the division with weak denominator.
      - :math:`L` is the dynamic range of the pixel-values (typically this is
        :math:`2^{\#\text{bits per pixel}}-1`).

    the loss, or the Structural dissimilarity (DSSIM) can be finally described
    as:

    .. math::

      \text{loss}(x, y) = \frac{1 - \text{SSIM}(x, y)}{2}

    Arguments:
        window_size (int): the size of the kernel.
        max_val (float): the dynamic range of the images. Default: 1.
        reduction (str, optional): Specifies the reduction to apply to the
         output: 'none' | 'mean' | 'sum'. 'none': no reduction will be applied,
         'mean': the sum of the output will be divided by the number of elements
         in the output, 'sum': the output will be summed. Default: 'none'.

    Returns:
        Tensor: the ssim index.

    Shape:
        - Input: :math:`(B, C, H, W)`
        - Target :math:`(B, C, H, W)`
        - Output: scale, if reduction is 'none', then :math:`(B, C, H, W)`

    Examples::

        >>> input1 = torch.rand(1, 4, 5, 5)
        >>> input2 = torch.rand(1, 4, 5, 5)
        >>> ssim = kornia.losses.SSIM(5, reduction='none')
        >>> loss = ssim(input1, input2)  # 1x4x5x5
    """

    def __init__(
            self,
            window_size: int,
            reduction: str = "none",
            max_val: float = 1.0) -> None:
        super(SSIM, self).__init__(window_size, reduction, max_val)
        kornia.deprecation_warning("kornia.losses.SSIM", "kornia.nn.losses.SSIM")


######################
# functional interface
######################


def ssim(
        img1: torch.Tensor,
        img2: torch.Tensor,
        window_size: int,
        reduction: str = "none",
        max_val: float = 1.0) -> torch.Tensor:
    r"""Function that measures the Structural Similarity (SSIM) index between
    each element in the input `x` and target `y`.

    See :class:`~kornia.losses.SSIM` for details.
    """
    return kornia.nn.SSIM(window_size, reduction, max_val)(img1, img2)
