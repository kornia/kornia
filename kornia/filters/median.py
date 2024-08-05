from __future__ import annotations

import torch.nn.functional as F

from kornia.core import ImageModule as Module
from kornia.core import Tensor
from kornia.core.check import KORNIA_CHECK_IS_TENSOR, KORNIA_CHECK_SHAPE

from .kernels import _unpack_2d_ks, get_binary_kernel2d


def _compute_zero_padding(kernel_size: tuple[int, int] | int) -> tuple[int, int]:
    r"""Utility function that computes zero padding tuple."""
    ky, kx = _unpack_2d_ks(kernel_size)
    return (ky - 1) // 2, (kx - 1) // 2


def median_blur(input: Tensor, kernel_size: tuple[int, int] | int) -> Tensor:
    r"""Blur an image using the median filter.

    .. image:: _static/img/median_blur.png

    Args:
        input: the input image with shape :math:`(B,C,H,W)`.
        kernel_size: the blurring kernel size.

    Returns:
        the blurred input tensor with shape :math:`(B,C,H,W)`.

    .. note::
       See a working example `here <https://kornia.github.io/tutorials/nbs/filtering_operators.html>`__.

    Example:
        >>> input = torch.rand(2, 4, 5, 7)
        >>> output = median_blur(input, (3, 3))
        >>> output.shape
        torch.Size([2, 4, 5, 7])
    """
    KORNIA_CHECK_IS_TENSOR(input)
    KORNIA_CHECK_SHAPE(input, ["B", "C", "H", "W"])

    padding = _compute_zero_padding(kernel_size)

    # prepare kernel
    kernel: Tensor = get_binary_kernel2d(kernel_size, device=input.device, dtype=input.dtype)
    b, c, h, w = input.shape

    # map the local window to single vector
    features: Tensor = F.conv2d(input.reshape(b * c, 1, h, w), kernel, padding=padding, stride=1)
    features = features.view(b, c, -1, h, w)  # BxCx(K_h * K_w)xHxW

    # compute the median along the feature axis
    return features.median(dim=2)[0]


class MedianBlur(Module):
    r"""Blur an image using the median filter.

    Args:
        kernel_size: the blurring kernel size.

    Returns:
        the blurred input tensor.

    Shape:
        - Input: :math:`(B, C, H, W)`
        - Output: :math:`(B, C, H, W)`

    Example:
        >>> input = torch.rand(2, 4, 5, 7)
        >>> blur = MedianBlur((3, 3))
        >>> output = blur(input)
        >>> output.shape
        torch.Size([2, 4, 5, 7])
    """

    def __init__(self, kernel_size: tuple[int, int] | int) -> None:
        super().__init__()
        self.kernel_size = kernel_size

    def forward(self, input: Tensor) -> Tensor:
        return median_blur(input, self.kernel_size)
