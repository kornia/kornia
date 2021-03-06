from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from kornia.geometry.transform.pyramid import pyrdown
from .median import _compute_zero_padding

__all__ = [
    "max_blur_pool2d",
    "MaxBlurPool2d",
]


class MaxBlurPool2d(nn.Module):
    r"""Creates a module that computes pools and blurs and downsample a given
    feature map.

    See :cite:`zhang2019shiftinvar` for more details.

    Args:
        kernel_size (int): the kernel size for max pooling..
        ceil_mode (bool): should be true to match output size of conv2d with same kernel size.

    Shape:
        - Input: :math:`(B, C, H, W)`
        - Output: :math:`(B, C, H / 2, W / 2)`

    Returns:
        torch.Tensor: the transformed tensor.

    Examples:
        >>> input = torch.rand(1, 4, 4, 8)
        >>> pool = MaxBlurPool2d(kernel_size=3)
        >>> output = pool(input)  # 1x4x2x4
    """

    def __init__(self, kernel_size: int, ceil_mode: bool = False) -> None:
        super(MaxBlurPool2d, self).__init__()
        self.ceil_mode: bool = ceil_mode
        self.kernel_size: Tuple[int, int] = (kernel_size, kernel_size)

    def forward(self, input: torch.Tensor) -> torch.Tensor:  # type: ignore
        return max_blur_pool2d(input, self.kernel_size, self.ceil_mode)


def max_blur_pool2d(input: torch.Tensor, kernel_size: int, ceil_mode: bool = False) -> torch.Tensor:
    r"""Creates a module that computes pools and blurs and downsample a given
    feature map.

    See :class:`~kornia.contrib.MaxBlurPool2d` for details.
    """
    if not len(input.shape) == 4:
        raise ValueError(f"Invalid input shape, we expect BxCxHxW. Got: {input.shape}")
    padding: Tuple[int, int] = _compute_zero_padding(kernel_size)
    # compute local maxima
    x_max: torch.Tensor = F.max_pool2d(
        input, kernel_size=kernel_size,
        padding=padding, stride=1, ceil_mode=ceil_mode)

    # blur and downsample
    x_down: torch.Tensor = pyrdown(x_max)
    return x_down
