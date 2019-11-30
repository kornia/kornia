from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from kornia.geometry.transform.pyramid import pyrdown

__all__ = [
    "max_blur_pool2d",
    "MaxBlurPool2d",
]


def _compute_zero_padding(kernel_size: Tuple[int, int]) -> Tuple[int, int]:
    """Computes zero padding tuple."""
    padding = [(k - 1) // 2 for k in kernel_size]
    return padding[0], padding[1]


class MaxBlurPool2d(nn.Module):
    r"""Creates a module that computes pools and blurs and downsample a given
    feature map.

    See :cite:`zhang2019shiftinvar` for more details.

    Args:
        kernel_size (int): the kernel size for max pooling..

    Shape:
        - Input: :math:`(B, C, H, W)`
        - Output: :math:`(B, C, H / 2, W / 2)`

    Returns:
        torch.Tensor: the transformed tensor.

    Examples:
        >>> input = torch.rand(1, 4, 4, 8)
        >>> pool = kornia.contrib.MaxblurPool2d(kernel_size=3)
        >>> output = pool(input)  # 1x4x2x4
    """

    def __init__(self, kernel_size: int) -> None:
        super(MaxBlurPool2d, self).__init__()
        self.kernel_size: Tuple[int, int] = (kernel_size, kernel_size)
        self.padding: Tuple[int, int] = _compute_zero_padding(self.kernel_size)

    def forward(self, input: torch.Tensor) -> torch.Tensor:  # type: ignore
        if not torch.is_tensor(input):
            raise TypeError("Input input type is not a torch.Tensor. Got {}"
                            .format(type(input)))
        if not len(input.shape) == 4:
            raise ValueError("Invalid input shape, we expect BxCxHxW. Got: {}"
                             .format(input.shape))
        # compute local maxima
        x_max: torch.Tensor = F.max_pool2d(
            input, kernel_size=self.kernel_size,
            padding=self.padding, stride=1)

        # blur and downsample
        x_down: torch.Tensor = pyrdown(x_max)
        return x_down


######################
# functional interface
######################


def max_blur_pool2d(input: torch.Tensor, kernel_size: int) -> torch.Tensor:
    r"""Creates a module that computes pools and blurs and downsample a given
    feature map.

    See :class:`~kornia.contrib.MaxBlurPool2d` for details.
    """
    return MaxBlurPool2d(kernel_size)(input)
