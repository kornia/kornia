from typing import List

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

import kornia
from .kernels import get_pascal_kernel_2d
from .median import _compute_zero_padding


class BlurPool2D(nn.Module):
    r"""Compute blurs and downsample a given feature map.

    See :cite:`zhang2019shiftinvar` for more details.

    Args:
        kernel_size (int): the kernel size for max pooling.
        stride (int): stride for pooling.

    Shape:
        - Input: :math:`(B, C, H, W)`
        - Output: :math:`(N, C, H_{out}, W_{out})`, where

          .. math::
              H_{out} = \left\lfloor\frac{H_{in}  + 2 \times \text{kernel\_size//2}[0] -
                \text{kernel\_size}[0]}{\text{stride}[0]} + 1\right\rfloor

          .. math::
              W_{out} = \left\lfloor\frac{W_{in}  + 2 \times \text{kernel\_size//2}[1] -
                \text{kernel\_size}[1]}{\text{stride}[1]} + 1\right\rfloor

    Returns:
        torch.Tensor: the transformed tensor.

    Examples:
    >>> input = torch.eye(5)[None, None]
    >>> bp = BlurPool2D(kernel_size=3, stride=2)
    >>> bp(input)
    tensor([[[[0.3125, 0.0625, 0.0000],
              [0.0625, 0.3750, 0.0625],
              [0.0000, 0.0625, 0.3125]]]])
    """
    def __init__(self, kernel_size: int, stride: int = 2):
        super(BlurPool2D, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return blur_pool2d(input, self.kernel_size, self.stride)


class MaxBlurPool2D(nn.Module):
    r"""Compute pools and blurs and downsample a given feature map.

    Equivalent to ```nn.Sequential(nn.MaxPool2d(...), BlurPool2D(...))```
    See :cite:`zhang2019shiftinvar` for more details.

    Args:
        kernel_size (int): the kernel size for max pooling.
        stride (int): stride for pooling.
        max_pool_size (int): the kernel size for max pooling.
        ceil_mode (bool): should be true to match output size of conv2d with same kernel size.

    Shape:
        - Input: :math:`(B, C, H, W)`
        - Output: :math:`(B, C, H / stride, W / stride)`

    Returns:
        torch.Tensor: the transformed tensor.

    Examples:
    >>> input = torch.eye(5)[None, None]
    >>> mbp = MaxBlurPool2D(kernel_size=3, stride=2, max_pool_size=2, ceil_mode=False)
    >>> mbp(input)
    tensor([[[[0.5625, 0.3125],
              [0.3125, 0.8750]]]])
    >>> seq = nn.Sequential(nn.MaxPool2d(kernel_size=2, stride=1), BlurPool2D(kernel_size=3, stride=2))
    >>> seq(input)
    tensor([[[[0.5625, 0.3125],
              [0.3125, 0.8750]]]])
    """
    def __init__(self, kernel_size: int, stride: int = 2, max_pool_size: int = 2, ceil_mode: bool = False):
        super(MaxBlurPool2D, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.max_pool_size = max_pool_size
        self.ceil_mode = ceil_mode

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return max_blur_pool2d(input, self.kernel_size, self.stride, self.max_pool_size, self.ceil_mode)


def blur_pool2d(input: torch.Tensor, kernel_size: int, stride: int = 2):
    r"""Compute blurs and downsample a given feature map.

    See :cite:`zhang2019shiftinvar` for more details.

    Args:
        kernel_size (int): the kernel size for max pooling..
        ceil_mode (bool): should be true to match output size of conv2d with same kernel size.

    Shape:
        - Input: :math:`(B, C, H, W)`
        - Output: :math:`(N, C, H_{out}, W_{out})`, where

          .. math::
              H_{out} = \left\lfloor\frac{H_{in}  + 2 \times \text{kernel\_size//2}[0] -
                \text{kernel\_size}[0]}{\text{stride}[0]} + 1\right\rfloor

          .. math::
              W_{out} = \left\lfloor\frac{W_{in}  + 2 \times \text{kernel\_size//2}[1] -
                \text{kernel\_size}[1]}{\text{stride}[1]} + 1\right\rfloor

    Returns:
        torch.Tensor: the transformed tensor.

    Note:
        This function is tested against https://github.com/adobe/antialiased-cnns.

    Examples:
    >>> input = torch.eye(5)[None, None]
    >>> blur_pool2d(input, 3)
    tensor([[[[0.3125, 0.0625, 0.0000],
              [0.0625, 0.3750, 0.0625],
              [0.0000, 0.0625, 0.3125]]]])
    """
    padding: Tuple[int, int] = _compute_zero_padding((kernel_size, kernel_size))
    kernel = get_pascal_kernel_2d(kernel_size, norm=True).repeat((input.size(1), 1, 1, 1)).to(input)
    return F.conv2d(input, kernel, padding=padding, stride=stride, groups=input.size(1))


def max_blur_pool2d(
    input: torch.Tensor, kernel_size: int, stride: int = 2, max_pool_size: int = 2, ceil_mode: bool = False
) -> torch.Tensor:
    r"""Compute pools and blurs and downsample a given feature map.

    See :class:`~kornia.filters.MaxBlurPool2d` for details.

    Note:
        This function is tested against https://github.com/adobe/antialiased-cnns.

    Examples:
    >>> input = torch.eye(5)[None, None]
    >>> max_blur_pool2d(input, 3)
    tensor([[[[0.5625, 0.3125],
              [0.3125, 0.8750]]]])
    """
    if not len(input.shape) == 4:
        raise ValueError(f"Invalid input shape, we expect BxCxHxW. Got: {input.shape}")
    padding: Tuple[int, int] = _compute_zero_padding((kernel_size, kernel_size))
    # compute local maxima
    x_max: torch.Tensor = F.max_pool2d(
        input, kernel_size=max_pool_size,
        padding=0, stride=1, ceil_mode=ceil_mode)

    # blur and downsample
    kernel = get_pascal_kernel_2d(kernel_size, norm=True).repeat((input.size(1), 1, 1, 1)).to(input)
    x_down = F.conv2d(x_max, kernel, padding=padding, stride=stride, groups=x_max.size(1))
    return x_down
