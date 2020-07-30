from typing import Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from kornia.filters.kernels import normalize_kernel2d
import kornia.testing as testing


def compute_padding(kernel_size: List[int]) -> List[int]:
    """Computes padding tuple."""
    # 4 or 6 ints:  (padding_left, padding_right,padding_top,padding_bottom)
    # https://pytorch.org/docs/stable/nn.html#torch.nn.functional.pad
    assert len(kernel_size) >= 2, kernel_size
    computed = [k // 2 for k in kernel_size]

    # for even kernels we need to do asymetric padding :(

    out_padding = []

    for i in range(len(kernel_size)):
        computed_tmp = computed[-(i + 1)]
        if kernel_size[i] % 2 == 0:
            padding = computed_tmp - 1
        else:
            padding = computed_tmp
        out_padding.append(padding)
        out_padding.append(computed_tmp)
    return out_padding


def filter2D(input: torch.Tensor, kernel: torch.Tensor,
             border_type: str = 'reflect',
             normalized: bool = False) -> torch.Tensor:
    r"""Function that convolves a tensor with a 2d kernel.

    The function applies a given kernel to a tensor. The kernel is applied
    independently at each depth channel of the tensor. Before applying the
    kernel, the function applies padding according to the specified mode so
    that the output remains in the same shape.

    Args:
        input (torch.Tensor): the input tensor with shape of
          :math:`(B, C, H, W)`.
        kernel (torch.Tensor): the kernel to be convolved with the input
          tensor. The kernel shape must be :math:`(1, kH, kW)`.
        border_type (str): the padding mode to be applied before convolving.
          The expected modes are: ``'constant'``, ``'reflect'``,
          ``'replicate'`` or ``'circular'``. Default: ``'reflect'``.
        normalized (bool): If True, kernel will be L1 normalized.

    Return:
        torch.Tensor: the convolved tensor of same size and numbers of channels
        as the input with shape :math:`(B, C, H, W)`.

    Example:
        >>> input = torch.tensor([[[
        ...    [0., 0., 0., 0., 0.],
        ...    [0., 0., 0., 0., 0.],
        ...    [0., 0., 5., 0., 0.],
        ...    [0., 0., 0., 0., 0.],
        ...    [0., 0., 0., 0., 0.],]]])
        >>> kernel = torch.ones(1, 3, 3)
        >>> kornia.filter2D(input, kernel)
        torch.tensor([[[[0., 0., 0., 0., 0.]
                        [0., 5., 5., 5., 0.]
                        [0., 5., 5., 5., 0.]
                        [0., 5., 5., 5., 0.]
                        [0., 0., 0., 0., 0.]]]])
    """
    testing.check_is_tensor(input)
    testing.check_is_tensor(kernel)

    if not isinstance(border_type, str):
        raise TypeError("Input border_type is not string. Got {}"
                        .format(type(kernel)))

    if not len(input.shape) == 4:
        raise ValueError("Invalid input shape, we expect BxCxHxW. Got: {}"
                         .format(input.shape))

    if not len(kernel.shape) == 3 and kernel.shape[0] != 1:
        raise ValueError("Invalid kernel shape, we expect 1xHxW. Got: {}"
                         .format(kernel.shape))

    # prepare kernel
    b, c, h, w = input.shape
    tmp_kernel: torch.Tensor = kernel[None].type_as(input)

    if normalized:
        tmp_kernel = normalize_kernel2d(tmp_kernel)

    # pad the input tensor
    height, width = tmp_kernel.shape[-2:]
    padding_shape: List[int] = compute_padding([height, width])
    input_pad: torch.Tensor = F.pad(input, padding_shape, mode=border_type)

    # convolve the tensor with the kernel. Pick the fastest alg
    kernel_numel: int = height * width
    if kernel_numel > 81:
        b, c, hp, wp = input_pad.shape
        return F.conv2d(input_pad.reshape(b * c, 1, hp, wp), tmp_kernel, padding=0, stride=1).view(b, c, h, w)

    return F.conv2d(input_pad, tmp_kernel.expand(c, -1, -1, -1), groups=c, padding=0, stride=1)


def filter3D(input: torch.Tensor, kernel: torch.Tensor,
             border_type: str = 'replicate',
             normalized: bool = False) -> torch.Tensor:
    r"""Function that convolves a tensor with a 3d kernel.

    The function applies a given kernel to a tensor. The kernel is applied
    independently at each depth channel of the tensor. Before applying the
    kernel, the function applies padding according to the specified mode so
    that the output remains in the same shape.

    Args:
        input (torch.Tensor): the input tensor with shape of
          :math:`(B, C, D, H, W)`.
        kernel (torch.Tensor): the kernel to be convolved with the input
          tensor. The kernel shape must be :math:`(1, kD, kH, kW)`.
        border_type (str): the padding mode to be applied before convolving.
          The expected modes are: ``'constant'``,
          ``'replicate'`` or ``'circular'``. Default: ``'replicate'``.
        normalized (bool): If True, kernel will be L1 normalized.

    Return:
        torch.Tensor: the convolved tensor of same size and numbers of channels
        as the input with shape :math:`(B, C, D, H, W)`.

    Example:
        >>> input = torch.tensor([[[
        ...    [[0., 0., 0., 0., 0.],
        ...     [0., 0., 0., 0., 0.],
        ...     [0., 0., 0., 0., 0.],
        ...     [0., 0., 0., 0., 0.],
        ...     [0., 0., 0., 0., 0.]],
        ...
        ...    [[0., 0., 0., 0., 0.],
        ...     [0., 0., 0., 0., 0.],
        ...     [0., 0., 5., 0., 0.],
        ...     [0., 0., 0., 0., 0.],
        ...     [0., 0., 0., 0., 0.]],
        ...
        ...    [[0., 0., 0., 0., 0.],
        ...     [0., 0., 0., 0., 0.],
        ...     [0., 0., 0., 0., 0.],
        ...     [0., 0., 0., 0., 0.],
        ...     [0., 0., 0., 0., 0.]]]])
        >>> kernel = torch.ones(1, 3, 3, 3)
        >>> kornia.filter3D(input, kernel)
        torch.tensor([[[[[0., 0., 0., 0., 0.],
                         [0., 5., 5., 5., 0.],
                         [0., 5., 5., 5., 0.],
                         [0., 5., 5., 5., 0.],
                         [0., 0., 0., 0., 0.]],
                        [[0., 0., 0., 0., 0.],
                         [0., 5., 5., 5., 0.],
                         [0., 5., 5., 5., 0.],
                         [0., 5., 5., 5., 0.],
                         [0., 0., 0., 0., 0.]],
                        [[0., 0., 0., 0., 0.],
                         [0., 5., 5., 5., 0.],
                         [0., 5., 5., 5., 0.],
                         [0., 5., 5., 5., 0.],
                         [0., 0., 0., 0., 0.]]]])
    """
    testing.check_is_tensor(input)
    testing.check_is_tensor(kernel)

    if not isinstance(border_type, str):
        raise TypeError("Input border_type is not string. Got {}"
                        .format(type(kernel)))

    if not len(input.shape) == 5:
        raise ValueError("Invalid input shape, we expect BxCxDxHxW. Got: {}"
                         .format(input.shape))

    if not len(kernel.shape) == 4 and kernel.shape[0] != 1:
        raise ValueError("Invalid kernel shape, we expect 1xDxHxW. Got: {}"
                         .format(kernel.shape))

    # prepare kernel
    b, c, d, h, w = input.shape
    tmp_kernel: torch.Tensor = kernel[None].type_as(input)

    if normalized:
        bk, dk, hk, wk = kernel.shape
        tmp_kernel = normalize_kernel2d(tmp_kernel.view(
            bk, dk, hk * wk)).view_as(kernel)

    # pad the input tensor
    depth, height, width = tmp_kernel.shape[-3:]
    padding_shape: List[int] = compute_padding([depth, height, width])
    input_pad: torch.Tensor = F.pad(input, padding_shape, mode=border_type)

    # convolve the tensor with the kernel.
    return F.conv3d(input_pad, tmp_kernel.expand(c, -1, -1, -1, -1), groups=c, padding=0, stride=1)
