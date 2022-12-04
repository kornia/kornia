from typing import Tuple

import torch
import torch.nn.functional as F

from kornia.core import Module, Tensor
from kornia.testing import KORNIA_CHECK, KORNIA_CHECK_SHAPE

from .kernels import get_pascal_kernel_2d
from .median import _compute_zero_padding  # TODO: Move to proper place

__all__ = ["BlurPool2D", "MaxBlurPool2D", "blur_pool2d", "max_blur_pool2d", 'edge_aware_blur_pool2d']


class BlurPool2D(Module):
    r"""Compute blur (anti-aliasing) and downsample a given feature map.

    See :cite:`zhang2019shiftinvar` for more details.

    Args:
        kernel_size: the kernel size for max pooling.
        stride: stride for pooling.

    Shape:
        - Input: :math:`(B, C, H, W)`
        - Output: :math:`(N, C, H_{out}, W_{out})`, where

          .. math::
              H_{out} = \left\lfloor\frac{H_{in}  + 2 \times \text{kernel\_size//2}[0] -
                \text{kernel\_size}[0]}{\text{stride}[0]} + 1\right\rfloor

          .. math::
              W_{out} = \left\lfloor\frac{W_{in}  + 2 \times \text{kernel\_size//2}[1] -
                \text{kernel\_size}[1]}{\text{stride}[1]} + 1\right\rfloor

    Examples:
        >>> from kornia.filters.blur_pool import BlurPool2D
        >>> input = torch.eye(5)[None, None]
        >>> bp = BlurPool2D(kernel_size=3, stride=2)
        >>> bp(input)
        tensor([[[[0.3125, 0.0625, 0.0000],
                  [0.0625, 0.3750, 0.0625],
                  [0.0000, 0.0625, 0.3125]]]])
    """

    def __init__(self, kernel_size: int, stride: int = 2):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.register_buffer('kernel', get_pascal_kernel_2d(kernel_size, norm=True))

    def forward(self, input: Tensor) -> Tensor:
        # To align the logic with the whole lib
        kernel = torch.as_tensor(self.kernel, device=input.device, dtype=input.dtype)
        return _blur_pool_by_kernel2d(input, kernel.repeat((input.shape[1], 1, 1, 1)), self.stride)


class MaxBlurPool2D(Module):
    r"""Compute pools and blurs and downsample a given feature map.

    Equivalent to ```nn.Sequential(nn.MaxPool2d(...), BlurPool2D(...))```

    See :cite:`zhang2019shiftinvar` for more details.

    Args:
        kernel_size: the kernel size for max pooling.
        stride: stride for pooling.
        max_pool_size: the kernel size for max pooling.
        ceil_mode: should be true to match output size of conv2d with same kernel size.

    Shape:
        - Input: :math:`(B, C, H, W)`
        - Output: :math:`(B, C, H / stride, W / stride)`

    Returns:
        torch.Tensor: the transformed tensor.

    Examples:
        >>> import torch.nn as nn
        >>> from kornia.filters.blur_pool import BlurPool2D
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
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.max_pool_size = max_pool_size
        self.ceil_mode = ceil_mode
        self.register_buffer('kernel', get_pascal_kernel_2d(kernel_size, norm=True))

    def forward(self, input: Tensor) -> Tensor:
        # To align the logic with the whole lib
        kernel = torch.as_tensor(self.kernel, device=input.device, dtype=input.dtype)
        return _max_blur_pool_by_kernel2d(
            input, kernel.repeat((input.size(1), 1, 1, 1)), self.stride, self.max_pool_size, self.ceil_mode
        )


def blur_pool2d(input: Tensor, kernel_size: int, stride: int = 2):
    r"""Compute blurs and downsample a given feature map.

    .. image:: _static/img/blur_pool2d.png

    See :class:`~kornia.filters.BlurPool2D` for details.

    See :cite:`zhang2019shiftinvar` for more details.

    Args:
        kernel_size: the kernel size for max pooling..
        ceil_mode: should be true to match output size of conv2d with same kernel size.

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
        the transformed tensor.

    .. note::
        This function is tested against https://github.com/adobe/antialiased-cnns.

    .. note::
       See a working example `here <https://kornia-tutorials.readthedocs.io/en/latest/
       filtering_operators.html>`__.

    Examples:
        >>> input = torch.eye(5)[None, None]
        >>> blur_pool2d(input, 3)
        tensor([[[[0.3125, 0.0625, 0.0000],
                  [0.0625, 0.3750, 0.0625],
                  [0.0000, 0.0625, 0.3125]]]])
    """
    kernel = get_pascal_kernel_2d(kernel_size, norm=True).repeat((input.size(1), 1, 1, 1)).to(input)
    return _blur_pool_by_kernel2d(input, kernel, stride)


def max_blur_pool2d(
    input: Tensor, kernel_size: int, stride: int = 2, max_pool_size: int = 2, ceil_mode: bool = False
) -> Tensor:
    r"""Compute pools and blurs and downsample a given feature map.

    .. image:: _static/img/max_blur_pool2d.png

    See :class:`~kornia.filters.MaxBlurPool2D` for details.

    Args:
        kernel_size: the kernel size for max pooling.
        stride: stride for pooling.
        max_pool_size: the kernel size for max pooling.
        ceil_mode: should be true to match output size of conv2d with same kernel size.

    .. note::
        This function is tested against https://github.com/adobe/antialiased-cnns.

    .. note::
       See a working example `here <https://kornia-tutorials.readthedocs.io/en/latest/
       filtering_operators.html>`__.

    Examples:
        >>> input = torch.eye(5)[None, None]
        >>> max_blur_pool2d(input, 3)
        tensor([[[[0.5625, 0.3125],
                  [0.3125, 0.8750]]]])
    """
    if not len(input.shape) == 4:
        raise ValueError(f"Invalid input shape, we expect BxCxHxW. Got: {input.shape}")
    kernel = get_pascal_kernel_2d(kernel_size, norm=True).repeat((input.shape[1], 1, 1, 1)).to(input)
    return _max_blur_pool_by_kernel2d(input, kernel, stride, max_pool_size, ceil_mode)


def _blur_pool_by_kernel2d(input: Tensor, kernel: Tensor, stride: int):
    """Compute blur_pool by a given :math:`CxC_{out}xNxN` kernel."""
    if not (len(kernel.shape) == 4 and kernel.shape[-1] == kernel.shape[-2]):
        raise AssertionError(f"Invalid kernel shape. Expect CxC_outxNxN, Got {kernel.shape}")
    padding: Tuple[int, int] = _compute_zero_padding((kernel.shape[-2], kernel.shape[-1]))
    return F.conv2d(input, kernel, padding=padding, stride=stride, groups=input.shape[1])


def _max_blur_pool_by_kernel2d(input: Tensor, kernel: Tensor, stride: int, max_pool_size: int, ceil_mode: bool):
    """Compute max_blur_pool by a given :math:`CxC_{out}xNxN` kernel."""
    if not (len(kernel.shape) == 4 and kernel.shape[-1] == kernel.shape[-2]):
        raise AssertionError(f"Invalid kernel shape. Expect CxC_outxNxN, Got {kernel.shape}")
    # compute local maxima
    input = F.max_pool2d(input, kernel_size=max_pool_size, padding=0, stride=1, ceil_mode=ceil_mode)
    # blur and downsample
    padding: Tuple[int, int] = _compute_zero_padding((kernel.shape[-2], kernel.shape[-1]))
    return F.conv2d(input, kernel, padding=padding, stride=stride, groups=input.size(1))


def edge_aware_blur_pool2d(
    input: Tensor,
    kernel_size: int,
    edge_threshold: float = 1.25,
    edge_dilation_kernel_size: int = 3,
    epsilon: float = 1e-6,
) -> Tensor:
    r"""Blur the input tensor while maintaining its edges.

    Args:
        input: the input image to blur with shape :math:`(B, C, H, W)`.
        kernel_size: the kernel size for max pooling.
        edge_threshold: positive threshold for the edge decision rule; edge/non-edge.
        edge_dilation_kernel_size: the kernel size for dilating the edges.
        epsilon: for numerical stability.

    Returns:
        The blurred tensor of shape :math:`(B, C, H, W)`.
    """
    KORNIA_CHECK_SHAPE(input, ["B", "C", "H", "W"])
    KORNIA_CHECK(edge_threshold > 0.0, f"edge threshold should be positive, but got '{edge_threshold}'")

    input = F.pad(input, (2, 2, 2, 2), mode="reflect")  # pad to avoid artifacts near physical edges
    blurred_input = blur_pool2d(input, kernel_size=kernel_size, stride=1)  # blurry version of the input

    # calculate the edges (add epsilon to avoid taking the log of 0)
    log_input, log_thresh = torch.log2(input + epsilon), torch.log2(torch.tensor(edge_threshold))
    edges_x = log_input[..., :, 4:] - log_input[..., :, :-4]
    edges_y = log_input[..., 4:, :] - log_input[..., :-4, :]
    edges_x, edges_y = torch.mean(edges_x, dim=-3, keepdim=True), torch.mean(edges_y, dim=-3, keepdim=True)
    edges_x_mask, edges_y_mask = edges_x.abs() > log_thresh.to(edges_x), edges_y.abs() > log_thresh.to(edges_y)
    edges_xy_mask = (edges_x_mask[..., 2:-2, :] + edges_y_mask[..., :, 2:-2]).type_as(input)

    # dilate the content edges to have a soft mask of edges
    dilated_edges = F.max_pool3d(edges_xy_mask, edge_dilation_kernel_size, 1, edge_dilation_kernel_size // 2)

    # slice the padded regions
    input = input[..., 2:-2, 2:-2]
    blurred_input = blurred_input[..., 2:-2, 2:-2]

    # fuse the input image on edges and blurry input everywhere else
    blurred = dilated_edges * input + (1.0 - dilated_edges) * blurred_input

    return blurred
