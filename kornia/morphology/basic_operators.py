from typing import Tuple, Union, List

import torch
import torch.nn as nn
import torch.nn.functional as F


# _se_to_mask
def _se_to_mask(se: torch.Tensor) -> torch.Tensor:
    se_h, se_w = se.size()
    se_flat = se.view(-1)
    num_feats = se_h * se_w
    out = torch.zeros(num_feats, 1, se_h, se_w, dtype=se.dtype, device=se.device)
    for i in range(num_feats):
        y = i // se_w
        x = i % se_w
        out[i, 0, y, x] = (se_flat[i] >= 0).float()
    return out


def dilation(tensor: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    r"""Returns the dilated image applying the same kernel in each channel.

    The kernel must have 2 dimensions, each one defined by an odd number.
    Dilation is equivalent to eroding the background thus dilation(x, k) == -erosion(-x, k).

    Args:
       tensor (torch.Tensor): Image with shape :math:`(B, C, H, W)`.
       kernel (torch.Tensor): Structuring element with shape :math:`(k_x, k_y)`.

    Returns:
       torch.Tensor: Dilated image with shape :math:`(B, C, H, W)`.

    Example:
        >>> tensor = torch.rand(1, 3, 5, 5)
        >>> kernel = torch.ones(3, 3)
        >>> dilated_img = dilation(tensor, kernel)
    """
    return -erosion(-tensor, kernel)


# erosion
def erosion(tensor: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    r"""Returns the eroded image applying the same kernel in each channel.

    The kernel must have 2 dimensions, each one defined by an odd number.

    Args:
       tensor (torch.Tensor): Image with shape :math:`(B, C, H, W)`.
       kernel (torch.Tensor): Structuring element with shape :math:`(k_x, k_y)`.

    Returns:
       torch.Tensor: Eroded image with shape :math:`(B, C, H, W)`.

    Example:
        >>> tensor = torch.rand(1, 3, 5, 5)
        >>> kernel = torch.ones(5, 5)
        >>> output = erosion(tensor, kernel)
    """

    if not isinstance(tensor, torch.Tensor):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(
            type(tensor)))

    if len(tensor.shape) != 4:
        raise ValueError("Input size must have 4 dimensions. Got {}".format(
            tensor.dim()))

    if not isinstance(kernel, torch.Tensor):
        raise TypeError("Kernel type is not a torch.Tensor. Got {}".format(
            type(kernel)))

    if len(kernel.shape) != 2:
        raise ValueError("Kernel size must have 2 dimensions. Got {}".format(
            kernel.dim()))

    # prepare kernel
    se_e: torch.Tensor = kernel - 1.
    kernel_e: torch.Tensor = _se_to_mask(se_e)

    # pad
    se_h, se_w = kernel.shape
    pad_e: List[int] = [se_w // 2, se_w // 2, se_h // 2, se_h // 2]

    output: torch.Tensor = tensor.view(
        tensor.shape[0] * tensor.shape[1], 1, tensor.shape[2], tensor.shape[3])
    output = F.pad(output, pad_e, mode='constant', value=1.)
    output = F.conv2d(output, kernel_e) - se_e.view(1, -1, 1, 1)
    # TODO: upgrade to: `output = torch.amin(output, dim=1)` after dropping pytorch 1.6 support
    output = torch.min(output, dim=1)[0]

    return output.view_as(tensor)
