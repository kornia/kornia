import torch
import torch.nn as nn
import torch.nn.functional as F

from kornia.morphology.basic_operators import dilation, erosion


# open
def open(tensor: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    r"""Returns the opened image, (that means, dilation after an erosion) applying the same kernel in each channel.

    The kernel must have 2 dimensions, each one defined by an odd number.

    Args:
       tensor (torch.Tensor): Image with shape :math:`(B, C, H, W)`.
       kernel (torch.Tensor): Structuring element with shape :math:`(k_x, k_y)`.

    Returns:
       torch.Tensor: Dilated image with shape :math:`(B, C, H, W)`.

    Example:
        >>> tensor = torch.rand(1, 3, 5, 5)
        >>> kernel = torch.ones(3, 3)
        >>> opened_img = open(tensor, kernel)
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

    return dilation(erosion(tensor, kernel), kernel)


# close
def close(tensor: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    r"""Returns the closed image, (that means, erosion after a dilation) applying the same kernel in each channel.

    The kernel must have 2 dimensions, each one defined by an odd number.

    Args:
       tensor (torch.Tensor): Image with shape :math:`(B, C, H, W)`.
       kernel (torch.Tensor): Structuring element with shape :math:`(k_x, k_y)`.

    Returns:
       torch.Tensor: Dilated image with shape :math:`(B, C, H, W)`.

    Example:
        >>> tensor = torch.rand(1, 3, 5, 5)
        >>> kernel = torch.ones(3, 3)
        >>> closed_img = close(tensor, kernel)
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

    return erosion(dilation(tensor, kernel), kernel)
