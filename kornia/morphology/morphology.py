import torch
import torch.nn as nn
import torch.nn.functional as F

from kornia.morphology.basic_operators import dilation, erosion
from kornia.morphology.open_close import open, close


# morphological gradient
def gradient(tensor: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    r"""Returns the morphological gradient of an image.

    That means, (dilation - erosion) applying the same kernel in each channel.
    The kernel must have 2 dimensions, each one defined by an odd number.

    Args:
       tensor (torch.Tensor): Image with shape :math:`(B, C, H, W)`.
       kernel (torch.Tensor): Structuring element with shape :math:`(k_x, k_y)`.

    Returns:
       torch.Tensor: Dilated image with shape :math:`(B, C, H, W)`.

    Example:
        >>> tensor = torch.rand(1, 3, 5, 5)
        >>> kernel = torch.ones(3, 3)
        >>> gradient_img = gradient(tensor, kernel)
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

    return dilation(tensor, kernel) - erosion(tensor, kernel)


# top_hat
def top_hat(tensor: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    r"""Returns the top hat tranformation of an image.

    That means, (image - opened_image) applying the same kernel in each channel.
    The kernel must have 2 dimensions, each one defined by an odd number.

    See :class:`~kornia.morphology.open` for details.

    Args:
       tensor (torch.Tensor): Image with shape :math:`(B, C, H, W)`.
       kernel (torch.Tensor): Structuring element with shape :math:`(k_x, k_y)`.

    Returns:
       torch.Tensor: Top hat transformated image with shape :math:`(B, C, H, W)`.

    Example:
        >>> tensor = torch.rand(1, 3, 5, 5)
        >>> kernel = torch.ones(3, 3)
        >>> top_hat_img = top_hat(tensor, kernel)
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

    return tensor - open(tensor, kernel)


# black_hat
def black_hat(tensor: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    r"""Returns the black hat tranformation of an image.

    That means, (closed_image - image) applying the same kernel in each channel.
    The kernel must have 2 dimensions, each one defined by an odd number.

    See :class:`~kornia.morphology.close` for details.

    Args:
       tensor (torch.Tensor): Image with shape :math:`(B, C, H, W)`.
       kernel (torch.Tensor): Structuring element with shape :math:`(k_x, k_y)`.

    Returns:
       torch.Tensor: Top hat transformated image with shape :math:`(B, C, H, W)`.

    Example:
        >>> tensor = torch.rand(1, 3, 5, 5)
        >>> kernel = torch.ones(3, 3)
        >>> black_hat_img = black_hat(tensor, kernel)
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

    return close(tensor, kernel) - tensor
