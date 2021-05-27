import torch
import torch.nn.functional as F

from typing import List, Optional


# Dilation
def dilation(tensor: torch.Tensor, kernel: torch.Tensor, origin: Optional[List[int]] = None) -> torch.Tensor:
    r"""Returns the dilated image applying the same kernel in each channel.

    The kernel must have 2 dimensions.

    Args:
        tensor (torch.Tensor): Image with shape :math:`(B, C, H, W)`.
        kernel (torch.Tensor): Structuring element with shape :math:`(k_x, k_y)`.
        origin (List[int], Tuple[int, int]): Origin of the structuring element. Default is None and uses the center of
        the structuring element as origin (rounding towards zero).

    Returns:
        torch.Tensor: Dilated image with shape :math:`(B, C, H, W)`.

    Example:
        >>> tensor = torch.rand(1, 3, 5, 5)
        >>> kernel = torch.ones(3, 3)
        >>> dilated_img = dilation(tensor, kernel)
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

    # origin
    se_h, se_w = kernel.shape
    if origin is None:
        origin = [se_h // 2, se_w // 2]

    # pad
    pad_e: List[int] = [origin[1], se_w - origin[1] - 1, origin[0], se_h - origin[0] - 1]
    output: torch.Tensor = F.pad(tensor, pad_e, mode='constant', value=0.)

    # computation
    output = output.unfold(2, se_h, 1).unfold(3, se_w, 1)
    output, _ = torch.max(output + kernel.flip((0, 1)), 4)
    output, _ = torch.max(output, 4)

    return output


# Erosion
def erosion(tensor: torch.Tensor, kernel: torch.Tensor, origin: Optional[List[int]] = None) -> torch.Tensor:
    r"""Returns the eroded image applying the same kernel in each channel.

    The kernel must have 2 dimensions.

    Args:
        tensor (torch.Tensor): Image with shape :math:`(B, C, H, W)`.
        kernel (torch.Tensor): Structuring element with shape :math:`(k_x, k_y)`.
        origin (List[int], Tuple[int, int]): Origin of the structuring element. Default is None and uses the center of
        the structuring element as origin (rounding towards zero).

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

    # origin
    se_h, se_w = kernel.shape
    if origin is None:
        origin = [se_h // 2, se_w // 2]

    # pad
    pad_e: List[int] = [origin[1], se_w - origin[1] - 1, origin[0], se_h - origin[0] - 1]
    output: torch.Tensor = F.pad(tensor, pad_e, mode='constant', value=1.)

    # computation
    output = output.unfold(2, se_h, 1).unfold(3, se_w, 1)
    output, _ = torch.min(output - kernel, 4)
    output, _ = torch.min(output, 4)

    return output


# Opening
def opening(tensor: torch.Tensor, kernel: torch.Tensor, origin: Optional[List[int]] = None) -> torch.Tensor:
    r"""Returns the opened image, (that means, dilation after an erosion) applying the same kernel in each channel.

    The kernel must have 2 dimensions.

    Args:
        tensor (torch.Tensor): Image with shape :math:`(B, C, H, W)`.
        kernel (torch.Tensor): Structuring element with shape :math:`(k_x, k_y)`.
        origin (List[int], Tuple[int, int]): Origin of the structuring element. Default is None and uses the center of
        the structuring element as origin (rounding towards zero).

    Returns:
       torch.Tensor: Opened image with shape :math:`(B, C, H, W)`.

    Example:
        >>> tensor = torch.rand(1, 3, 5, 5)
        >>> kernel = torch.ones(3, 3)
        >>> opened_img = opening(tensor, kernel)
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

    return dilation(erosion(tensor, kernel, origin=origin), kernel, origin=origin)


# Closing
def closing(tensor: torch.Tensor, kernel: torch.Tensor, origin: Optional[List[int]] = None) -> torch.Tensor:
    r"""Returns the closed image, (that means, erosion after a dilation) applying the same kernel in each channel.

    The kernel must have 2 dimensions.

    Args:
        tensor (torch.Tensor): Image with shape :math:`(B, C, H, W)`.
        kernel (torch.Tensor): Structuring element with shape :math:`(k_x, k_y)`.
        origin (List[int], Tuple[int, int]): Origin of the structuring element. Default is None and uses the center of
        the structuring element as origin (rounding towards zero).

    Returns:
       torch.Tensor: Closed image with shape :math:`(B, C, H, W)`.

    Example:
        >>> tensor = torch.rand(1, 3, 5, 5)
        >>> kernel = torch.ones(3, 3)
        >>> closed_img = closing(tensor, kernel)
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

    return erosion(dilation(tensor, kernel, origin=origin), kernel, origin=origin)


# Morphological Gradient
def gradient(tensor: torch.Tensor, kernel: torch.Tensor, origin: Optional[List[int]] = None) -> torch.Tensor:
    r"""Returns the morphological gradient of an image.

    That means, (dilation - erosion) applying the same kernel in each channel.
    The kernel must have 2 dimensions.

    Args:
        tensor (torch.Tensor): Image with shape :math:`(B, C, H, W)`.
        kernel (torch.Tensor): Structuring element with shape :math:`(k_x, k_y)`.
        origin (List[int], Tuple[int, int]): Origin of the structuring element. Default is None and uses the center of
        the structuring element as origin (rounding towards zero).

    Returns:
       torch.Tensor: Gradient image with shape :math:`(B, C, H, W)`.

    Example:
        >>> tensor = torch.rand(1, 3, 5, 5)
        >>> kernel = torch.ones(3, 3)
        >>> gradient_img = gradient(tensor, kernel)
    """

    return dilation(tensor, kernel, origin=origin) - erosion(tensor, kernel, origin=origin)


# Top Hat
def top_hat(tensor: torch.Tensor, kernel: torch.Tensor, origin: Optional[List[int]] = None) -> torch.Tensor:
    r"""Returns the top hat tranformation of an image.

    That means, (image - opened_image) applying the same kernel in each channel.
    The kernel must have 2 dimensions.

    See :opening for details.

    Args:
        tensor (torch.Tensor): Image with shape :math:`(B, C, H, W)`.
        kernel (torch.Tensor): Structuring element with shape :math:`(k_x, k_y)`.
        origin (List[int], Tuple[int, int]): Origin of the structuring element. Default is None and uses the center of
        the structuring element as origin (rounding towards zero).

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

    return tensor - opening(tensor, kernel, origin=origin)


# Bottom Hat
def bottom_hat(tensor: torch.Tensor, kernel: torch.Tensor, origin: Optional[List[int]] = None) -> torch.Tensor:
    r"""Returns the bottom hat tranformation of an image.

    That means, (closed_image - image) applying the same kernel in each channel.
    The kernel must have 2 dimensions.

    See :closing for details.

    Args:
        tensor (torch.Tensor): Image with shape :math:`(B, C, H, W)`.
        kernel (torch.Tensor): Structuring element with shape :math:`(k_x, k_y)`.
        origin (List[int], Tuple[int, int]): Origin of the structuring element. Default is None and uses the center of
        the structuring element as origin (rounding towards zero).

    Returns:
       torch.Tensor: Top hat transformated image with shape :math:`(B, C, H, W)`.

    Example:
        >>> tensor = torch.rand(1, 3, 5, 5)
        >>> kernel = torch.ones(3, 3)
        >>> bottom_hat_img = bottom_hat(tensor, kernel)
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

    return closing(tensor, kernel, origin=origin) - tensor
