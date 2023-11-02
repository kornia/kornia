import math

import torch
from torch import nn

from kornia.filters import filter2d
from kornia.utils import create_meshgrid


def distance_transform(image: torch.Tensor, kernel_size: int = 3, h: float = 0.35) -> torch.Tensor:
    r"""Approximates the Manhattan distance transform of images using cascaded convolution operations.

    The value at each pixel in the output represents the distance to the nearest non-zero pixel in the image image.
    It uses the method described in :cite:`pham2021dtlayer`.
    The transformation is applied independently across the channel dimension of the images.

    Args:
        image: Image with shape :math:`(B,C,H,W)`.
        kernel_size: size of the convolution kernel.
        h: value that influence the approximation of the min function.

    Returns:
        tensor with shape :math:`(B,C,H,W)`.

    Example:
        >>> tensor = torch.zeros(1, 1, 5, 5)
        >>> tensor[:,:, 1, 2] = 1
        >>> dt = kornia.contrib.distance_transform(tensor)
    """
    if not isinstance(image, torch.Tensor):
        raise TypeError(f"image type is not a torch.Tensor. Got {type(image)}")

    if not len(image.shape) == 4:
        raise ValueError(f"Invalid image shape, we expect BxCxHxW. Got: {image.shape}")

    if kernel_size % 2 == 0:
        raise ValueError("Kernel size must be an odd number.")

    # n_iters is set such that the DT will be able to propagate from any corner of the image to its far,
    # diagonally opposite corner
    n_iters: int = math.ceil(max(image.shape[2], image.shape[3]) / math.floor(kernel_size / 2))
    grid = create_meshgrid(
        kernel_size, kernel_size, normalized_coordinates=False, device=image.device, dtype=image.dtype
    )

    grid -= math.floor(kernel_size / 2)
    kernel = torch.hypot(grid[0, :, :, 0], grid[0, :, :, 1])
    kernel = torch.exp(kernel / -h).unsqueeze(0)

    out = torch.zeros_like(image)

    # It is possible to avoid cloning the image if boundary = image, but this would require modifying the image tensor.
    boundary = image.clone()
    signal_ones = torch.ones_like(boundary)

    for i in range(n_iters):
        cdt = filter2d(boundary, kernel, border_type="replicate")
        cdt = -h * torch.log(cdt)

        # We are calculating log(0) above.
        cdt = torch.nan_to_num(cdt, posinf=0.0)

        mask = torch.where(cdt > 0, 1.0, 0.0)
        if mask.sum() == 0:
            break

        offset: int = i * kernel_size // 2
        out += (offset + cdt) * mask
        boundary = torch.where(mask == 1, signal_ones, boundary)

    return out


class DistanceTransform(nn.Module):
    r"""Module that approximates the Manhattan (city block) distance transform of images using convolutions.

    Args:
        kernel_size: size of the convolution kernel.
        h: value that influence the approximation of the min function.
    """

    def __init__(self, kernel_size: int = 3, h: float = 0.35) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.h = h

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        # If images have multiple channels, view the channels in the batch dimension to match kernel shape.
        if image.shape[1] > 1:
            image_in = image.view(-1, 1, image.shape[-2], image.shape[-1])
        else:
            image_in = image

        return distance_transform(image_in, self.kernel_size, self.h).view_as(image)
