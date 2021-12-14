import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import kornia


def make_cdt_kernel(
    kernel_size: int,
) -> torch.Tensor:
    # Value of h is derived from the parameters and reference code given by the authors who proposed the algorithm.
    h = -0.35

    grid = kornia.utils.create_meshgrid(kernel_size, kernel_size, normalized_coordinates=False)
    grid = grid - math.floor(kernel_size / 2)
    kernel = torch.hypot(grid[0, :, :, 0], grid[0, :, :, 1])
    kernel = torch.exp(kernel / h)

    # for BCHW tensors
    kernel = torch.unsqueeze(kernel, 0)
    kernel = torch.unsqueeze(kernel, 0)

    return kernel


def distance_transform(
    image: torch.Tensor,
    kernel_size: int = 7
) -> torch.Tensor:
    r"""Approximates the Manhattan distance transform of images using convolutions.

    The value at each pixel in the output represents the distance to the nearest non-zero pixel in the image image.
    It uses the method described in :cite:`pham2021dtlayer`.
    The transformation is applied independently across the channel dimension of the images.


    Args:
        image: Image with shape :math:`(B,C,H,W)`.
        kernel_size: size of the convolution kernel. Larger kernels are more accurate but less numerically stable.

    Returns:
        tensor with shape :math:`(B,C,H,W)`.

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
    kernel: torch.Tensor = make_cdt_kernel(kernel_size).to(image)

    out = torch.zeros_like(image)

    # It is possible to avoid cloning the image if boundary = image, but this would require modifying the image tensor.
    boundary = image.clone()

    # If image images have multiple channels, view the channels in the batch dimension to match kernel shape.
    if image.shape[1] > 1:
        out = out.view(-1, 1, image.shape[-2], image.shape[-1])
        boundary = boundary.view(-1, 1, image.shape[-2], image.shape[-1])

    for i in range(n_iters):
        cdt = F.conv2d(boundary, kernel, padding='same')
        cdt = -0.35 * torch.log(cdt)

        # We are calculating log(0) above.
        cdt = torch.nan_to_num(cdt, posinf=0.0)

        mask = cdt > 0
        if mask.sum() == 0:
            break

        offset: int = i * kernel_size / 2
        out[mask] += offset + cdt[mask]
        boundary[mask] = 1

    # View channels in the channel dimension, if they were added to batch dimension during transform.
    if image.shape[1] > 1:
        out = out.view(image.shape)

    return out.view_as(image)


class ConvDistanceTransform(nn.Module):
    r"""Module that approximates the Manhattan (city block) distance transform of images using convolutions.

    Args:
        kernel_size: size of the convolution kernel. Larger kernels are more accurate but less numerically stable.

    """
    def __init__(
        self,
        kernel_size: int = 7
    ):
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("Kernel size must be an odd number.")
        self.kernel_size = kernel_size

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return distance_transform(image, self.kernel_size)
