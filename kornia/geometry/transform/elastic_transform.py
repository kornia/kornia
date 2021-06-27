from typing import Tuple

import torch
import torch.nn.functional as F

import kornia
from kornia.filters.kernels import get_gaussian_kernel2d

__all__ = ["elastic_transform2d"]


def elastic_transform2d(
    image: torch.Tensor,
    noise: torch.Tensor,
    kernel_size: Tuple[int, int] = (63, 63),
    sigma: Tuple[float, float] = (32.0, 32.0),
    alpha: Tuple[float, float] = (1.0, 1.0),
    align_corners: bool = False,
    mode: str = 'bilinear',
) -> torch.Tensor:
    r"""Applies elastic transform of images as described in :cite:`Simard2003BestPF`.

    .. image:: _static/img/elastic_transform2d.png

    Args:
        image: Input image to be transformed with shape :math:`(B, C, H, W)`.
        noise: Noise image used to spatially transform the input image. Same
          resolution as the input image with shape :math:`(B, 2, H, W)`. The coordinates order
          it is expected to be in x-y.
        kernel_size: the size of the Gaussian kernel.
        sigma: The standard deviation of the Gaussian in the y and x directions,
          respecitvely. Larger sigma results in smaller pixel displacements.
        alpha : The scaling factor that controls the intensity of the deformation
          in the y and x directions, respectively.
        align_corners: Interpolation flag used by ```grid_sample```.
        mode: Interpolation mode used by ```grid_sample```. Either ``'bilinear'`` or ``'nearest'``.

    .. note:
        ```sigma``` and ```alpha``` can also be a ``torch.Tensor``. However, you could not torchscript
         this function with tensor until PyTorch 1.8 is released.

    Returns:
        the elastically transformed input image with shape :math:`(B,C,H,W)`.

    Example:
        >>> image = torch.rand(1, 3, 5, 5)
        >>> noise = torch.rand(1, 2, 5, 5, requires_grad=True)
        >>> image_hat = elastic_transform2d(image, noise, (3, 3))
        >>> image_hat.mean().backward()

        >>> image = torch.rand(1, 3, 5, 5)
        >>> noise = torch.rand(1, 2, 5, 5)
        >>> sigma = torch.tensor([4., 4.], requires_grad=True)
        >>> image_hat = elastic_transform2d(image, noise, (3, 3), sigma)
        >>> image_hat.mean().backward()

        >>> image = torch.rand(1, 3, 5, 5)
        >>> noise = torch.rand(1, 2, 5, 5)
        >>> alpha = torch.tensor([16., 32.], requires_grad=True)
        >>> image_hat = elastic_transform2d(image, noise, (3, 3), alpha=alpha)
        >>> image_hat.mean().backward()
    """
    if not isinstance(image, torch.Tensor):
        raise TypeError(f"Input image is not torch.Tensor. Got {type(image)}")

    if not isinstance(noise, torch.Tensor):
        raise TypeError(f"Input noise is not torch.Tensor. Got {type(noise)}")

    if not len(image.shape) == 4:
        raise ValueError(f"Invalid image shape, we expect BxCxHxW. Got: {image.shape}")

    if not len(noise.shape) == 4 or noise.shape[1] != 2:
        raise ValueError(f"Invalid noise shape, we expect Bx2xHxW. Got: {noise.shape}")

    # Get Gaussian kernel for 'y' and 'x' displacement
    kernel_x: torch.Tensor = get_gaussian_kernel2d(kernel_size, (sigma[0], sigma[0]))[None]
    kernel_y: torch.Tensor = get_gaussian_kernel2d(kernel_size, (sigma[1], sigma[1]))[None]

    # Convolve over a random displacement matrix and scale them with 'alpha'
    disp_x: torch.Tensor = noise[:, :1]
    disp_y: torch.Tensor = noise[:, 1:]

    disp_x = kornia.filters.filter2d(disp_x, kernel=kernel_y, border_type='constant') * alpha[0]
    disp_y = kornia.filters.filter2d(disp_y, kernel=kernel_x, border_type='constant') * alpha[1]

    # stack and normalize displacement
    disp = torch.cat([disp_x, disp_y], dim=1).permute(0, 2, 3, 1)

    # Warp image based on displacement matrix
    b, c, h, w = image.shape
    grid = kornia.utils.create_meshgrid(h, w, device=image.device).to(image.dtype)
    warped = F.grid_sample(image, (grid + disp).clamp(-1, 1), align_corners=align_corners, mode=mode)

    return warped
