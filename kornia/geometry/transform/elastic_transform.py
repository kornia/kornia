from typing import Tuple

import torch
import torch.nn.functional as F
import warnings

import kornia
from kornia.filters.kernels import get_gaussian_kernel2d

__all__ = ["elastic_transform2d"]


def elastic_transform2d(
    image: torch.Tensor,
    noise: torch.Tensor,
    kernel_size: Tuple[int, int] = (63, 63),
    sigma: Tuple[float, float] = (32., 32.),
    alpha: Tuple[float, float] = (16., 16.),
    align_corners: bool = False,
    mode: str = 'bilinear',
    approach: str = 'coarse_noise'
) -> torch.Tensor:
    r"""Applies elastic transform of images as described in :cite:`Simard2003BestPF`.

    .. warning::
        'smoothing' approach is deprecated and will be removed > 0.6.0. Pleas use the more computational efficent
        approach 'upsampling' which is similar to the elasticdeform package.

    Args:
        image (torch.Tensor): Input image to be transformed with shape :math:`(B, C, H_1, W_1)`.
        noise (torch.Tensor): Noise image used to spatially transform the input image. Abitory spatial resolution
        :math:`(B, 2, H_2, W_2)`. The coordinates order it is expected to be in x-y.
        kernel_size (Tuple[int, int]): The size of the Gaussian kernel. Default: (63, 63). Only used if 'smoothing'
          approach is utilized.
        sigma (Tuple[float, float]): The standard deviation of the Gaussian in the y and x directions,
          respecitvely. Larger sigma results in smaller pixel displacements. Default: (32, 32). Only used if
          'smoothing' approach is utilized.
        alpha (Tuple[float, float]): The scaling factor that controls the intensity of the deformation in pixels
          in the y and x directions, respectively. Default: (16, 16).
        align_corners (bool): Interpolation flag used by `grid_sample`. Default: False.
        mode (str): Interpolation mode used by `grid_sample`. Either 'bilinear' or 'nearest'. Default: 'bilinear'.
        approach: (str): Transformation approach to be peformed. Either 'coarse_noise' or 'smoothing'. 'coarse_noise'
          utilizes a coarse noise (deformation) tensor, which gets upsampled to the image resolution. 'smoothing' uses
          a noise tensor with the same spatial resolution as image, which gets smoothed by a Gaussian filter to achive
          smooth deformations.

    .. note:
        `sigma` and `alpha` can also be a `torch.Tensor`. However, you could not torchscript
         this function with tensor until PyTorch 1.8 is released.

    .. note:
        `noise` recomended to be in the range between minus one and one. Spatial dimensions of `noise` recommended to
        be smaller than `image` shape.

    Returns:
        torch.Tensor: the elastically transformed input image with shape :math:`(B,C,H,W)`.

    Example:
        >>> image = torch.rand(1, 3, 32, 32)
        >>> noise = torch.randn(1, 2, 4, 4, requires_grad=True)
        >>> image_hat = elastic_transform2d(image, noise)
        >>> image_hat.mean().backward()

        >>> image = torch.rand(1, 3, 32, 32)
        >>> noise = torch.randn(1, 2, 4, 4, requires_grad=True)
        >>> image_hat = elastic_transform2d(image, noise, alpha=(24, 24), approach='coarse_noise')
        >>> image_hat.mean().backward()

        >>> image = torch.rand(1, 3, 32, 32)
        >>> noise = torch.randn(1, 2, 8, 8, requires_grad=True)
        >>> alpha = torch.tensor([24, 24])
        >>> image_hat = elastic_transform2d(image, noise, alpha=alpha, approach='coarse_noise')
        >>> image_hat.mean().backward()

        >>> image = torch.rand(1, 3, 5, 5)
        >>> noise = torch.randn(1, 2, 5, 5, requires_grad=True)
        >>> image_hat = elastic_transform2d(image, noise, (3, 3), approach='smoothing')
        >>> image_hat.mean().backward()
    """

    # Check parameters
    if not isinstance(image, torch.Tensor):
        raise TypeError(f"Input image is not torch.Tensor. Got {type(image)}")

    if not isinstance(noise, torch.Tensor):
        raise TypeError(f"Input noise is not torch.Tensor. Got {type(noise)}")

    if not len(image.shape) == 4:
        raise ValueError(f"Invalid image shape, we expect BxCxHxW. Got: {image.shape}")

    if not len(noise.shape) == 4 or noise.shape[1] != 2:
        raise ValueError(f"Invalid noise shape, we expect Bx2xHxW. Got: {noise.shape}")

    if approach not in ['smoothing', 'coarse_noise']:
        raise ValueError(f"Invalid approach, choose either 'smoothing' or 'coarse_noise'. Got: {approach}")

    # Perform deprecated code if 'smoothing' is utilized
    if approach == 'smoothing':

        # If jit scripting is performed warning is omitted
        if not torch.jit.is_scripting():
            # Show warning if smoothing is utilized.
            warnings.warn("approach='smoothing' is deprecated and will be removed > 0.6.0. "
                          "Please use approach='coarse_noise'.", category=DeprecationWarning, stacklevel=2)

        # Get Gaussian kernel for 'y' and 'x' displacement
        kernel_x: torch.Tensor = get_gaussian_kernel2d(kernel_size, (sigma[0], sigma[0]))[None]
        kernel_y: torch.Tensor = get_gaussian_kernel2d(kernel_size, (sigma[1], sigma[1]))[None]

        # Convolve over a random displacement matrix and scale them with 'alpha'
        disp_x: torch.Tensor = noise[:, :1]
        disp_y: torch.Tensor = noise[:, 1:]

        disp_x = kornia.filters.filter2D(disp_x, kernel=kernel_y, border_type='constant') * alpha[0]
        disp_y = kornia.filters.filter2D(disp_y, kernel=kernel_x, border_type='constant') * alpha[1]

        # stack and normalize displacement
        disp = torch.cat([disp_x, disp_y], dim=1).permute(0, 2, 3, 1)

        # Warp image based on displacement matrix
        b, c, h, w = image.shape
        grid = kornia.utils.create_meshgrid(h, w, device=image.device).to(image.dtype)
        warped = F.grid_sample(image, (grid + disp).clamp(-1, 1), align_corners=align_corners, mode=mode)

        return warped

    # Perform 'coarse_noise' approach
    b, c, h, w = image.shape

    # Upsample noise ('coarse_noise') to the size of image
    disp = F.interpolate(noise, size=(h, w), mode=mode, align_corners=align_corners)
    disp = disp.permute(0, 2, 3, 1)

    # Apply alpha relative to image size
    disp[..., 0] = disp[..., 0] * alpha[0] / float(h)
    disp[..., 1] = disp[..., 1] * alpha[1] / float(w)

    # Warp image based on displacement matrix
    grid = kornia.utils.create_meshgrid(h, w, device=image.device).to(image.dtype)
    warped = F.grid_sample(image, (grid + disp).clamp(-1, 1), align_corners=align_corners, mode=mode)

    return warped
