from typing import Tuple, Union

import torch.nn.functional as F

from kornia.core import Tensor, concatenate, tensor
from kornia.core.check import KORNIA_CHECK_IS_TENSOR, KORNIA_CHECK_SHAPE
from kornia.filters import filter2d
from kornia.filters.kernels import get_gaussian_kernel2d
from kornia.utils import create_meshgrid

__all__ = ["elastic_transform2d"]


def elastic_transform2d(
    image: Tensor,
    noise: Tensor,
    kernel_size: Tuple[int, int] = (63, 63),
    sigma: Union[Tuple[float, float], Tensor] = (32.0, 32.0),
    alpha: Union[Tuple[float, float], Tensor] = (1.0, 1.0),
    align_corners: bool = False,
    mode: str = "bilinear",
    padding_mode: str = "zeros",
) -> Tensor:
    r"""Apply elastic transform of images as described in :cite:`Simard2003BestPF`.

    .. image:: _static/img/elastic_transform2d.png

    Args:
        image: Input image to be transformed with shape :math:`(B, C, H, W)`.
        noise: Noise image used to spatially transform the input image. Same
          resolution as the input image with shape :math:`(B, 2, H, W)`. The coordinates order
          it is expected to be in x-y.
        kernel_size: the size of the Gaussian kernel.
        sigma: The standard deviation of the Gaussian in the y and x directions,
          respectively. Larger sigma results in smaller pixel displacements.
        alpha : The scaling factor that controls the intensity of the deformation
          in the y and x directions, respectively.
        align_corners: Interpolation flag used by ```grid_sample```.
        mode: Interpolation mode used by ```grid_sample```. Either ``'bilinear'`` or ``'nearest'``.
        padding_mode: The padding used by ```grid_sample```. Either ``'zeros'``, ``'border'`` or ``'refection'``.

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
    KORNIA_CHECK_IS_TENSOR(image)
    KORNIA_CHECK_IS_TENSOR(noise)
    KORNIA_CHECK_SHAPE(image, ["B", "C", "H", "W"])
    KORNIA_CHECK_SHAPE(noise, ["B", "C", "H", "W"])

    device, dtype = image.device, image.dtype
    # if isinstance(sigma, tuple):
    #    sigma_t = tensor(sigma, device=device, dtype=dtype)
    if isinstance(sigma, Tensor):
        sigma = sigma.expand(2)[None, ...]
    #        sigma = sigma.to(device=device, dtype=dtype)

    # Get Gaussian kernel for 'y' and 'x' displacement
    kernel_x = get_gaussian_kernel2d(kernel_size, sigma)  # _t[0].expand(2).unsqueeze(0))
    kernel_y = get_gaussian_kernel2d(kernel_size, sigma)  # _t[1].expand(2).unsqueeze(0))

    if isinstance(alpha, Tensor):
        alpha_x = alpha[0]
        alpha_y = alpha[1]
    else:
        alpha_x = tensor(alpha[0], device=device, dtype=dtype)
        alpha_y = tensor(alpha[1], device=device, dtype=dtype)

    # Convolve over a random displacement matrix and scale them with 'alpha'
    disp_x = noise[:, :1]
    disp_y = noise[:, 1:]

    disp_x = filter2d(disp_x, kernel=kernel_y, border_type="constant") * alpha_x
    disp_y = filter2d(disp_y, kernel=kernel_x, border_type="constant") * alpha_y

    # stack and normalize displacement
    disp = concatenate([disp_x, disp_y], 1).permute(0, 2, 3, 1)

    # Warp image based on displacement matrix
    _, _, h, w = image.shape
    grid = create_meshgrid(h, w, device=image.device).to(image.dtype)
    warped = F.grid_sample(
        image, (grid + disp).clamp(-1, 1), align_corners=align_corners, mode=mode, padding_mode=padding_mode
    )

    return warped
