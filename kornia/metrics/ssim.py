from typing import List

import torch
from torch import nn

from kornia.filters import filter2d_separable, get_gaussian_kernel1d
from kornia.filters.filter import _compute_padding


def _crop(img: torch.Tensor, cropping_shape: List[int]) -> torch.Tensor:
    """Crop out the part of "valid" convolution area."""
    return torch.nn.functional.pad(
        img, (-cropping_shape[2], -cropping_shape[3], -cropping_shape[0], -cropping_shape[1])
    )


def ssim(
    img1: torch.Tensor,
    img2: torch.Tensor,
    window_size: int,
    max_val: float = 1.0,
    eps: float = 1e-12,
    padding: str = "same",
) -> torch.Tensor:
    r"""Function that computes the Structural Similarity (SSIM) index map between two images.

    Measures the (SSIM) index between each element in the input `x` and target `y`.

    The index can be described as:

    .. math::

      \text{SSIM}(x, y) = \frac{(2\mu_x\mu_y+c_1)(2\sigma_{xy}+c_2)}
      {(\mu_x^2+\mu_y^2+c_1)(\sigma_x^2+\sigma_y^2+c_2)}

    where:
      - :math:`c_1=(k_1 L)^2` and :math:`c_2=(k_2 L)^2` are two variables to
        stabilize the division with weak denominator.
      - :math:`L` is the dynamic range of the pixel-values (typically this is
        :math:`2^{\#\text{bits per pixel}}-1`).

    Args:
        img1: the first input image with shape :math:`(B, C, H, W)`.
        img2: the second input image with shape :math:`(B, C, H, W)`.
        window_size: the size of the gaussian kernel to smooth the images.
        max_val: the dynamic range of the images.
        eps: Small value for numerically stability when dividing.
        padding: ``'same'`` | ``'valid'``. Whether to only use the "valid" convolution
         area to compute SSIM to match the MATLAB implementation of original SSIM paper.

    Returns:
       The ssim index map with shape :math:`(B, C, H, W)`.

    Examples:
        >>> input1 = torch.rand(1, 4, 5, 5)
        >>> input2 = torch.rand(1, 4, 5, 5)
        >>> ssim_map = ssim(input1, input2, 5)  # 1x4x5x5
    """
    if not isinstance(img1, torch.Tensor):
        raise TypeError(f"Input img1 type is not a torch.Tensor. Got {type(img1)}")

    if not isinstance(img2, torch.Tensor):
        raise TypeError(f"Input img2 type is not a torch.Tensor. Got {type(img2)}")

    if not isinstance(max_val, float):
        raise TypeError(f"Input max_val type is not a float. Got {type(max_val)}")

    if not len(img1.shape) == 4:
        raise ValueError(f"Invalid img1 shape, we expect BxCxHxW. Got: {img1.shape}")

    if not len(img2.shape) == 4:
        raise ValueError(f"Invalid img2 shape, we expect BxCxHxW. Got: {img2.shape}")

    if not img1.shape == img2.shape:
        raise ValueError(f"img1 and img2 shapes must be the same. Got: {img1.shape} and {img2.shape}")

    # prepare kernel
    kernel: torch.Tensor = get_gaussian_kernel1d(window_size, 1.5, device=img1.device, dtype=img1.dtype)

    # compute coefficients
    C1: float = (0.01 * max_val) ** 2
    C2: float = (0.03 * max_val) ** 2

    # compute local mean per channel
    mu1: torch.Tensor = filter2d_separable(img1, kernel, kernel)
    mu2: torch.Tensor = filter2d_separable(img2, kernel, kernel)

    cropping_shape: List[int] = []
    if padding == "valid":
        height = width = kernel.shape[-1]
        cropping_shape = _compute_padding([height, width])
        mu1 = _crop(mu1, cropping_shape)
        mu2 = _crop(mu2, cropping_shape)
    elif padding == "same":
        pass

    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2

    mu_img1_sq = filter2d_separable(img1**2, kernel, kernel)
    mu_img2_sq = filter2d_separable(img2**2, kernel, kernel)
    mu_img1_img2 = filter2d_separable(img1 * img2, kernel, kernel)

    if padding == "valid":
        mu_img1_sq = _crop(mu_img1_sq, cropping_shape)
        mu_img2_sq = _crop(mu_img2_sq, cropping_shape)
        mu_img1_img2 = _crop(mu_img1_img2, cropping_shape)
    elif padding == "same":
        pass

    # compute local sigma per channel
    sigma1_sq = mu_img1_sq - mu1_sq
    sigma2_sq = mu_img2_sq - mu2_sq
    sigma12 = mu_img1_img2 - mu1_mu2

    # compute the similarity index map
    num: torch.Tensor = (2.0 * mu1_mu2 + C1) * (2.0 * sigma12 + C2)
    den: torch.Tensor = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)

    return num / (den + eps)


class SSIM(nn.Module):
    r"""Create a module that computes the Structural Similarity (SSIM) index between two images.

    Measures the (SSIM) index between each element in the input `x` and target `y`.

    The index can be described as:

    .. math::

      \text{SSIM}(x, y) = \frac{(2\mu_x\mu_y+c_1)(2\sigma_{xy}+c_2)}
      {(\mu_x^2+\mu_y^2+c_1)(\sigma_x^2+\sigma_y^2+c_2)}

    where:
      - :math:`c_1=(k_1 L)^2` and :math:`c_2=(k_2 L)^2` are two variables to
        stabilize the division with weak denominator.
      - :math:`L` is the dynamic range of the pixel-values (typically this is
        :math:`2^{\#\text{bits per pixel}}-1`).

    Args:
        window_size: the size of the gaussian kernel to smooth the images.
        max_val: the dynamic range of the images.
        eps: Small value for numerically stability when dividing.
        padding: ``'same'`` | ``'valid'``. Whether to only use the "valid" convolution
         area to compute SSIM to match the MATLAB implementation of original SSIM paper.

    Shape:
        - Input: :math:`(B, C, H, W)`.
        - Target :math:`(B, C, H, W)`.
        - Output: :math:`(B, C, H, W)`.

    Examples:
        >>> input1 = torch.rand(1, 4, 5, 5)
        >>> input2 = torch.rand(1, 4, 5, 5)
        >>> ssim = SSIM(5)
        >>> ssim_map = ssim(input1, input2)  # 1x4x5x5
    """

    def __init__(self, window_size: int, max_val: float = 1.0, eps: float = 1e-12, padding: str = "same") -> None:
        super().__init__()
        self.window_size: int = window_size
        self.max_val: float = max_val
        self.eps = eps
        self.padding = padding

    def forward(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        return ssim(img1, img2, self.window_size, self.max_val, self.eps, self.padding)
