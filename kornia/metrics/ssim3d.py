# LICENSE HEADER MANAGED BY add-license-header
#
# Copyright 2018 Kornia Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from typing import List

import torch
import torch.nn.functional as F
from torch import nn

from kornia.core.check import KORNIA_CHECK, KORNIA_CHECK_IS_TENSOR, KORNIA_CHECK_SHAPE
from kornia.filters import filter3d, get_gaussian_kernel3d
from kornia.filters.filter import _VALID_PADDING, _compute_padding


def _crop(img: torch.Tensor, cropping_shape: List[int]) -> torch.Tensor:
    """Crop out the part of "valid" convolution area."""
    return F.pad(
        img,
        (
            -cropping_shape[4],
            -cropping_shape[5],
            -cropping_shape[2],
            -cropping_shape[3],
            -cropping_shape[0],
            -cropping_shape[1],
        ),
    )


def ssim3d(
    img1: torch.Tensor,
    img2: torch.Tensor,
    window_size: int,
    max_val: float = 1.0,
    eps: float = 1e-12,
    padding: str = "same",
) -> torch.Tensor:
    r"""Compute the Structural Similarity (SSIM) index map between two images.

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

    Convention:
        - See the Convention block of :func:`~kornia.metrics.ssim` for the window, reflected border, ``padding`` and
          ``max_val``. ``ssim3d`` applies them to :math:`(D, H, W)` with an isotropic Gaussian window, and ``'valid'``
          crops every axis. A volume of identical slices reproduces the 2-D map up to floating-point roundoff.
        - The Gaussian window uses the computation dtype of the inputs; integer and half-precision inputs are
          evaluated in float32.

    Args:
        img1: the first input image with shape :math:`(B, C, D, H, W)`.
        img2: the second input image with shape :math:`(B, C, D, H, W)`.
        window_size: the size of the gaussian kernel to smooth the images.
        max_val: the dynamic range of the images.
        eps: Small value for numerically stability when dividing.
        padding: ``'same'`` | ``'valid'``. Whether to only use the "valid" convolution
         area to compute SSIM to match the MATLAB implementation of original SSIM paper.

    Returns:
       The ssim index map with shape :math:`(B, C, D, H, W)`, or :math:`(B, C, D - 2p, H - 2p, W - 2p)` with
       ``p = window_size // 2`` under ``padding='valid'``.

    Note:
        The volume is reflected at its faces for ``padding='same'``, as :func:`kornia.metrics.ssim` does for
        images, so every spatial size must be larger than ``window_size // 2``. The Gaussian window is built in
        the device and dtype of the inputs.
        Integer images are converted to float32 before computing the local moments.
        Half-precision inputs are evaluated in float32 for numerical stability.
        Filtering runs with autocast disabled; the result uses the promoted input dtype.

    Examples:
        >>> input1 = torch.rand(1, 4, 5, 5, 5)
        >>> input2 = torch.rand(1, 4, 5, 5, 5)
        >>> ssim_map = ssim3d(input1, input2, 5)  # 1x4x5x5x5

    """
    KORNIA_CHECK_IS_TENSOR(img1)
    KORNIA_CHECK_IS_TENSOR(img2)
    KORNIA_CHECK_SHAPE(img1, ["B", "C", "D", "H", "W"])
    KORNIA_CHECK_SHAPE(img2, ["B", "C", "D", "H", "W"])
    KORNIA_CHECK(img1.shape == img2.shape, f"img1 and img2 shapes must be the same. Got: {img1.shape} and {img2.shape}")

    if not isinstance(max_val, float):
        raise TypeError(f"Input max_val type is not a float. Got {type(max_val)}")

    KORNIA_CHECK(
        str(padding).lower() in _VALID_PADDING,
        f"Invalid padding mode, {padding}. Expected one of {_VALID_PADDING}",
    )
    # the check is case-insensitive, so dispatch on the lower-case spelling as well
    padding = str(padding).lower()

    # Preserve fractional Gaussian weights and avoid integer moment overflow.
    if not img1.is_floating_point() and not img1.is_complex():
        img1 = img1.to(torch.float32)
    if not img2.is_floating_point() and not img2.is_complex():
        img2 = img2.to(torch.float32)

    output_dtype = torch.promote_types(img1.dtype, img2.dtype)
    # Half-precision moments can overflow before the SSIM ratio is formed.
    if img1.dtype in (torch.float16, torch.bfloat16):
        img1 = img1.float()
    if img2.dtype in (torch.float16, torch.bfloat16):
        img2 = img2.float()
    # Mixed inputs are filtered in their common dtype, which is also the dtype of the window.
    compute_dtype = torch.promote_types(img1.dtype, img2.dtype)
    img1, img2 = img1.to(compute_dtype), img2.to(compute_dtype)

    # prepare kernel
    kernel: torch.Tensor = get_gaussian_kernel3d(
        (window_size, window_size, window_size), (1.5, 1.5, 1.5), device=img1.device, dtype=compute_dtype
    )

    # compute coefficients
    C1: float = (0.01 * max_val) ** 2
    C2: float = (0.03 * max_val) ** 2

    # compute local mean per channel
    with torch.autocast(device_type=img1.device.type, enabled=False):
        mu1: torch.Tensor = filter3d(img1, kernel, border_type="reflect")
        mu2: torch.Tensor = filter3d(img2, kernel, border_type="reflect")

    cropping_shape: List[int] = []
    if padding == "valid":
        depth, height, width = kernel.shape[-3:]
        cropping_shape = _compute_padding([depth, height, width])
        mu1 = _crop(mu1, cropping_shape)
        mu2 = _crop(mu2, cropping_shape)
    elif padding == "same":
        pass

    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2

    with torch.autocast(device_type=img1.device.type, enabled=False):
        mu_img1_sq = filter3d(img1**2, kernel, border_type="reflect")
        mu_img2_sq = filter3d(img2**2, kernel, border_type="reflect")
        mu_img1_img2 = filter3d(img1 * img2, kernel, border_type="reflect")

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

    return (num / (den + eps)).to(output_dtype)


class SSIM3D(nn.Module):
    r"""Create a module that computes the Structural Similarity (SSIM) index between two 3D images.

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

    Convention:
        See the Convention block of :func:`~kornia.metrics.ssim3d`.

    Args:
        window_size: the size of the gaussian kernel to smooth the images.
        max_val: the dynamic range of the images.
        eps: Small value for numerically stability when dividing.
        padding: ``'same'`` | ``'valid'``. Whether to only use the "valid" convolution
         area to compute SSIM to match the MATLAB implementation of original SSIM paper.

    Shape:
        - Input: :math:`(B, C, D, H, W)`.
        - Target :math:`(B, C, D, H, W)`.
        - Output: :math:`(B, C, D, H, W)`, or :math:`(B, C, D - 2p, H - 2p, W - 2p)` with ``p = window_size // 2``
          under ``padding='valid'``.

    Examples:
        >>> input1 = torch.rand(1, 4, 5, 5, 5)
        >>> input2 = torch.rand(1, 4, 5, 5, 5)
        >>> ssim = SSIM3D(5)
        >>> ssim_map = ssim(input1, input2)  # 1x4x5x5x5

    """

    def __init__(self, window_size: int, max_val: float = 1.0, eps: float = 1e-12, padding: str = "same") -> None:
        super().__init__()
        self.window_size: int = window_size
        self.max_val: float = max_val
        self.eps = eps
        self.padding = padding

    def forward(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        """Compute a 3D SSIM map between two volume batches.

        Args:
            img1: First volume tensor with shape :math:`(B, C, D, H, W)`.
            img2: Second volume tensor with the same shape as ``img1``.

        Returns:
            Tensor with shape :math:`(B, C, D, H, W)`, cropped under ``padding='valid'``, containing local
            structural similarity values across depth, height, and width.
        """
        return ssim3d(img1, img2, self.window_size, self.max_val, self.eps, self.padding)
