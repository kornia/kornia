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

from typing import Tuple, Union

import torch
import torch.nn.functional as F

from kornia.core.check import KORNIA_CHECK_IS_TENSOR, KORNIA_CHECK_SHAPE
from kornia.filters import filter2d_separable
from kornia.filters.kernels import get_gaussian_kernel1d
from kornia.geometry.grid import create_meshgrid

__all__ = ["elastic_transform2d"]


def elastic_transform2d(
    image: torch.Tensor,
    noise: torch.Tensor,
    kernel_size: Tuple[int, int] = (63, 63),
    sigma: Union[Tuple[float, float], torch.Tensor] = (32.0, 32.0),
    alpha: Union[Tuple[float, float], torch.Tensor] = (1.0, 1.0),
    align_corners: bool = False,
    mode: str = "bilinear",
    padding_mode: str = "zeros",
) -> torch.Tensor:
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
        padding_mode: The padding used by ```grid_sample```. Either ``'torch.zeros'``, ``'border'`` or ``'refection'``.

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

    # Normalise sigma to a (B, 2) = (sigma_y, sigma_x) tensor, matching get_gaussian_kernel2d.
    if isinstance(sigma, torch.Tensor):
        sigma = sigma.expand(2)[None, ...]
    else:
        sigma = torch.tensor([sigma], device=device, dtype=dtype)
    ksize_y, ksize_x = kernel_size
    sigma_y, sigma_x = sigma[:, 0, None], sigma[:, 1, None]

    # The Gaussian used to smooth the displacement field is separable, so filter with two 1-D
    # passes rather than a full (ksize_y x ksize_x) 2-D convolution: numerically identical up to
    # float-accumulation order (~1e-7) and ~30-60x cheaper for the default (63, 63) kernel.
    kernel_1d_x = get_gaussian_kernel1d(ksize_x, sigma_x, device=device, dtype=dtype)
    kernel_1d_y = get_gaussian_kernel1d(ksize_y, sigma_y, device=device, dtype=dtype)

    if isinstance(alpha, torch.Tensor):
        alpha_x = alpha[0]
        alpha_y = alpha[1]
    else:
        alpha_x = torch.tensor(alpha[0], device=device, dtype=dtype)
        alpha_y = torch.tensor(alpha[1], device=device, dtype=dtype)

    # Convolve over the random displacement matrix and scale by 'alpha'.
    disp_x = filter2d_separable(noise[:, :1], kernel_1d_x, kernel_1d_y, border_type="constant") * alpha_x
    disp_y = filter2d_separable(noise[:, 1:], kernel_1d_x, kernel_1d_y, border_type="constant") * alpha_y

    # torch.stack and F.normalize displacement
    disp = torch.cat([disp_x, disp_y], 1).permute(0, 2, 3, 1)

    # Warp image based on displacement matrix
    _, _, h, w = image.shape
    grid = create_meshgrid(h, w, device=image.device).to(image.dtype)
    warped = F.grid_sample(
        image, (grid + disp).clamp(-1, 1), align_corners=align_corners, mode=mode, padding_mode=padding_mode
    )

    return warped
