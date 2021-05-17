from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from kornia.filters import (
    gaussian_blur2d, spatial_gradient
)


def canny(input: torch.Tensor, kernel_size: Tuple[int, int] = (5,5), sigma: Tuple[float, float] = (1,1), normalized: bool = True, eps: float = 1e-6) -> torch.Tensor:
    r"""Computes the Canny operator and returns the magnitude per channel.

    Args:
        input (torch.Tensor): the input image with shape :math:`(B,C,H,W)`.
        normalized (bool): if True, L1 norm of the kernel is set to 1.
        eps (float): regularization number to avoid NaN during backprop. Default: 1e-6.

    Return:
        torch.Tensor: the canny edge gradient magnitudes map with shape :math:`(B,C,H,W)`.

    Example:
        >>> input = torch.rand(1, 3, 4, 4)
        >>> output = canny(input)  # 1x3x4x4
        >>> output.shape
        torch.Size([1, 3, 4, 4])
    """
    if not isinstance(input, torch.Tensor):
        raise TypeError("Input type is not a torch.Tensor. Got {}"
                        .format(type(input)))

    if not len(input.shape) == 4:
        raise ValueError("Invalid input shape, we expect BxCxHxW. Got: {}"
                         .format(input.shape))

    # Gaussian filter
    blurred: torch.Tensor = gaussian_blur2d(input, kernel_size, sigma)

    # comput the x/y gradients
    edges: torch.Tensor = spatial_gradient(blurred, normalized=normalized)

    # unpack the edges
    gx: torch.Tensor = edges[:, :, 0]
    gy: torch.Tensor = edges[:, :, 1]

    # compute gradient maginitude
    magnitude: torch.Tensor = torch.sqrt(gx * gx + gy * gy + eps)
    angle: torch.Tensor = torch.atan2(gy, gx)

    # Non-maximal suppression

    # Threshold & Hysteresis

    return magnitude


class Canny(nn.Module):
    r"""Computes the Canny operator and returns the magnitude per channel.

    Args:
        normalized (bool): if True, L1 norm of the kernel is set to 1.
        eps (float): regularization number to avoid NaN during backprop. Default: 1e-6.

    Return:
        torch.Tensor: the sobel edge gradient magnitudes map.

    Shape:
        - Input: :math:`(B, C, H, W)`
        - Output: :math:`(B, C, H, W)`

    Examples:
        >>> input = torch.rand(1, 3, 4, 4)
        >>> output = Sobel()(input)  # 1x3x4x4
    """

    def __init__(self,
                 kernel_size: Tuple[int, int] = (5,5), sigma: Tuple[float, float] = (1,1), normalized: bool = True, eps: float = 1e-6) -> None:
        super(Canny, self).__init__()

        # Gaussian blur parameters
        self.kernel_size = kernel_size
        self.sigma = sigma

        # Spatial gradients
        self.normalized = normalized

        self.eps: float = eps

    def __repr__(self) -> str:
        return self.__class__.__name__ + '('\
            'kernel_size=' + str(self.kernel_size) +\
            'sigma=' + str(self.sigma) + ')'

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return canny(input, self.kernel_size, self.sigma, self.normalized, self.eps)

