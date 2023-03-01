from __future__ import annotations

import torch.nn.functional as F

from kornia.core import Module, Tensor
from kornia.core.check import KORNIA_CHECK_IS_TENSOR, KORNIA_CHECK_SHAPE

from .gaussian import get_gaussian_kernel2d
from .kernels import _unpack_2d_ks
from .median import _compute_zero_padding


def bilateral_blur(
    input: Tensor,
    kernel_size: tuple[int, int] | int,
    sigma_color: float,
    sigma_space: tuple[float, float],
    border_type: str = 'reflect',
    color_distance_type: str = "l1",
) -> Tensor:
    KORNIA_CHECK_IS_TENSOR(input)
    KORNIA_CHECK_SHAPE(input, ['B', 'C', 'H', 'W'])

    kx, ky = _unpack_2d_ks(kernel_size)
    pad_x, pad_y = _compute_zero_padding(kernel_size)

    padded = F.pad(input, (pad_x, pad_x, pad_y, pad_y), mode=border_type)
    unfolded = padded.unfold(2, ky, 1).unfold(3, kx, 1).flatten(-2)  # (B, C, H, W, K x K)

    diff = unfolded - input.unsqueeze(-1)
    if color_distance_type == "l1":
        color_distance_sq = diff.abs().sum(1, keepdim=True).square()
    elif color_distance_type == "l2":
        color_distance_sq = diff.square().sum(1, keepdim=True)
    else:
        raise ValueError("color_distance_type only acceps l1 or l2")
    color_kernel = (-0.5 / sigma_color**2 * color_distance_sq).exp()  # (B, 1, H, W, K x K)

    space_kernel = get_gaussian_kernel2d(kernel_size, sigma_space, device=input.device, dtype=input.dtype).view(
        1, 1, 1, 1, -1
    )

    kernel = space_kernel * color_kernel
    out = (unfolded * kernel).sum(-1) / kernel.sum(-1)
    return out


class BilateralBlur(Module):
    def __init__(
        self,
        kernel_size: tuple[int, int] | int,
        sigma_color: float,
        sigma_space: tuple[float, float],
        border_type: str = 'reflect',
        color_distance_type: str = "l1",
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.sigma_color = sigma_color
        self.sigma_space = sigma_space
        self.border_type = border_type
        self.color_distance_type = color_distance_type

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}"
            f"(kernel_size={self.kernel_size}, "
            f"sigma_color={self.sigma_color}, "
            f"sigma_space={self.sigma_space}, "
            f"border_type={self.border_type}, "
            f"color_distance_type={self.color_distance_type})"
        )

    def forward(self, input: Tensor) -> Tensor:
        return bilateral_blur(
            input, self.kernel_size, self.sigma_color, self.sigma_space, self.border_type, self.color_distance_type
        )
