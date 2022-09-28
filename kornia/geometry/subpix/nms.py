from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _get_nms_kernel2d(kx: int, ky: int) -> torch.Tensor:
    """Utility function, which returns neigh2channels conv kernel."""
    numel: int = ky * kx
    center: int = numel // 2
    weight = torch.eye(numel)
    weight[center, center] = 0
    return weight.view(numel, 1, ky, kx)


def _get_nms_kernel3d(kd: int, ky: int, kx: int) -> torch.Tensor:
    """Utility function, which returns neigh2channels conv kernel."""
    numel: int = kd * ky * kx
    center: int = numel // 2
    weight = torch.eye(numel)
    weight[center, center] = 0
    return weight.view(numel, 1, kd, ky, kx)


class NonMaximaSuppression2d(nn.Module):
    r"""Apply non maxima suppression to filter."""

    def __init__(self, kernel_size: tuple[int, int]):
        super().__init__()
        self.kernel_size: tuple[int, int] = kernel_size
        self.padding: tuple[int, int, int, int] = self._compute_zero_padding2d(kernel_size)
        self.register_buffer('kernel', _get_nms_kernel2d(*kernel_size))

    @staticmethod
    def _compute_zero_padding2d(kernel_size: tuple[int, int]) -> tuple[int, int, int, int]:
        if not isinstance(kernel_size, tuple):
            raise AssertionError(type(kernel_size))
        if len(kernel_size) != 2:
            raise AssertionError(kernel_size)

        def pad(x):
            return (x - 1) // 2  # zero padding function

        ky, kx = kernel_size  # we assume a cubic kernel
        return (pad(ky), pad(ky), pad(kx), pad(kx))

    def forward(self, x: torch.Tensor, mask_only: bool = False) -> torch.Tensor:  # type: ignore
        if len(x.shape) != 4:
            raise AssertionError(x.shape)
        B, CH, H, W = x.size()
        # find local maximum values
        x_padded = F.pad(x, list(self.padding)[::-1], mode='replicate')
        B, CH, HP, WP = x_padded.size()

        max_non_center = (
            F.conv2d(x_padded.view(B * CH, 1, HP, WP), self.kernel.to(x.device, x.dtype), stride=1)  # type: ignore
            .view(B, CH, -1, H, W)
            .max(dim=2)[0]
        )
        mask = x > max_non_center
        if mask_only:
            return mask
        return x * (mask.to(x.dtype))


class NonMaximaSuppression3d(nn.Module):
    r"""Apply non maxima suppression to filter."""

    def __init__(self, kernel_size: tuple[int, int, int]):
        super().__init__()
        self.kernel_size: tuple[int, int, int] = kernel_size
        self.padding: tuple[int, int, int, int, int, int] = self._compute_zero_padding3d(kernel_size)
        self.kernel = _get_nms_kernel3d(*kernel_size)

    @staticmethod
    def _compute_zero_padding3d(kernel_size: tuple[int, int, int]) -> tuple[int, int, int, int, int, int]:
        if not isinstance(kernel_size, tuple):
            raise AssertionError(type(kernel_size))
        if len(kernel_size) != 3:
            raise AssertionError(kernel_size)

        def pad(x):
            return (x - 1) // 2  # zero padding function

        kd, ky, kx = kernel_size  # we assume a cubic kernel
        return pad(kd), pad(kd), pad(ky), pad(ky), pad(kx), pad(kx)

    def forward(self, x: torch.Tensor, mask_only: bool = False) -> torch.Tensor:  # type: ignore
        if len(x.shape) != 5:
            raise AssertionError(x.shape)
        # find local maximum values
        B, CH, D, H, W = x.size()
        if self.kernel_size == (3, 3, 3):
            mask = torch.zeros(B, CH, D, H, W, device=x.device, dtype=torch.bool)
            center = slice(1, -1)
            left = slice(0, -2)
            right = slice(2, None)
            center_tensor = x[..., center, center, center]
            mask[..., 1:-1, 1:-1, 1:-1] = (
                (center_tensor > x[..., center, center, left])
                & (center_tensor > x[..., center, center, right])
                & (center_tensor > x[..., center, left, center])
                & (center_tensor > x[..., center, left, left])
                & (center_tensor > x[..., center, left, right])
                & (center_tensor > x[..., center, right, center])
                & (center_tensor > x[..., center, right, left])
                & (center_tensor > x[..., center, right, right])
                & (center_tensor > x[..., left, center, center])
                & (center_tensor > x[..., left, center, left])
                & (center_tensor > x[..., left, center, right])
                & (center_tensor > x[..., left, left, center])
                & (center_tensor > x[..., left, left, left])
                & (center_tensor > x[..., left, left, right])
                & (center_tensor > x[..., left, right, center])
                & (center_tensor > x[..., left, right, left])
                & (center_tensor > x[..., left, right, right])
                & (center_tensor > x[..., right, center, center])
                & (center_tensor > x[..., right, center, left])
                & (center_tensor > x[..., right, center, right])
                & (center_tensor > x[..., right, left, center])
                & (center_tensor > x[..., right, left, left])
                & (center_tensor > x[..., right, left, right])
                & (center_tensor > x[..., right, right, center])
                & (center_tensor > x[..., right, right, left])
                & (center_tensor > x[..., right, right, right])
            )
        else:
            max_non_center = (
                F.conv3d(
                    F.pad(x, list(self.padding)[::-1], mode='replicate'),
                    self.kernel.repeat(CH, 1, 1, 1, 1).to(x.device, x.dtype),
                    stride=1,
                    groups=CH,
                )
                .view(B, CH, -1, D, H, W)
                .max(dim=2, keepdim=False)[0]
            )
            mask = x > max_non_center
        if mask_only:
            return mask
        return x * (mask.to(x.dtype))


# functional api


def nms2d(input: torch.Tensor, kernel_size: tuple[int, int], mask_only: bool = False) -> torch.Tensor:
    r"""Apply non maxima suppression to filter.

    See :class:`~kornia.geometry.subpix.NonMaximaSuppression2d` for details.
    """
    return NonMaximaSuppression2d(kernel_size)(input, mask_only)


def nms3d(input: torch.Tensor, kernel_size: tuple[int, int, int], mask_only: bool = False) -> torch.Tensor:
    r"""Apply non maxima suppression to filter.

    See :class:`~kornia.feature.NonMaximaSuppression3d` for details.
    """
    return NonMaximaSuppression3d(kernel_size)(input, mask_only)
