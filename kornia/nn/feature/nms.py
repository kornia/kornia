from typing import Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

import kornia


class NonMaximaSuppression2d(nn.Module):
    r"""Applies non maxima suppression to filter.
    """

    def __init__(self, kernel_size: Tuple[int, int]):
        super(NonMaximaSuppression2d, self).__init__()
        self.kernel_size: Tuple[int, int] = kernel_size
        self.padding: Tuple[int, int,
                            int, int] = self._compute_zero_padding2d(kernel_size)
        self.kernel = kornia.feature.nms._get_nms_kernel2d(*kernel_size)

    @staticmethod
    def _compute_zero_padding2d(
            kernel_size: Tuple[int, int]) -> Tuple[int, int, int, int]:
        assert isinstance(kernel_size, tuple), type(kernel_size)
        assert len(kernel_size) == 2, kernel_size

        def pad(x):
            return (x - 1) // 2  # zero padding function

        ky, kx = kernel_size     # we assume a cubic kernel
        return (pad(ky), pad(ky), pad(kx), pad(kx))

    def forward(self, x: torch.Tensor, mask_only: bool = False) -> torch.Tensor:  # type: ignore
        assert len(x.shape) == 4, x.shape
        B, CH, H, W = x.size()
        # find local maximum values
        max_non_center = F.conv2d(F.pad(x, list(self.padding)[::-1], mode='replicate'),
                                  self.kernel.repeat(CH, 1, 1, 1).to(x.device, x.dtype),
                                  stride=1,
                                  groups=CH).view(B, CH, -1, H, W).max(dim=2)[0]
        mask = x > max_non_center
        if mask_only:
            return mask
        return x * (mask.to(x.dtype))


class NonMaximaSuppression3d(nn.Module):
    r"""Applies non maxima suppression to filter.
    """

    def __init__(self, kernel_size: Tuple[int, int, int]):
        super(NonMaximaSuppression3d, self).__init__()
        self.kernel_size: Tuple[int, int, int] = kernel_size
        self.padding: Tuple[int, int, int,
                            int, int, int] = self._compute_zero_padding3d(kernel_size)
        self.kernel = kornia.feature.nms._get_nms_kernel3d(*kernel_size)

    @staticmethod
    def _compute_zero_padding3d(
            kernel_size: Tuple[int, int, int]) -> Tuple[int, int, int, int, int, int]:
        assert isinstance(kernel_size, tuple), type(kernel_size)
        assert len(kernel_size) == 3, kernel_size

        def pad(x):
            return (x - 1) // 2  # zero padding function

        kd, ky, kx = kernel_size     # we assume a cubic kernel
        return pad(kd), pad(kd), pad(ky), pad(ky), pad(kx), pad(kx)

    def forward(self, x: torch.Tensor, mask_only: bool = False) -> torch.Tensor:  # type: ignore
        assert len(x.shape) == 5, x.shape
        # find local maximum values
        B, CH, D, H, W = x.size()
        max_non_center = F.conv3d(F.pad(x, list(self.padding)[::-1], mode='replicate'),
                                  self.kernel.repeat(CH, 1, 1, 1, 1).to(x.device, x.dtype),
                                  stride=1,
                                  groups=CH).view(B, CH, -1, D, H, W).max(dim=2, keepdim=False)[0]
        mask = x > max_non_center
        if mask_only:
            return mask
        return x * (mask.to(x.dtype))
