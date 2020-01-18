from typing import Tuple, Union

import torch
import torch.nn as nn


class NonMaximaSuppression2d(nn.Module):
    r"""Applies non maxima suppression to filter.
    """

    def __init__(self, kernel_size: Tuple[int, int]):
        super(NonMaximaSuppression2d, self).__init__()
        self.kernel_size: Tuple[int, int] = kernel_size
        self.padding: Tuple[int,
                            int] = self._compute_zero_padding2d(kernel_size)
        self.max_pool2d = nn.MaxPool2d(kernel_size, stride=1,
                                       padding=self.padding)

    @staticmethod
    def _compute_zero_padding2d(
            kernel_size: Tuple[int, int]) -> Tuple[int, int]:
        assert isinstance(kernel_size, tuple), type(kernel_size)
        assert len(kernel_size) == 2, kernel_size

        def pad(x):
            return (x - 1) // 2  # zero padding function

        ky, kx = kernel_size     # we assume a cubic kernel
        return (pad(ky), pad(kx))

    def forward(self, x: torch.Tensor, mask_only: bool = False) -> torch.Tensor:  # type: ignore
        assert len(x.shape) == 4, x.shape
        # find local maximum values
        if mask_only:
            return x == self.max_pool2d(x)
        return x * (x == self.max_pool2d(x)).to(x.dtype)


class NonMaximaSuppression3d(nn.Module):
    r"""Applies non maxima suppression to filter.
    """

    def __init__(self, kernel_size: Tuple[int, int, int]):
        super(NonMaximaSuppression3d, self).__init__()
        self.kernel_size: Tuple[int, int, int] = kernel_size
        self.padding: Tuple[int,
                            int,
                            int] = self._compute_zero_padding3d(kernel_size)
        self.max_pool3d = nn.MaxPool3d(kernel_size, stride=1,
                                       padding=self.padding)

    @staticmethod
    def _compute_zero_padding3d(
            kernel_size: Tuple[int, int, int]) -> Tuple[int, int, int]:
        assert isinstance(kernel_size, tuple), type(kernel_size)
        assert len(kernel_size) == 3, kernel_size

        def pad(x):
            return (x - 1) // 2  # zero padding function

        kd, ky, kx = kernel_size     # we assume a cubic kernel
        return pad(kd), pad(ky), pad(kx)

    def forward(self, x: torch.Tensor, mask_only: bool = False) -> torch.Tensor:  # type: ignore
        assert len(x.shape) == 5, x.shape
        # find local maximum values
        if mask_only:
            return x == self.max_pool3d(x)
        return x * (x == self.max_pool3d(x)).to(x.dtype)

# functiona api


def nms2d(
        input: torch.Tensor, kernel_size: Tuple[int, int], mask_only: bool = False) -> torch.Tensor:
    r"""Applies non maxima suppression to filter.

    See :class:`~kornia.feature.NonMaximaSuppression2d` for details.
    """
    return NonMaximaSuppression2d(kernel_size)(input, mask_only)


def nms3d(
        input: torch.Tensor, kernel_size: Tuple[int, int, int], mask_only: bool = False) -> torch.Tensor:
    r"""Applies non maxima suppression to filter.

    See :class:`~kornia.feature.NonMaximaSuppression3d` for details.
    """
    return NonMaximaSuppression3d(kernel_size)(input, mask_only)
