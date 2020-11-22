from typing import Tuple, Union

import torch

import kornia


def _get_nms_kernel2d(kx: int, ky: int) -> torch.Tensor:
    """Utility function, which returns neigh2channels conv kernel"""
    numel: int = ky * kx
    center: int = numel // 2
    weight = torch.eye(numel)
    weight[center, center] = 0
    return weight.view(numel, 1, ky, kx)


def _get_nms_kernel3d(kd: int, ky: int, kx: int) -> torch.Tensor:
    """Utility function, which returns neigh2channels conv kernel"""
    numel: int = kd * ky * kx
    center: int = numel // 2
    weight = torch.eye(numel)
    weight[center, center] = 0
    return weight.view(numel, 1, kd, ky, kx)


class NonMaximaSuppression2d(kornia.nn.NonMaximaSuppression2d):
    r"""Applies non maxima suppression to filter.
    """

    def __init__(self, kernel_size: Tuple[int, int]):
        super(NonMaximaSuppression2d, self).__init__(kernel_size)
        kornia.deprecation_warning(
            "kornia.feature.NonMaximaSuppression2d", "kornia.nn.NonMaximaSuppression2d")


class NonMaximaSuppression3d(kornia.nn.NonMaximaSuppression3d):
    r"""Applies non maxima suppression to filter.
    """

    def __init__(self, kernel_size: Tuple[int, int, int]):
        super(NonMaximaSuppression3d, self).__init__(kernel_size)
        kornia.deprecation_warning(
            "kornia.feature.NonMaximaSuppression3d", "kornia.nn.NonMaximaSuppression3d")


# functiona api


def nms2d(
        input: torch.Tensor, kernel_size: Tuple[int, int], mask_only: bool = False) -> torch.Tensor:
    r"""Applies non maxima suppression to filter.

    See :class:`~kornia.feature.NonMaximaSuppression2d` for details.
    """
    return kornia.nn.NonMaximaSuppression2d(kernel_size)(input, mask_only)


def nms3d(
        input: torch.Tensor, kernel_size: Tuple[int, int, int], mask_only: bool = False) -> torch.Tensor:
    r"""Applies non maxima suppression to filter.

    See :class:`~kornia.feature.NonMaximaSuppression3d` for details.
    """
    return kornia.nn.NonMaximaSuppression3d(kernel_size)(input, mask_only)
