from typing import Tuple

import torch
from kornia.filters.max_blur_pool import (
    MaxBlurPool2d as _MaxBlurPool2d,
    max_blur_pool2d as _max_blur_pool2d
)

__all__ = [
    "max_blur_pool2d",
    "MaxBlurPool2d",
]


class MaxBlurPool2d(_MaxBlurPool2d):
    __doc__ = _MaxBlurPool2d.__doc__

    def __init__(self, kernel_size: int, ceil_mode: bool = False) -> None:
        super(MaxBlurPool2d, self).__init__(kernel_size,  ceil_mode)
        raise DeprecationWarning(
            "`MaxBlurPool2d` is deprecated. Please use `kornia.filters.MaxBlurPool2d instead.`")


def max_blur_pool2d(input: torch.Tensor, kernel_size: int, ceil_mode: bool = False) -> torch.Tensor:
    r"""Creates a module that computes pools and blurs and downsample a given
    feature map.

    See :class:`~kornia.contrib.MaxBlurPool2d` for details.
    """
    raise DeprecationWarning(
        "`max_blur_pool2d` is deprecated. Please use `kornia.filters.max_blur_pool2d instead.`")
    return _max_blur_pool2d(input, kernel_size, ceil_mode)
