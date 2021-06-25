import warnings

import torch

from kornia.filters.blur_pool import max_blur_pool2d as _max_blur_pool2d  # For doctest
from kornia.filters.blur_pool import MaxBlurPool2D

__all__ = ["max_blur_pool2d", "MaxBlurPool2d"]


class MaxBlurPool2d(MaxBlurPool2D):
    __doc__ = MaxBlurPool2D.__doc__

    def __init__(self, kernel_size: int, ceil_mode: bool = False) -> None:
        super(MaxBlurPool2d, self).__init__(kernel_size, stride=2, max_pool_size=2, ceil_mode=ceil_mode)
        warnings.warn(
            "`MaxBlurPool2d` is deprecated and will be removed after > 0.6. "
            "Please use `kornia.filters.MaxBlurPool2D instead.`",
            DeprecationWarning,
            stacklevel=2,
        )


def max_blur_pool2d(input: torch.Tensor, kernel_size: int, ceil_mode: bool = False) -> torch.Tensor:
    r"""Creates a module that computes pools and blurs and downsample a given feature map.

    See :class:`~kornia.contrib.MaxBlurPool2d` for details.
    """
    warnings.warn(
        "`max_blur_pool2d` is deprecated and will be removed after > 0.6. "
        "Please use `kornia.filters.max_blur_pool2d instead.`",
        DeprecationWarning,
        stacklevel=2,
    )
    return _max_blur_pool2d(input, kernel_size, stride=2, max_pool_size=2, ceil_mode=ceil_mode)
