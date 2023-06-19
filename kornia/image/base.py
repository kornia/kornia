from __future__ import annotations

from dataclasses import dataclass

from kornia.core import Tensor


@dataclass
class ImageSize:
    r"""Data class to represent image shape.

    Args:
        height: image height.
        width: image width
    Example:
        >>> size = ImageSize(3, 4)
        >>> size.height
        3
        >>> size.width
        4
    """
    height: int | Tensor
    width: int | Tensor
