from typing import Callable

import torch
import torch.nn as nn


class Lambda(nn.Module):
    """Applies user-defined lambda as a transform.

    Args:
        func: Callable function.

    Returns:
        The output of the user-defined lambda.

    Example:
        >>> import kornia
        >>> x = torch.rand(1, 3, 5, 5)
        >>> f = Lambda(lambda x: kornia.color.rgb_to_grayscale(x))
        >>> f(x).shape
        torch.Size([1, 1, 5, 5])
    """

    def __init__(self, func: Callable) -> None:
        super().__init__()
        if not callable(func):
            raise TypeError(f"Argument lambd should be callable, got {repr(type(func).__name__)}")

        self.func = func

    def forward(self, img: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return self.func(img, *args, **kwargs)
