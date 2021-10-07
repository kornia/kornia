from typing import Callable

import torch
import torch.nn as nn


class Lambda(nn.Module):
    """Applies user-defined lambda as a transform.

    Args:
        func: Callable function.

    Returns:
        A torch.Tensor which is the output of the user-defined lambda
    """

    def __init__(self, func: Callable) -> None:
        super().__init__()
        if not callable(func):
            raise TypeError(f"Argument lambd should be callable, got {repr(type(func).__name__)}")

        self.func = func

    def forward(self, img: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return self.func(img, *args, **kwargs)
