from typing import Tuple, List

import torch
import torch.nn as nn


__all__ = [
    "Histogram",
]


class Histogram(nn.Module):
    r"""
    """

    def __init__(self, alpha: float, beta: float, gamma: float) -> None:
        super(Histogram, self).__init__()
        raise NotImplementedError()

    def forward(self, image: torch.Tensor) -> torch.Tensor:  # type: ignore
        return image
