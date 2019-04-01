from typing import Optional

import numpy as np

import torch
import torch.nn as nn


class ToTensor(nn.Module):
    r"""Converts a ``numpy.ndarray`` to tensor.

    Converts a numpy.ndarray (*, H, W, C) in the range
    [0, 255] to a torch.Tensor of shape (*, C, H, W).
    """
    def __init__(self) -> None:
        super(ToTensor, self).__init__()

    def forward(self, input: np.ndarray) -> torch.Tensor:
        r"""
        Args:
            input (np.ndarray): Image to be converted to tensor.

        Returns:
            torch.Tensor: Converted image.
        """
        return to_tensor(input)

    def __repr__(self):
        return self.__class__.__name__ + '()'


def to_tensor(input: np.ndarray) -> torch.Tensor:
    r"""Conver a ``numpy.ndarray`` to tensor.

    See ``~torchgeometry.augmentation.ToTensor`` for more details.

    Args:
        input (np.ndarray): Image to be converted to tensor.

    Returns:
        torch.Tensor: Converted image.
    """
    if not type(input) == np.ndarray:
        raise TypeError("Input type is not a numpy.ndarray. Got {}".format(
            type(input)))

    img: torch.Tensor = torch.from_numpy(input)

    if len(input.shape) == 2:
        img = torch.unsqueeze(img, dim=-1)
    elif len(input.shape) < 2:
        raise ValueError("Input size must be a three dimensional array")

    img = img.transpose(-2,-1).transpose(-3, -2)
    return img.contiguous()
