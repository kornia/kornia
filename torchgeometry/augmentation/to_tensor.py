from typing import Optional

import torch
import torch.nn as nn

class ToTensor(nn.Module):
    r"""Converts a ``numpy.ndarray`` to tensor.

    Converts a numpy.ndarray (*, H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape (*, C x H x W).
    """
    def __init__(self) -> None:
        super(ToTensor, self).__init__()

    def forward(self, input):
        if not type(image) == np.ndarray:
            raise TypeError("Input type is not a numpy.ndarray. Got {}".format(
                type(image)))

        if len(image.shape) <= 2:
            raise ValueError("Input size must be a three dimensional array")

        img = torch.from_numpy(input)
        img = img.transpose(-2,-1).transpose(-3, -2)
        return img.contiguous()

    def __repr__(self):
        return self.__class__.__name__ + '()'
