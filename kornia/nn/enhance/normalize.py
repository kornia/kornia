"""Module containing functionals for intensity normalisation."""

from typing import Union

import torch
import torch.nn as nn

import kornia


__all__ = [
    "Normalize",
    "Denormalize",
]


class Normalize(nn.Module):
    r"""Normalize a tensor image or a batch of tensor images with mean and standard deviation.

    Input must be a tensor of shape (C, H, W) or a batch of tensors :math:`(*, C, H, W)`.

    Given mean: ``(M1,...,Mn)`` and std: ``(S1,..,Sn)`` for ``n`` channels,
    this transform will normalize each channel of the input ``torch.Tensor``
    i.e. ``input[channel] = (input[channel] - mean[channel]) / std[channel]``

    Args:
        mean (torch.Tensor or float): Mean for each channel.
        std (torch.Tensor or float): Standard deviations for each channel.

    """

    def __init__(self, mean: Union[torch.Tensor, float], std: Union[torch.Tensor, float]) -> None:

        super(Normalize, self).__init__()

        self.mean = mean
        self.std = std

    def forward(self, input: torch.Tensor) -> torch.Tensor:  # type: ignore
        """Normalises an input tensor by the mean and standard deviation.

        Args:
            input: image tensor of size (*, H, W).

        Returns:
            normalised tensor with same size as input (*, H, W).

        """
        return kornia.enhance.normalize(input, self.mean, self.std)

    def __repr__(self):
        repr = "(mean={0}, std={1})".format(self.mean, self.std)
        return self.__class__.__name__ + repr


class Denormalize(nn.Module):
    r"""Denormalize a tensor image or a batch of tensor images.

    Input must be a tensor of shape (C, H, W) or a batch of tensors :math:`(*, C, H, W)`.

    Given mean: ``(M1,...,Mn)`` and std: ``(S1,..,Sn)`` for ``n`` channels,
    this transform will denormalize each channel of the input ``torch.Tensor``
    i.e. ``input[channel] = (input[channel] * std[channel]) + mean[channel]``

    Args:
        mean (torch.Tensor or float): Mean for each channel.
        std (torch.Tensor or float): Standard deviations for each channel.

    """

    def __init__(self, mean: Union[torch.Tensor, float], std: Union[torch.Tensor, float]) -> None:

        super(Denormalize, self).__init__()

        self.mean = mean
        self.std = std

    def forward(self, input: torch.Tensor) -> torch.Tensor:  # type: ignore
        """Denormalises an input tensor by the mean and standard deviation.

        Args:
            input: image tensor of size (*, H, W).

        Returns:
            normalised tensor with same size as input (*, H, W).

        """
        return kornia.enhance.denormalize(input, self.mean, self.std)

    def __repr__(self):
        repr = "(mean={0}, std={1})".format(self.mean, self.std)
        return self.__class__.__name__ + repr
