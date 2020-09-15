"""Module containing functionals for intensity normalisation."""

from typing import Union

import torch
import torch.nn as nn


__all__ = [
    "normalize",
    "normalize_min_max",
    "denormalize",
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
        return normalize(input, self.mean, self.std)

    def __repr__(self):
        repr = "(mean={0}, std={1})".format(self.mean, self.std)
        return self.__class__.__name__ + repr


def normalize(
    data: torch.Tensor, mean: Union[torch.Tensor, float], std: Union[torch.Tensor, float]
) -> torch.Tensor:
    r"""Normalise the image with channel-wise mean and standard deviation.

    See :class:`~kornia.color.Normalize` for details.

    Args:
        data (torch.Tensor): The image tensor to be normalised.
        mean (torch.Tensor or float): Mean for each channel.
        std (torch.Tensor or float): Standard deviations for each channel.

    Returns:
        torch.Tensor: The normalised image tensor.

    """
    if isinstance(mean, float):
        mean = torch.tensor([mean])  # prevent 0 sized tensors

    if isinstance(std, float):
        std = torch.tensor([std])  # prevent 0 sized tensors

    if not torch.is_tensor(data):
        raise TypeError("data should be a tensor. Got {}".format(type(data)))

    if not torch.is_tensor(mean):
        raise TypeError("mean should be a tensor or a float. Got {}".format(type(mean)))

    if not torch.is_tensor(std):
        raise TypeError("std should be a tensor or float. Got {}".format(type(std)))

    # Allow broadcast on channel dimension
    if mean.shape and mean.shape[0] != 1:
        if mean.shape[0] != data.shape[-3] and mean.shape[:2] != data.shape[:2]:
            raise ValueError("mean length and number of channels do not match")

    # Allow broadcast on channel dimension
    if std.shape and std.shape[0] != 1:
        if std.shape[0] != data.shape[-3] and std.shape[:2] != data.shape[:2]:
            raise ValueError("std length and number of channels do not match")

    if mean.shape:
        mean = mean[..., :, None, None].to(data.device)
    if std.shape:
        std = std[..., :, None, None].to(data.device)

    out: torch.Tensor = (data - mean) / std

    return out


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
        return denormalize(input, self.mean, self.std)

    def __repr__(self):
        repr = "(mean={0}, std={1})".format(self.mean, self.std)
        return self.__class__.__name__ + repr


def denormalize(
    data: torch.Tensor, mean: Union[torch.Tensor, float], std: Union[torch.Tensor, float]
) -> torch.Tensor:
    r"""Denormalize the image given channel-wise mean and standard deviation.

    See :class:`~kornia.color.Normalize` for details.

    Args:
        data (torch.Tensor): The image tensor to be normalised.
        mean (torch.Tensor or float): Mean for each channel.
        std (torch.Tensor or float): Standard deviations for each channel.

    Returns:
        torch.Tensor: The normalised image tensor.

    """
    if isinstance(mean, float):
        mean = torch.tensor([mean])  # prevent 0 sized tensors

    if isinstance(std, float):
        std = torch.tensor([std])  # prevent 0 sized tensors

    if not torch.is_tensor(data):
        raise TypeError("data should be a tensor. Got {}".format(type(data)))

    if not torch.is_tensor(mean):
        raise TypeError("mean should be a tensor or a float. Got {}".format(type(mean)))

    if not torch.is_tensor(std):
        raise TypeError("std should be a tensor or float. Got {}".format(type(std)))

    # Allow broadcast on channel dimension
    if mean.shape and mean.shape[0] != 1:
        if mean.shape[0] != data.shape[-3] and mean.shape[:2] != data.shape[:2]:
            raise ValueError("mean length and number of channels do not match")

    # Allow broadcast on channel dimension
    if std.shape and std.shape[0] != 1:
        if std.shape[0] != data.shape[-3] and std.shape[:2] != data.shape[:2]:
            raise ValueError("std length and number of channels do not match")

    if mean.shape:
        mean = mean[..., :, None, None].to(data.device)
    if std.shape:
        std = std[..., :, None, None].to(data.device)

    out: torch.Tensor = (data * std) + mean

    return out


def normalize_min_max(x: torch.Tensor, a: float = 0., b: float = 1., eps: float = 1e-6) -> torch.Tensor:
    r"""Normalise an image tensor by MinMax and re-scales the value between a range.

    The data is normalised using the following formulation:

    .. math::
        y_i = (b - a) * \frac{x_i - \text{min}(x)}{\text{max}(x) - \text{min}(x)} + a

    Args:
        x (torch.Tensor): The image tensor to be normalised with shape :math:`(B, C, H, W)`.
        a (float): The minimum value for the new range. Default: 0.
        b (float): The maximum value for the new range. Default: 1.
        eps (float): Float number to avoid zero division. Default: 1e-6.

    Returns:
        torch.Tensor: The normalised image tensor with same shape as input :math:`(B, C, H, W)`.

    """
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"data should be a tensor. Got: {type(x)}.")

    if not isinstance(a, float):
        raise TypeError(f"'a' should be a float. Got: {type(a)}.")

    if not isinstance(b, float):
        raise TypeError(f"'b' should be a float. Got: {type(b)}.")

    if len(x.shape) != 4:
        raise ValueError(f"Input shape must be a 4d tensor. Got: {x.shape}.")

    B, C, H, W = x.shape

    x_min: torch.Tensor = x.view(B, C, -1).min(-1)[0].view(B, C, 1, 1)
    x_max: torch.Tensor = x.view(B, C, -1).max(-1)[0].view(B, C, 1, 1)

    x_out: torch.Tensor = (
        (b - a) * (x - x_min) / (x_max - x_min + eps) + a
    )
    return x_out.expand_as(x)
