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
    r"""Normalize a tensor image with mean and standard deviation.

    .. math::
        \text{input[channel] = (input[channel] - mean[channel]) / std[channel]}

    Where `mean` is :math:`(M_1, ..., M_n)` and `std` :math:`(S_1, ..., S_n)` for `n` channels,

    Args:
        mean (Union[torch.Tensor, float]): Mean for each channel.
        std (Union[torch.Tensor, float]): Standard deviations for each channel.

    Shape:
        - Input: Image tensor of size :math:`(*, C, H, W)`.
        - Output: Normalised tensor with same size as input :math:`(*, C, H, W)`.

    Examples:
        >>> x = torch.rand(1, 4, 3, 3)
        >>> out = Normalize(0.0, 255.)(x)
        >>> out.shape
        torch.Size([1, 4, 3, 3])

        >>> x = torch.rand(1, 4, 3, 3)
        >>> mean = torch.zeros(1, 4)
        >>> std = 255. * torch.ones(1, 4)
        >>> out = Normalize(mean, std)(x)
        >>> out.shape
        torch.Size([1, 4, 3, 3])
    """

    def __init__(self, mean: Union[torch.Tensor, float], std: Union[torch.Tensor, float]) -> None:

        super(Normalize, self).__init__()

        self.mean = mean
        self.std = std

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return normalize(input, self.mean, self.std)

    def __repr__(self):
        repr = "(mean={0}, std={1})".format(self.mean, self.std)
        return self.__class__.__name__ + repr


def normalize(
    data: torch.Tensor, mean: Union[torch.Tensor, float], std: Union[torch.Tensor, float]
) -> torch.Tensor:
    r"""Normalize a tensor image with mean and standard deviation.

    .. math::
        \text{input[channel] = (input[channel] - mean[channel]) / std[channel]}

    Where `mean` is :math:`(M_1, ..., M_n)` and `std` :math:`(S_1, ..., S_n)` for `n` channels,

    Args:
        data (torch.Tensor): Image tensor of size :math:`(*, C, H, W)`.
        mean (Union[torch.Tensor, float]): Mean for each channel.
        std (Union[torch.Tensor, float]): Standard deviations for each channel.

    Return:
        torch.Tensor: Normalised tensor with same size as input :math:`(*, C, H, W)`.

    Examples:
        >>> x = torch.rand(1, 4, 3, 3)
        >>> out = normalize(x, 0.0, 255.)
        >>> out.shape
        torch.Size([1, 4, 3, 3])

        >>> x = torch.rand(1, 4, 3, 3)
        >>> mean = torch.zeros(1, 4)
        >>> std = 255. * torch.ones(1, 4)
        >>> out = normalize(x, mean, std)
        >>> out.shape
        torch.Size([1, 4, 3, 3])
    """
    if isinstance(mean, float):
        mean = torch.tensor([mean])  # prevent 0 sized tensors

    if isinstance(std, float):
        std = torch.tensor([std])  # prevent 0 sized tensors

    if not isinstance(data, torch.Tensor):
        raise TypeError("data should be a tensor. Got {}".format(type(data)))

    if not isinstance(mean, torch.Tensor):
        raise TypeError("mean should be a tensor or a float. Got {}".format(type(mean)))

    if not isinstance(std, torch.Tensor):
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
    r"""Denormalize a tensor image with mean and standard deviation.

    .. math::
        \text{input[channel] = (input[channel] * mean[channel]) + std[channel]}

    Where `mean` is :math:`(M_1, ..., M_n)` and `std` :math:`(S_1, ..., S_n)` for `n` channels,

    Args:
        mean (Union[torch.Tensor, float]): Mean for each channel.
        std (Union[torch.Tensor, float]): Standard deviations for each channel.

    Shape:
        - Input: Image tensor of size :math:`(*, C, H, W)`.
        - Output: Denormalised tensor with same size as input :math:`(*, C, H, W)`.

    Examples:
        >>> x = torch.rand(1, 4, 3, 3)
        >>> out = Denormalize(0.0, 255.)(x)
        >>> out.shape
        torch.Size([1, 4, 3, 3])

        >>> x = torch.rand(1, 4, 3, 3)
        >>> mean = torch.zeros(1, 4)
        >>> std = 255. * torch.ones(1, 4)
        >>> out = Denormalize(mean, std)(x)
        >>> out.shape
        torch.Size([1, 4, 3, 3])
    """

    def __init__(self, mean: Union[torch.Tensor, float], std: Union[torch.Tensor, float]) -> None:

        super(Denormalize, self).__init__()

        self.mean = mean
        self.std = std

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return denormalize(input, self.mean, self.std)

    def __repr__(self):
        repr = "(mean={0}, std={1})".format(self.mean, self.std)
        return self.__class__.__name__ + repr


def denormalize(
    data: torch.Tensor, mean: Union[torch.Tensor, float], std: Union[torch.Tensor, float]
) -> torch.Tensor:
    r"""Denormalize a tensor image with mean and standard deviation.

    .. math::
        \text{input[channel] = (input[channel] * mean[channel]) + std[channel]}

    Where `mean` is :math:`(M_1, ..., M_n)` and `std` :math:`(S_1, ..., S_n)` for `n` channels,

    Args:
        input (torch.Tensor): Image tensor of size :math:`(*, C, H, W)`.
        mean (Union[torch.Tensor, float]): Mean for each channel.
        std (Union[torch.Tensor, float]): Standard deviations for each channel.

    Return:
        torch.Tensor: Denormalised tensor with same size as input :math:`(*, C, H, W)`.

    Examples:
        >>> x = torch.rand(1, 4, 3, 3)
        >>> out = denormalize(x, 0.0, 255.)
        >>> out.shape
        torch.Size([1, 4, 3, 3])

        >>> x = torch.rand(1, 4, 3, 3)
        >>> mean = torch.zeros(1, 4)
        >>> std = 255. * torch.ones(1, 4)
        >>> out = denormalize(x, mean, std)
        >>> out.shape
        torch.Size([1, 4, 3, 3])
    """

    if isinstance(mean, float):
        mean = torch.tensor([mean])  # prevent 0 sized tensors

    if isinstance(std, float):
        std = torch.tensor([std])  # prevent 0 sized tensors

    if not isinstance(data, torch.Tensor):
        raise TypeError("data should be a tensor. Got {}".format(type(data)))

    if not isinstance(mean, torch.Tensor):
        raise TypeError("mean should be a tensor or a float. Got {}".format(type(mean)))

    if not isinstance(std, torch.Tensor):
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


def normalize_min_max(x: torch.Tensor, min_val: float = 0., max_val: float = 1., eps: float = 1e-6) -> torch.Tensor:
    r"""Normalise an image tensor by MinMax and re-scales the value between a range.

    The data is normalised using the following formulation:

    .. math::
        y_i = (b - a) * \frac{x_i - \text{min}(x)}{\text{max}(x) - \text{min}(x)} + a

    where :math:`a` is :math:`\text{min_val}` and :math:`b` is :math:`\text{max_val}`.

    Args:
        x (torch.Tensor): The image tensor to be normalised with shape :math:`(B, C, H, W)`.
        min_val (float): The minimum value for the new range. Default: 0.
        max_val (float): The maximum value for the new range. Default: 1.
        eps (float): Float number to avoid zero division. Default: 1e-6.

    Returns:
        torch.Tensor: The normalised image tensor with same shape as input :math:`(B, C, H, W)`.

    Example:
        >>> x = torch.rand(1, 5, 3, 3)
        >>> x_norm = normalize_min_max(x, min_val=-1., max_val=1.)
        >>> x_norm.min()
        tensor(-1.)
        >>> x_norm.max()
        tensor(1.0000)
    """
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"data should be a tensor. Got: {type(x)}.")

    if not isinstance(min_val, float):
        raise TypeError(f"'min_val' should be a float. Got: {type(min_val)}.")

    if not isinstance(max_val, float):
        raise TypeError(f"'b' should be a float. Got: {type(max_val)}.")

    if len(x.shape) != 4:
        raise ValueError(f"Input shape must be a 4d tensor. Got: {x.shape}.")

    B, C, H, W = x.shape

    x_min: torch.Tensor = x.view(B, C, -1).min(-1)[0].view(B, C, 1, 1)
    x_max: torch.Tensor = x.view(B, C, -1).max(-1)[0].view(B, C, 1, 1)

    x_out: torch.Tensor = (
        (max_val - min_val) * (x - x_min) / (x_max - x_min + eps) + min_val
    )
    return x_out.expand_as(x)
