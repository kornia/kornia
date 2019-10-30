import torch
import torch.nn as nn
from typing import Union


class Normalize(nn.Module):
    r"""Normalize a tensor image or a batch of tensor images
    with mean and standard deviation. Input must be a tensor of shape (C, H, W)
    or a batch of tensors :math:`(*, C, H, W)`.

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
