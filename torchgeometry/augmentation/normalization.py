import torch
import torch.nn as nn
from typing import Sequence


class Normalise(nn.Module):

    """Normalize a tensor image with mean and standard deviation.
    Given mean: ``(M1,...,Mn)`` and std: ``(S1,..,Sn)`` for ``n`` channels,
    this transform will normalize each channel of the input ``torch.*Tensor``
    i.e. ``input[channel] = (input[channel] - mean[channel]) / std[channel]``

    Args:
        mean (Sequence of floats): Mean for each channel.
        std (Sequence of floats): Standard deviation for each channel.
        inplace (bool): Perform normalisation in-place.
    """

    def __init__(self, mean: Sequence[float], std: Sequence[float],
                 inplace: bool = False) -> None:

        super(Normalise, self).__init__()

        self.mean = mean
        self.std = std
        self.inplace = inplace

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return normalise(input, self.mean, self.std, self.inplace)

    def __repr__(self):
        repr = '(mean={0}, std={1}, inplace={2})'.format(self.mean,
                                                         self.std, self.inplace)
        return self.__class__.__name__ + repr


def normalise(data: torch.Tensor, mean: Sequence[float],
              std: Sequence[float], inplace: bool = False) -> torch.Tensor:

    """Normalise the image with channel-wise mean and standard deviation.

    Args:
        data (Tensor): The image tensor to be normalised.
        mean (Sequence of float): Mean for each channel.
        std (Sequence of float): Standard deviations for each channel.
        inplace (bool): Perform normalisation in-place.

    Returns:
        Tensor: The normalised image tensor.
    """

    assert len(mean) == data.shape[-3], 'Mean lenght and number of channels do not match'
    assert len(std) == data.shape[-3], 'Std lenght and number of channels do not match'

    if not isinstance(data, torch.Tensor):
        raise TypeError('data is not a torch.Tensor')

    if inplace:
        data = data.clone()

    mean = torch.as_tensor(mean, device=data.device)[..., :, None, None]
    std = torch.as_tensor(std, device=data.device)[..., :, None, None]

    data.sub_(mean).div_(std)

    return data

# - denormalise
