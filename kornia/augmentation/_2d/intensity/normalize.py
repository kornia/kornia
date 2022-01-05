from typing import Dict, List, Optional, Tuple, Union

import torch

from kornia.augmentation._2d.intensity.base import IntensityAugmentationBase2D
from kornia.enhance import normalize


class Normalize(IntensityAugmentationBase2D):
    r"""Normalize tensor images with mean and standard deviation.

    .. math::
        \text{input[channel] = (input[channel] - mean[channel]) / std[channel]}

    Where `mean` is :math:`(M_1, ..., M_n)` and `std` :math:`(S_1, ..., S_n)` for `n` channels,

    Args:
        mean: Mean for each channel.
        std: Standard deviations for each channel.
        return_transform: if ``True`` return the matrix describing the transformation applied to each
                          input tensor. If ``False`` and the input is a tuple the applied transformation
                          won't be concatenated.
        p: probability of applying the transformation.
        keepdim: whether to keep the output shape the same as input (True) or broadcast it
                 to the batch form (False).

    Return:
        Normalised tensor with same size as input :math:`(*, C, H, W)`.

    .. note::
        This function internally uses :func:`kornia.enhance.normalize`.

    Examples:

        >>> norm = Normalize(mean=torch.zeros(4), std=torch.ones(4))
        >>> x = torch.rand(1, 4, 3, 3)
        >>> out = norm(x)
        >>> out.shape
        torch.Size([1, 4, 3, 3])
    """

    def __init__(
        self,
        mean: Union[torch.Tensor, Tuple[float], List[float], float],
        std: Union[torch.Tensor, Tuple[float], List[float], float],
        return_transform: bool = False,
        p: float = 1.0,
        keepdim: bool = False,
    ) -> None:
        super().__init__(p=p, return_transform=return_transform, same_on_batch=True, keepdim=keepdim)
        if isinstance(mean, float):
            mean = torch.tensor([mean])

        if isinstance(std, float):
            std = torch.tensor([std])

        if isinstance(mean, (tuple, list)):
            mean = torch.tensor(mean)

        if isinstance(std, (tuple, list)):
            std = torch.tensor(std)

        self.flags = dict(mean=mean, std=std)

    def apply_transform(
        self, input: torch.Tensor, params: Dict[str, torch.Tensor], transform: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        return normalize(input, self.flags["mean"], self.flags["std"])
