from __future__ import annotations

from typing import Any, Optional

import torch
from torch import Tensor

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
        mean: Tensor | tuple[float, ...] | list[float] | float,
        std: Tensor | tuple[float, ...] | list[float] | float,
        p: float = 1.0,
        keepdim: bool = False,
    ) -> None:
        super().__init__(p=p, same_on_batch=True, keepdim=keepdim)
        if isinstance(mean, (int, float)):
            mean = torch.tensor([mean])

        if isinstance(std, (int, float)):
            std = torch.tensor([std])

        if isinstance(mean, (tuple, list)):
            mean = torch.tensor(mean)

        if isinstance(std, (tuple, list)):
            std = torch.tensor(std)

        self.flags = {"mean": mean, "std": std}

    def apply_transform(
        self, input: Tensor, params: dict[str, Tensor], flags: dict[str, Any], transform: Optional[Tensor] = None
    ) -> Tensor:
        return normalize(input, flags["mean"], flags["std"])
