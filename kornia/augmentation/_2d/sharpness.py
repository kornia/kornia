from typing import Dict, Optional, Tuple, Union, cast

import torch

from kornia.augmentation.base_2d import IntensityAugmentationBase2D
from kornia.enhance import sharpness

from .. import random_generator as rg


class RandomSharpness(IntensityAugmentationBase2D):
    r"""Sharpen given tensor image or a batch of tensor images randomly.

    .. image:: _static/img/RandomSharpness.png

    Args:
        p: probability of applying the transformation.
        sharpness: factor of sharpness strength. Must be above 0.
        same_on_batch: apply the same transformation across the batch.
        return_transform: if ``True`` return the matrix describing the transformation applied to each
            input tensor. If ``False`` and the input is a tuple the applied transformation won't be concatenated.
        keepdim: whether to keep the output shape the same as input (True) or broadcast it
                 to the batch form (False).

    Shape:
        - Input: :math:`(C, H, W)` or :math:`(B, C, H, W)`, Optional: :math:`(B, 3, 3)`
        - Output: :math:`(B, C, H, W)`

    .. note::
        This function internally uses :func:`kornia.enhance.sharpness`.

    Examples:
        >>> rng = torch.manual_seed(0)
        >>> input = torch.rand(1, 1, 5, 5)
        >>> sharpness = RandomSharpness(1., p=1.)
        >>> sharpness(input)
        tensor([[[[0.4963, 0.7682, 0.0885, 0.1320, 0.3074],
                  [0.6341, 0.4810, 0.7367, 0.4177, 0.6323],
                  [0.3489, 0.4428, 0.1562, 0.2443, 0.2939],
                  [0.5185, 0.6462, 0.7050, 0.2288, 0.2823],
                  [0.6816, 0.9152, 0.3971, 0.8742, 0.4194]]]])

    To apply the exact augmenation again, you may take the advantage of the previous parameter state:
        >>> input = torch.randn(1, 3, 32, 32)
        >>> aug = RandomSharpness(1., p=1.)
        >>> (aug(input) == aug(input, params=aug._params)).all()
        tensor(True)
    """

    def __init__(
        self,
        sharpness: Union[torch.Tensor, float, Tuple[float, float], torch.Tensor] = 0.5,
        same_on_batch: bool = False,
        return_transform: bool = False,
        p: float = 0.5,
        keepdim: bool = False,
    ) -> None:
        super().__init__(p=p, return_transform=return_transform, same_on_batch=same_on_batch, keepdim=keepdim)
        self._param_generator = cast(
            rg.PlainUniformGenerator, rg.PlainUniformGenerator((sharpness, "sharpness", 0.0, (0, float("inf"))))
        )

    def apply_transform(
        self, input: torch.Tensor, params: Dict[str, torch.Tensor], transform: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        factor = params["sharpness"]
        return sharpness(input, factor)
