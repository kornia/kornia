from typing import Any, Dict, Optional, cast

from torch import Tensor
from kornia.augmentation import random_generator as rg
from kornia.augmentation._2d.intensity.base import IntensityAugmentationBase2D
from kornia.enhance import add_snow


class RandomSnow(IntensityAugmentationBase2D):
    """Randomly add snow to an image.

    Args:
        snow_coef_low: minimum possible value of snow added;
          recommended interval - [0, 1], should always be positive.
        snow_coef_high: maximum possible value of snow added;
          recommended interval - [0, 1], should always be positive.
        brightness_coef: regulates the overall amount of snow on the image;
          the larger, the more snow there is.
        same_on_batch: apply the same transformation across the batch.
        p: probability of applying the transformation.
        keepdim: whether to keep the output shape the same as input ``True`` or broadcast it
          to the batch form ``False``.

    Note:
        Input tensor must be float and normalized into [0, 1].

    Examples:
        >>> import torch
        >>> rng = torch.manual_seed(0)
        >>> inp = torch.rand(1, 3, 5, 5)
        >>> aug = RandomSnow(0, 0, 0)
        >>> ((inp == aug(inp)).double()).all()
        tensor(True)

        >>> rng = torch.manual_seed(0)
        >>> inp = torch.rand(1, 3, 5, 5)
        >>> inp
        tensor([[[[0.4963, 0.7682, 0.0885, 0.1320, 0.3074],
                  [0.6341, 0.4901, 0.8964, 0.4556, 0.6323],
                  [0.3489, 0.4017, 0.0223, 0.1689, 0.2939],
                  [0.5185, 0.6977, 0.8000, 0.1610, 0.2823],
                  [0.6816, 0.9152, 0.3971, 0.8742, 0.4194]],
        <BLANKLINE>
                 [[0.5529, 0.9527, 0.0362, 0.1852, 0.3734],
                  [0.3051, 0.9320, 0.1759, 0.2698, 0.1507],
                  [0.0317, 0.2081, 0.9298, 0.7231, 0.7423],
                  [0.5263, 0.2437, 0.5846, 0.0332, 0.1387],
                  [0.2422, 0.8155, 0.7932, 0.2783, 0.4820]],
        <BLANKLINE>
                 [[0.8198, 0.9971, 0.6984, 0.5675, 0.8352],
                  [0.2056, 0.5932, 0.1123, 0.1535, 0.2417],
                  [0.7262, 0.7011, 0.2038, 0.6511, 0.7745],
                  [0.4369, 0.5191, 0.6159, 0.8102, 0.9801],
                  [0.1147, 0.3168, 0.6965, 0.9143, 0.9351]]]])
        >>> aug = RandomSnow(0, 0.2, 0.5, p=1.)
        >>> aug(inp)
        tensor([[[[0.4963, 0.7682, 0.0442, 0.0660, 0.3074],
                  [0.6341, 0.4901, 0.8964, 0.2278, 0.6323],
                  [0.1744, 0.4017, 0.0223, 0.1689, 0.2939],
                  [0.5185, 0.6977, 0.8000, 0.1610, 0.2823],
                  [0.6816, 0.9152, 0.3971, 0.8742, 0.4194]],
        <BLANKLINE>
                 [[0.5529, 0.9527, 0.0181, 0.0926, 0.3734],
                  [0.3051, 0.9320, 0.1759, 0.1349, 0.1507],
                  [0.0159, 0.2081, 0.9298, 0.7231, 0.7423],
                  [0.5263, 0.2437, 0.5846, 0.0332, 0.1387],
                  [0.2422, 0.8155, 0.7932, 0.2783, 0.4820]],
        <BLANKLINE>
                 [[0.8198, 0.9971, 0.3492, 0.2838, 0.8352],
                  [0.2056, 0.5932, 0.1123, 0.0767, 0.2417],
                  [0.3631, 0.7011, 0.2038, 0.6511, 0.7745],
                  [0.4369, 0.5191, 0.6159, 0.8102, 0.9801],
                  [0.1147, 0.3168, 0.6965, 0.9143, 0.9351]]]])
    """

    def __init__(
        self,
        snow_coef_low: float = 0,
        snow_coef_high: float = 1,
        brightness_coef: float = 2.5,
        same_on_batch: bool = False,
        p: float = 0.5,
        keepdim: bool = False,
        return_transform: Optional[bool] = None,
    ) -> None:
        super().__init__(p=p, return_transform=return_transform, same_on_batch=same_on_batch, keepdim=keepdim)
        self._param_generator = cast(
            rg.PlainUniformGenerator,
            rg.PlainUniformGenerator((snow_coef_low, "snow_point", (
                snow_coef_low + snow_coef_high) / 2, (snow_coef_low, snow_coef_high)))
        )
        self.brightness_coef = brightness_coef

    def apply_transform(
        self, inp: Tensor, params: Dict[str, Tensor], flags: Dict[str, Any], transform: Optional[Tensor] = None
    ) -> Tensor:
        return add_snow(inp, params['snow_point'], self.brightness_coef)
