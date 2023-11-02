from typing import Any, Dict, Optional

from kornia.augmentation import random_generator as rg
from kornia.augmentation._2d.intensity.base import IntensityAugmentationBase2D
from kornia.core import Tensor
from kornia.enhance import shift_rgb


class RandomRGBShift(IntensityAugmentationBase2D):
    """Randomly shift each channel of an image.

    Args:
        r_shift_limit: maximum value up to which the shift value can be generated for red channel;
          recommended interval - [0, 1], should always be positive
        g_shift_limit: maximum value up to which the shift value can be generated for green channel;
          recommended interval - [0, 1], should always be positive
        b_shift_limit: maximum value up to which the shift value can be generated for blue channel;
          recommended interval - [0, 1], should always be positive
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
        >>> aug = RandomRGBShift(0, 0, 0)
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
        >>> aug = RandomRGBShift(p=1.)
        >>> aug(inp)
        tensor([[[[0.9374, 1.0000, 0.5297, 0.5732, 0.7486],
                  [1.0000, 0.9313, 1.0000, 0.8968, 1.0000],
                  [0.7901, 0.8429, 0.4635, 0.6100, 0.7351],
                  [0.9597, 1.0000, 1.0000, 0.6022, 0.7234],
                  [1.0000, 1.0000, 0.8383, 1.0000, 0.8606]],
        <BLANKLINE>
                 [[0.6524, 1.0000, 0.1357, 0.2847, 0.4729],
                  [0.4046, 1.0000, 0.2754, 0.3693, 0.2502],
                  [0.1312, 0.3076, 1.0000, 0.8226, 0.8418],
                  [0.6258, 0.3432, 0.6841, 0.1327, 0.2382],
                  [0.3417, 0.9150, 0.8927, 0.3778, 0.5815]],
        <BLANKLINE>
                 [[0.3850, 0.5623, 0.2636, 0.1328, 0.4005],
                  [0.0000, 0.1584, 0.0000, 0.0000, 0.0000],
                  [0.2914, 0.2663, 0.0000, 0.2163, 0.3397],
                  [0.0021, 0.0843, 0.1811, 0.3754, 0.5453],
                  [0.0000, 0.0000, 0.2617, 0.4795, 0.5003]]]])
    """

    def __init__(
        self,
        r_shift_limit: float = 0.5,
        g_shift_limit: float = 0.5,
        b_shift_limit: float = 0.5,
        same_on_batch: bool = False,
        p: float = 0.5,
        keepdim: bool = False,
    ) -> None:
        super().__init__(p=p, same_on_batch=same_on_batch, keepdim=keepdim)
        self._param_generator = rg.PlainUniformGenerator(
            (r_shift_limit, "r_shift", 0, (-r_shift_limit, r_shift_limit)),
            (g_shift_limit, "g_shift", 0, (-g_shift_limit, g_shift_limit)),
            (b_shift_limit, "b_shift", 0, (-b_shift_limit, b_shift_limit)),
        )

    def apply_transform(
        self, inp: Tensor, params: Dict[str, Tensor], flags: Dict[str, Any], transform: Optional[Tensor] = None
    ) -> Tensor:
        return shift_rgb(inp, params["r_shift"], params["g_shift"], params["b_shift"])
