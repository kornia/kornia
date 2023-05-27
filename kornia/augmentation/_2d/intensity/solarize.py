from typing import Any, Dict, List, Optional, Tuple, Union

from kornia.augmentation import random_generator as rg
from kornia.augmentation._2d.intensity.base import IntensityAugmentationBase2D
from kornia.core import Tensor
from kornia.enhance import solarize


class RandomSolarize(IntensityAugmentationBase2D):
    r"""Solarize given tensor image or a batch of tensor images randomly.

    .. image:: _static/img/RandomSolarize.png

    Args:
        p: probability of applying the transformation.
        thresholds:
            If float x, threshold will be generated from (0.5 - x, 0.5 + x).
            If tuple (x, y), threshold will be generated from (x, y).
        additions:
            If float x, addition will be generated from (-x, x).
            If tuple (x, y), addition will be generated from (x, y).
        same_on_batch: apply the same transformation across the batch.
        keepdim: whether to keep the output shape the same as input (True) or broadcast it
                 to the batch form (False).

    Shape:
        - Input: :math:`(C, H, W)` or :math:`(B, C, H, W)`, Optional: :math:`(B, 3, 3)`
        - Output: :math:`(B, C, H, W)`

    .. note::
        This function internally uses :func:`kornia.enhance.solarize`.

    Examples:
        >>> rng = torch.manual_seed(0)
        >>> input = torch.rand(1, 1, 5, 5)
        >>> solarize = RandomSolarize(0.1, 0.1, p=1.)
        >>> solarize(input)
        tensor([[[[0.4132, 0.1412, 0.1790, 0.2226, 0.3980],
                  [0.2754, 0.4194, 0.0130, 0.4538, 0.2771],
                  [0.4394, 0.4923, 0.1129, 0.2594, 0.3844],
                  [0.3909, 0.2118, 0.1094, 0.2516, 0.3728],
                  [0.2278, 0.0000, 0.4876, 0.0353, 0.5100]]]])

    To apply the exact augmenation again, you may take the advantage of the previous parameter state:
        >>> input = torch.randn(1, 3, 32, 32)
        >>> aug = RandomSolarize(0.1, 0.1, p=1.)
        >>> (aug(input) == aug(input, params=aug._params)).all()
        tensor(True)
    """

    def __init__(
        self,
        thresholds: Union[Tensor, float, Tuple[float, float], List[float]] = 0.1,
        additions: Union[Tensor, float, Tuple[float, float], List[float]] = 0.1,
        same_on_batch: bool = False,
        p: float = 0.5,
        keepdim: bool = False,
    ) -> None:
        super().__init__(p=p, same_on_batch=same_on_batch, keepdim=keepdim)
        self._param_generator = rg.PlainUniformGenerator(
            (thresholds, "thresholds", 0.5, (0.0, 1.0)), (additions, "additions", 0.0, (-0.5, 0.5))
        )

    def apply_transform(
        self, input: Tensor, params: Dict[str, Tensor], flags: Dict[str, Any], transform: Optional[Tensor] = None
    ) -> Tensor:
        thresholds = params["thresholds"]
        additions: Optional[Tensor]
        if "additions" in params:
            additions = params["additions"]
        else:
            additions = None
        return solarize(input, thresholds, additions)
