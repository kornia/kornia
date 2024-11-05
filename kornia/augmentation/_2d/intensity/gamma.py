from typing import Any, Dict, Optional, Tuple

from kornia.augmentation import random_generator as rg
from kornia.augmentation._2d.intensity.base import IntensityAugmentationBase2D
from kornia.core import Tensor
from kornia.enhance.adjust import adjust_gamma


class RandomGamma(IntensityAugmentationBase2D):
    r"""Apply a random transformation to the gamma of a tensor image.

    This implementation aligns PIL. Hence, the output is close to TorchVision.

    .. image:: _static/img/RandomGamma.png

    Args:
        p: probability of applying the transformation.
        gamma: the gamma factor to apply.
        gain: the gain factor to apply.
        same_on_batch: apply the same transformation across the batch.
        keepdim: whether to keep the output shape the same as input (True) or broadcast it
                 to the batch form (False).
    Shape:
        - Input: :math:`(C, H, W)` or :math:`(B, C, H, W)`, Optional: :math:`(B, 3, 3)`
        - Output: :math:`(B, C, H, W)`

    .. note::
        This function internally uses :func:`kornia.enhance.adjust_gamma`

    Examples:
        >>> rng = torch.manual_seed(0)
        >>> inputs = torch.rand(1, 3, 3, 3)
        >>> aug = RandomGamma((0.5,2.),(1.5,1.5),p=1.)
        >>> aug(inputs)
        tensor([[[[1.0000, 1.0000, 0.3912],
                  [0.4883, 0.7801, 1.0000],
                  [1.0000, 1.0000, 0.9702]],
        <BLANKLINE>
                 [[1.0000, 0.8368, 0.9048],
                  [0.1824, 0.5597, 0.7609],
                  [1.0000, 1.0000, 1.0000]],
        <BLANKLINE>
                 [[0.5452, 0.7441, 1.0000],
                  [1.0000, 0.8990, 1.0000],
                  [0.9267, 1.0000, 1.0000]]]])

    To apply the exact augmenation again, you may take the advantage of the previous parameter state:
        >>> input = torch.rand(1, 3, 32, 32)
        >>> aug = RandomGamma((0.8,1.2), p=1.)
        >>> (aug(input) == aug(input, params=aug._params)).all()
        tensor(True)
    """

    def __init__(
        self,
        gamma: Tuple[float, float] = (1.0, 1.0),
        gain: Tuple[float, float] = (1.0, 1.0),
        same_on_batch: bool = False,
        p: float = 1.0,
        keepdim: bool = False,
    ) -> None:
        super().__init__(p=p, same_on_batch=same_on_batch, keepdim=keepdim)
        self._param_generator = rg.PlainUniformGenerator(
            (gamma, "gamma_factor", None, None), (gain, "gain_factor", None, None)
        )

    def apply_transform(
        self, input: Tensor, params: Dict[str, Tensor], flags: Dict[str, Any], transform: Optional[Tensor] = None
    ) -> Tensor:
        gamma_factor = params["gamma_factor"].to(input)
        gain_factor = params["gain_factor"].to(input)
        return adjust_gamma(input, gamma_factor, gain_factor)
