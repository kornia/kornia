from typing import Any, Dict, Optional, Tuple

from kornia.augmentation import random_generator as rg
from kornia.augmentation._2d.intensity.base import IntensityAugmentationBase2D
from kornia.augmentation.utils import _range_bound
from kornia.constants import pi
from kornia.core import Tensor
from kornia.enhance.adjust import adjust_hue


class RandomHue(IntensityAugmentationBase2D):
    r"""Apply a random transformation to the hue of a tensor image.

    This implementation aligns PIL. Hence, the output is close to TorchVision.

    .. image:: _static/img/RandomHue.png

    Args:
        hue: the saturation factor to apply.
        same_on_batch: apply the same transformation across the batch.
        p: probability of applying the transformation.
        keepdim: whether to keep the output shape the same as input (True) or broadcast it
                 to the batch form (False).
    Shape:
        - Input: :math:`(C, H, W)` or :math:`(B, C, H, W)`, Optional: :math:`(B, 3, 3)`
        - Output: :math:`(B, C, H, W)`

    .. note::
        This function internally uses :func:`kornia.enhance.adjust_hue`

    Examples:
        >>> rng = torch.manual_seed(0)
        >>> inputs = torch.rand(1, 3, 3, 3)
        >>> aug = RandomHue(hue = (-0.5,0.5),p=1.)
        >>> aug(inputs)
        tensor([[[[0.3993, 0.2823, 0.6816],
                  [0.6117, 0.2090, 0.4081],
                  [0.4693, 0.5529, 0.9527]],
        <BLANKLINE>
                 [[0.1610, 0.5962, 0.4971],
                  [0.9152, 0.3971, 0.8742],
                  [0.4194, 0.6771, 0.7162]],
        <BLANKLINE>
                 [[0.6323, 0.7682, 0.0885],
                  [0.0223, 0.1689, 0.2939],
                  [0.5185, 0.8964, 0.4556]]]])

    To apply the exact augmenation again, you may take the advantage of the previous parameter state:

        >>> input = torch.rand(1, 3, 32, 32)
        >>> aug = RandomHue((-0.2,0.2), p=1.)
        >>> (aug(input) == aug(input, params=aug._params)).all()
        tensor(True)
    """

    def __init__(
        self, hue: Tuple[float, float] = (0.0, 0.0), same_on_batch: bool = False, p: float = 1.0, keepdim: bool = False
    ) -> None:
        super().__init__(p=p, same_on_batch=same_on_batch, keepdim=keepdim)
        self.hue: Tensor = _range_bound(hue, "hue", bounds=(-0.5, 0.5))
        self._param_generator = rg.PlainUniformGenerator((self.hue, "hue_factor", None, None))

    def apply_transform(
        self, input: Tensor, params: Dict[str, Tensor], flags: Dict[str, Any], transform: Optional[Tensor] = None
    ) -> Tensor:
        hue_factor = params["hue_factor"].to(input)
        return adjust_hue(input, hue_factor * 2 * pi)
