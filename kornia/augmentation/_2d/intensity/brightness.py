from typing import Any, Dict, Optional, Tuple

from kornia.augmentation import random_generator as rg
from kornia.augmentation._2d.intensity.base import IntensityAugmentationBase2D
from kornia.augmentation.utils import _range_bound
from kornia.core import Tensor
from kornia.enhance.adjust import adjust_brightness


class RandomBrightness(IntensityAugmentationBase2D):
    r"""Apply a random transformation to the brightness of a tensor image.

    This implementation aligns PIL. Hence, the output is close to TorchVision.

    .. image:: _static/img/RandomBrightness.png

    Args:
        brightness: the brightness factor to apply
        clip_output: if true clip output
        silence_instantiation_warning: if True, silence the warning at instantiation.
        same_on_batch: apply the same transformation across the batch.
        p: probability of applying the transformation.
        keepdim: whether to keep the output shape the same as input (True) or broadcast it
                 to the batch form (False).
    Shape:
        - Input: :math:`(C, H, W)` or :math:`(B, C, H, W)`, Optional: :math:`(B, 3, 3)`
        - Output: :math:`(B, C, H, W)`

    .. note::
        This function internally uses :func:`kornia.enhance.adjust_brightness`

    Examples:
        >>> rng = torch.manual_seed(0)
        >>> inputs = torch.rand(1, 3, 3, 3)
        >>> aug = RandomBrightness(brightness = (0.5,2.),p=1.)
        >>> aug(inputs)
        tensor([[[[0.0505, 0.3225, 0.0000],
                  [0.0000, 0.0000, 0.1883],
                  [0.0443, 0.4507, 0.0099]],
        <BLANKLINE>
                 [[0.1866, 0.0000, 0.0000],
                  [0.0000, 0.0000, 0.0000],
                  [0.0728, 0.2519, 0.3543]],
        <BLANKLINE>
                 [[0.0000, 0.0000, 0.2359],
                  [0.4694, 0.0000, 0.4284],
                  [0.0000, 0.1072, 0.5070]]]])

    To apply the exact augmenation again, you may take the advantage of the previous parameter state:

        >>> input = torch.rand(1, 3, 32, 32)
        >>> aug = RandomBrightness((0.8,1.2), p=1.)
        >>> (aug(input) == aug(input, params=aug._params)).all()
        tensor(True)
    """

    def __init__(
        self,
        brightness: Tuple[float, float] = (1.0, 1.0),
        clip_output: bool = True,
        same_on_batch: bool = False,
        p: float = 1.0,
        keepdim: bool = False,
    ) -> None:
        super().__init__(p=p, same_on_batch=same_on_batch, keepdim=keepdim)
        self.brightness: Tensor = _range_bound(brightness, "brightness", center=1.0, bounds=(0.0, 2.0))
        self._param_generator = rg.PlainUniformGenerator((self.brightness, "brightness_factor", None, None))

        self.clip_output = clip_output

    def apply_transform(
        self, input: Tensor, params: Dict[str, Tensor], flags: Dict[str, Any], transform: Optional[Tensor] = None
    ) -> Tensor:
        brightness_factor = params["brightness_factor"].to(input)
        return adjust_brightness(input, brightness_factor - 1, self.clip_output)
