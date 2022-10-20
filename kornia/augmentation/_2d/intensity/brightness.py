from typing import Any, Dict, Optional, Tuple, cast

from torch import Tensor

from kornia.augmentation import random_generator as rg
from kornia.augmentation._2d.intensity.base import IntensityAugmentationBase2D
from kornia.augmentation.utils import  _range_bound

from kornia.enhance.adjust import adjust_brightness


class RandomBrightness(IntensityAugmentationBase2D):
    r"""Apply a random transformation to the brightness of a tensor image.

    This implementation aligns PIL. Hence, the output is close to TorchVision.

    .. image:: _static/img/RandomBrighness.png

    Args:
        p: probability of applying the transformation.
        brightness: the brightness factor to apply
        clip_output: if true clip output
        silence_instantiation_warning: if True, silence the warning at instantiation.
        same_on_batch: apply the same transformation across the batch.
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
        tensor([[[[0.9963, 1.0000, 0.5885],
                  [0.6320, 0.8074, 1.0000],
                  [0.9901, 1.0000, 0.9556]],
        <BLANKLINE>
                 [[1.0000, 0.8489, 0.9017],
                  [0.5223, 0.6689, 0.7939],
                  [1.0000, 1.0000, 1.0000]],
        <BLANKLINE>
                 [[0.6610, 0.7823, 1.0000],
                  [1.0000, 0.8971, 1.0000],
                  [0.9194, 1.0000, 1.0000]]]])

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
        return_transform: Optional[bool] = None,
    ) -> None:
        super().__init__(p=p, return_transform=return_transform, same_on_batch=same_on_batch, keepdim=keepdim)
        brightness  = _range_bound(brightness, 'brightness', center=1.0, bounds=(0, 2))
        self._param_generator = cast(
            rg.PlainUniformGenerator,
            rg.PlainUniformGenerator((brightness, "brightness_factor", None, None))
        )
        self.clip_output = clip_output

    def apply_transform(
        self, input: Tensor, params: Dict[str, Tensor], flags: Dict[str, Any], transform: Optional[Tensor] = None
    ) -> Tensor:
        brightness_factor = params["brightness_factor"].to(input)
        return adjust_brightness(input, brightness_factor-1, self.clip_output)
