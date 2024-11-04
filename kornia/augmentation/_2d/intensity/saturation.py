from typing import Any, Dict, Optional, Tuple

from kornia.augmentation import random_generator as rg
from kornia.augmentation._2d.intensity.base import IntensityAugmentationBase2D
from kornia.augmentation.utils import _range_bound
from kornia.core import Tensor
from kornia.enhance.adjust import adjust_saturation


class RandomSaturation(IntensityAugmentationBase2D):
    r"""Apply a random transformation to the saturation of a tensor image.

    This implementation aligns PIL. Hence, the output is close to TorchVision.

    .. image:: _static/img/RandomSaturation.png

    Args:
        p: probability of applying the transformation.
        saturation: the saturation factor to apply.
        same_on_batch: apply the same transformation across the batch.
        keepdim: whether to keep the output shape the same as input (True) or broadcast it
                 to the batch form (False).
    Shape:
        - Input: :math:`(C, H, W)` or :math:`(B, C, H, W)`, Optional: :math:`(B, 3, 3)`
        - Output: :math:`(B, C, H, W)`

    .. note::
        This function internally uses :func:`kornia.enhance.adjust_saturation`

    Examples:
        >>> rng = torch.manual_seed(0)
        >>> inputs = torch.rand(1, 3, 3, 3)
        >>> aug = RandomSaturation(saturation = (0.5,2.),p=1.)
        >>> aug(inputs)
        tensor([[[[0.5569, 0.7682, 0.3529],
                  [0.4811, 0.3474, 0.7411],
                  [0.5028, 0.8964, 0.6772]],
        <BLANKLINE>
                 [[0.6323, 0.5358, 0.5265],
                  [0.4203, 0.2706, 0.5525],
                  [0.5185, 0.7863, 0.8681]],
        <BLANKLINE>
                 [[0.3711, 0.4989, 0.6816],
                  [0.9152, 0.3971, 0.8742],
                  [0.4636, 0.7060, 0.9527]]]])

    To apply the exact augmenation again, you may take the advantage of the previous parameter state:

        >>> input = torch.rand(1, 3, 32, 32)
        >>> aug = RandomSaturation((0.8,1.2), p=1.)
        >>> (aug(input) == aug(input, params=aug._params)).all()
        tensor(True)
    """

    def __init__(
        self,
        saturation: Tuple[float, float] = (1.0, 1.0),
        same_on_batch: bool = False,
        p: float = 1.0,
        keepdim: bool = False,
    ) -> None:
        super().__init__(p=p, same_on_batch=same_on_batch, keepdim=keepdim)
        self.saturation: Tensor = _range_bound(saturation, "saturation", center=1.0)
        self._param_generator = rg.PlainUniformGenerator((self.saturation, "saturation_factor", None, None))

    def apply_transform(
        self, input: Tensor, params: Dict[str, Tensor], flags: Dict[str, Any], transform: Optional[Tensor] = None
    ) -> Tensor:
        saturation_factor = params["saturation_factor"].to(input)
        return adjust_saturation(input, saturation_factor)
