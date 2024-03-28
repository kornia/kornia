from typing import Any, Dict, Optional, Tuple

from kornia.augmentation import random_generator as rg
from kornia.augmentation._2d.intensity.base import IntensityAugmentationBase2D
from kornia.augmentation.utils import _range_bound
from kornia.core import Tensor
from kornia.enhance.adjust import adjust_contrast


class RandomContrast(IntensityAugmentationBase2D):
    r"""Apply a random transformation to the contrast of a tensor image.

    This implementation aligns PIL. Hence, the output is close to TorchVision.

    .. image:: _static/img/RandomContrast.png

    Args:
        contrast: the contrast factor to apply.
        clip_output: if true clip output.
        same_on_batch: apply the same transformation across the batch.
        p: probability of applying the transformation.
        keepdim: whether to keep the output shape the same as input (True) or broadcast it
                 to the batch form (False).
    Shape:
        - Input: :math:`(C, H, W)` or :math:`(B, C, H, W)`, Optional: :math:`(B, 3, 3)`
        - Output: :math:`(B, C, H, W)`

    .. note::
        This function internally uses :func:`kornia.enhance.adjust_contrast`

    Examples:
        >>> rng = torch.manual_seed(0)
        >>> inputs = torch.rand(1, 3, 3, 3)
        >>> aug = RandomContrast(contrast = (0.5, 2.), p = 1.)
        >>> aug(inputs)
        tensor([[[[0.2750, 0.4258, 0.0490],
                  [0.0732, 0.1704, 0.3514],
                  [0.2716, 0.4969, 0.2525]],
        <BLANKLINE>
                 [[0.3505, 0.1934, 0.2227],
                  [0.0124, 0.0936, 0.1629],
                  [0.2874, 0.3867, 0.4434]],
        <BLANKLINE>
                 [[0.0893, 0.1564, 0.3778],
                  [0.5072, 0.2201, 0.4845],
                  [0.2325, 0.3064, 0.5281]]]])

    To apply the exact augmenation again, you may take the advantage of the previous parameter state:

        >>> input = torch.rand(1, 3, 32, 32)
        >>> aug = RandomContrast((0.8,1.2), p=1.)
        >>> (aug(input) == aug(input, params=aug._params)).all()
        tensor(True)
    """

    def __init__(
        self,
        contrast: Tuple[float, float] = (1.0, 1.0),
        clip_output: bool = True,
        same_on_batch: bool = False,
        p: float = 1.0,
        keepdim: bool = False,
    ) -> None:
        super().__init__(p=p, same_on_batch=same_on_batch, keepdim=keepdim)
        self.contrast: Tensor = _range_bound(contrast, "contrast", center=1.0)
        self._param_generator = rg.PlainUniformGenerator((self.contrast, "contrast_factor", None, None))

        self.clip_output = clip_output

    def apply_transform(
        self, input: Tensor, params: Dict[str, Tensor], flags: Dict[str, Any], transform: Optional[Tensor] = None
    ) -> Tensor:
        contrast_factor = params["contrast_factor"].to(input)
        return adjust_contrast(input, contrast_factor, self.clip_output)
