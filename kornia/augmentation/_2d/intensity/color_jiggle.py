from typing import Any, Dict, List, Optional, Tuple, Union

from kornia.augmentation import random_generator as rg
from kornia.augmentation._2d.intensity.base import IntensityAugmentationBase2D
from kornia.constants import pi
from kornia.core import Tensor
from kornia.enhance import adjust_brightness, adjust_contrast, adjust_hue, adjust_saturation


class ColorJiggle(IntensityAugmentationBase2D):
    r"""Apply a random transformation to the brightness, contrast, saturation and hue of a tensor image.

    .. image:: _static/img/ColorJiggle.png

    Args:
        p: probability of applying the transformation.
        brightness: The brightness factor to apply.
        contrast: The contrast factor to apply.
        saturation: The saturation factor to apply.
        hue: The hue factor to apply.
        same_on_batch: apply the same transformation across the batch.
        keepdim: whether to keep the output shape the same as input (True) or broadcast it
                 to the batch form (False).
    Shape:
        - Input: :math:`(C, H, W)` or :math:`(B, C, H, W)`, Optional: :math:`(B, 3, 3)`
        - Output: :math:`(B, C, H, W)`

    .. note::
        This function internally uses :func:`kornia.enhance.adjust_brightness`,
        :func:`kornia.enhance.adjust_contrast`. :func:`kornia.enhance.adjust_saturation`,
        :func:`kornia.enhance.adjust_hue`.

    Examples:
        >>> rng = torch.manual_seed(0)
        >>> inputs = torch.ones(1, 3, 3, 3)
        >>> aug = ColorJiggle(0.1, 0.1, 0.1, 0.1, p=1.)
        >>> aug(inputs)
        tensor([[[[0.9993, 0.9993, 0.9993],
                  [0.9993, 0.9993, 0.9993],
                  [0.9993, 0.9993, 0.9993]],
        <BLANKLINE>
                 [[0.9993, 0.9993, 0.9993],
                  [0.9993, 0.9993, 0.9993],
                  [0.9993, 0.9993, 0.9993]],
        <BLANKLINE>
                 [[0.9993, 0.9993, 0.9993],
                  [0.9993, 0.9993, 0.9993],
                  [0.9993, 0.9993, 0.9993]]]])

    To apply the exact augmenation again, you may take the advantage of the previous parameter state:
        >>> input = torch.randn(1, 3, 32, 32)
        >>> aug = ColorJiggle(0.1, 0.1, 0.1, 0.1, p=1.)
        >>> (aug(input) == aug(input, params=aug._params)).all()
        tensor(True)
    """

    def __init__(
        self,
        brightness: Union[Tensor, float, Tuple[float, float], List[float]] = 0.0,
        contrast: Union[Tensor, float, Tuple[float, float], List[float]] = 0.0,
        saturation: Union[Tensor, float, Tuple[float, float], List[float]] = 0.0,
        hue: Union[Tensor, float, Tuple[float, float], List[float]] = 0.0,
        same_on_batch: bool = False,
        p: float = 1.0,
        keepdim: bool = False,
    ) -> None:
        super().__init__(p=p, same_on_batch=same_on_batch, keepdim=keepdim)
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
        self._param_generator = rg.ColorJiggleGenerator(brightness, contrast, saturation, hue)

    def apply_transform(
        self, input: Tensor, params: Dict[str, Tensor], flags: Dict[str, Any], transform: Optional[Tensor] = None
    ) -> Tensor:
        transforms = [
            lambda img: adjust_brightness(img, params["brightness_factor"] - 1)
            if (params["brightness_factor"] - 1 != 0).any()
            else img,
            lambda img: adjust_contrast(img, params["contrast_factor"])
            if (params["contrast_factor"] != 1).any()
            else img,
            lambda img: adjust_saturation(img, params["saturation_factor"])
            if (params["saturation_factor"] != 1).any()
            else img,
            lambda img: adjust_hue(img, params["hue_factor"] * 2 * pi) if (params["hue_factor"] != 0).any() else img,
        ]

        jittered = input
        for idx in params["order"].tolist():
            t = transforms[idx]
            jittered = t(jittered)

        return jittered
