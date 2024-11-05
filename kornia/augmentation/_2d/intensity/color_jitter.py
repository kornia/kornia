from typing import Any, Dict, List, Optional, Tuple, Union

import torch

from kornia.augmentation import random_generator as rg
from kornia.augmentation._2d.intensity.base import IntensityAugmentationBase2D
from kornia.constants import pi
from kornia.core import Tensor
from kornia.enhance import (
    adjust_brightness_accumulative,
    adjust_contrast_with_mean_subtraction,
    adjust_hue,
    adjust_saturation_with_gray_subtraction,
)


class ColorJitter(IntensityAugmentationBase2D):
    r"""Apply a random transformation to the brightness, contrast, saturation and hue of a tensor image.

    This implementation aligns PIL. Hence, the output is close to TorchVision. However, it does not
    follow the color theory and is not be actively maintained. Prefer using
    :func:`kornia.augmentation.ColorJiggle`

    .. image:: _static/img/ColorJitter.png

    Args:
        brightness: The brightness factor to apply.
        contrast: The contrast factor to apply.
        saturation: The saturation factor to apply.
        hue: The hue factor to apply.
        silence_instantiation_warning: if True, silence the warning at instantiation.
        same_on_batch: apply the same transformation across the batch.
        p: probability of applying the transformation.
        keepdim: whether to keep the output shape the same as input (True) or broadcast it
                 to the batch form (False).
    Shape:
        - Input: :math:`(C, H, W)` or :math:`(B, C, H, W)`, Optional: :math:`(B, 3, 3)`
        - Output: :math:`(B, C, H, W)`

    .. note::
        This function internally uses :func:`kornia.enhance.adjust_brightness_accumulative`,
        :func:`kornia.enhance.adjust_contrast_with_mean_subtraction`,
        :func:`kornia.enhance.adjust_saturation_with_gray_subtraction`,
        :func:`kornia.enhance.adjust_hue`.

    Examples:
        >>> rng = torch.manual_seed(0)
        >>> inputs = torch.ones(1, 3, 3, 3)
        >>> aug = ColorJitter(0.1, 0.1, 0.1, 0.1, p=1.)
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
        >>> aug = ColorJitter(0.1, 0.1, 0.1, 0.1, p=1.)
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
        self._param_generator = rg.ColorJitterGenerator(brightness, contrast, saturation, hue)

        # native functions
        self._brightness_fn = adjust_brightness_accumulative
        self._contrast_fn = adjust_contrast_with_mean_subtraction
        self._saturation_fn = adjust_saturation_with_gray_subtraction
        self._hue_fn = adjust_hue

    def apply_transform(
        self,
        input: Tensor,
        params: Dict[str, Tensor],
        flags: Dict[str, Any],
        transform: Optional[Tensor] = None,
    ) -> Tensor:
        transforms = [
            lambda img: (
                self._brightness_fn(img, params["brightness_factor"])
                if (params["brightness_factor"] != 0).any()
                else img
            ),
            lambda img: (
                self._contrast_fn(img, params["contrast_factor"]) if (params["contrast_factor"] != 1).any() else img
            ),
            lambda img: (
                self._saturation_fn(img, params["saturation_factor"])
                if (params["saturation_factor"] != 1).any()
                else img
            ),
            lambda img: (
                self._hue_fn(img, params["hue_factor"] * 2 * pi) if (params["hue_factor"] != 0).any() else img
            ),
        ]

        jittered = input
        for idx in params["order"]:
            t = transforms[idx]
            jittered = t(jittered)

        return jittered

    def compile(
        self,
        *,
        fullgraph: bool = False,
        dynamic: bool = False,
        backend: str = "inductor",
        mode: Optional[str] = None,
        options: Optional[Dict[Any, Any]] = None,
        disable: bool = False,
    ) -> "ColorJitter":
        self.brightness_fn = torch.compile(
            self._brightness_fn,
            fullgraph=fullgraph,
            dynamic=dynamic,
            backend=backend,
            mode=mode,
            options=options,
            disable=disable,
        )
        self.contrast_fn = torch.compile(
            self._contrast_fn,
            fullgraph=fullgraph,
            dynamic=dynamic,
            backend=backend,
            mode=mode,
            options=options,
            disable=disable,
        )
        self.saturation_fn = torch.compile(
            self._saturation_fn,
            fullgraph=fullgraph,
            dynamic=dynamic,
            backend=backend,
            mode=mode,
            options=options,
            disable=disable,
        )
        self.hue_fn = torch.compile(
            self._hue_fn,
            fullgraph=fullgraph,
            dynamic=dynamic,
            backend=backend,
            mode=mode,
            options=options,
            disable=disable,
        )
        return self
