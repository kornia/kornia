# LICENSE HEADER MANAGED BY add-license-header
#
# Copyright 2018 Kornia Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from collections.abc import Sequence
from typing import Any, Dict, List, Optional, Tuple, Union

import torch

from kornia.augmentation import random_generator as rg
from kornia.augmentation._2d.intensity.base import IntensityAugmentationBase2D
from kornia.constants import pi
from kornia.enhance import (
    adjust_brightness_accumulative,
    adjust_contrast_with_mean_subtraction,
    adjust_hue,
    adjust_saturation_with_gray_subtraction,
)


class ColorJitter(IntensityAugmentationBase2D):
    r"""Apply a random transformation to the brightness, contrast, saturation and hue of a torch.Tensor image.

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
        brightness: Union[torch.Tensor, float, Tuple[float, float], List[float]] = 0.0,
        contrast: Union[torch.Tensor, float, Tuple[float, float], List[float]] = 0.0,
        saturation: Union[torch.Tensor, float, Tuple[float, float], List[float]] = 0.0,
        hue: Union[torch.Tensor, float, Tuple[float, float], List[float]] = 0.0,
        same_on_batch: bool = False,
        p: float = 1.0,
        keepdim: bool = False,
        order: Optional[Sequence[int]] = None,
    ) -> None:
        super().__init__(p=p, same_on_batch=same_on_batch, keepdim=keepdim)

        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
        self._param_generator = rg.ColorJitterGenerator(brightness, contrast, saturation, hue)

        # A fixed application order (a permutation/subset of 0..3 for brightness, contrast,
        # saturation, hue) makes apply_transform a static Python loop instead of iterating the
        # random `order` tensor, so it becomes torch.compile fullgraph-safe. Default (None)
        # keeps the original random per-call order.
        if order is not None:
            order = tuple(int(i) for i in order)
            if not set(order) <= {0, 1, 2, 3}:
                raise ValueError(
                    f"`order` entries must be in 0..3 (brightness, contrast, saturation, hue). Got {order}"
                )
        self._fixed_order: Optional[Tuple[int, ...]] = order

        # native functions
        self._brightness_fn = adjust_brightness_accumulative
        self._contrast_fn = adjust_contrast_with_mean_subtraction
        self._saturation_fn = adjust_saturation_with_gray_subtraction
        self._hue_fn = adjust_hue

    def apply_transform(
        self,
        input: torch.Tensor,
        params: Dict[str, torch.Tensor],
        flags: Dict[str, Any],
        transform: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # torch.where (compute-then-select) rather than a Python `if ... .any() else img`:
        # numerically identical (the op runs on the whole batch or not at all, exactly as the
        # guard decided) but without a data-dependent branch, so it stays fullgraph-compilable.
        transforms = [
            lambda img: torch.where(
                (params["brightness_factor"] != 0).any(), self._brightness_fn(img, params["brightness_factor"]), img
            ),
            lambda img: torch.where(
                (params["contrast_factor"] != 1).any(), self._contrast_fn(img, params["contrast_factor"]), img
            ),
            lambda img: torch.where(
                (params["saturation_factor"] != 1).any(), self._saturation_fn(img, params["saturation_factor"]), img
            ),
            lambda img: torch.where(
                (params["hue_factor"] != 0).any(), self._hue_fn(img, params["hue_factor"] * 2 * pi), img
            ),
        ]

        # A fixed order is a static Python sequence (fullgraph); otherwise iterate the random
        # per-call `order` tensor (inherently data-dependent, not fullgraph-compilable).
        order: Sequence[int] = self._fixed_order if self._fixed_order is not None else params["order"]

        jittered = input
        for idx in order:
            jittered = transforms[idx](jittered)

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
