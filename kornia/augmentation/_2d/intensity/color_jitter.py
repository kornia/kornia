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
from kornia.augmentation._2d.intensity.base import IntensityAugmentationBase2D, _PicklableCompileMixin
from kornia.augmentation._2d.intensity.color_jiggle import (
    _adjust_hue,
    _apply_order_cond,
    _contiguous_output,
    _dispatch_color_steps,
    _Steps,
)
from kornia.constants import pi
from kornia.enhance import (
    adjust_brightness_accumulative,
    adjust_contrast_with_mean_subtraction,
    adjust_hue,
    adjust_saturation_with_gray_subtraction,
)


def _adjust_brightness(input: torch.Tensor, factor: torch.Tensor) -> torch.Tensor:
    return _contiguous_output(adjust_brightness_accumulative(input, factor))


def _adjust_contrast(input: torch.Tensor, factor: torch.Tensor) -> torch.Tensor:
    return _contiguous_output(adjust_contrast_with_mean_subtraction(input, factor))


def _adjust_saturation(input: torch.Tensor, factor: torch.Tensor) -> torch.Tensor:
    return _contiguous_output(adjust_saturation_with_gray_subtraction(input, factor))


# The brightness step tests its factor against 0 instead of the neutral 1 (#4785).
_NEUTRAL = (0.0, 1.0, 1.0, 0.0)
_BRANCHES: _Steps = (_adjust_brightness, _adjust_contrast, _adjust_saturation, _adjust_hue)


def _apply_cond(order: Tuple[int, ...], input: torch.Tensor, factors: Tuple[torch.Tensor, ...]) -> torch.Tensor:
    return _apply_order_cond(_BRANCHES, _NEUTRAL, order, input, factors)


class ColorJitter(_PicklableCompileMixin, IntensityAugmentationBase2D):
    r"""Apply a random transformation to the brightness, contrast, saturation and hue of a torch.Tensor image.

    The four steps are torchvision's formulas -- brightness multiplies the image by the factor, contrast blends it
    with its grayscale mean, saturation with its grayscale, and hue is torchvision's ``adjust_hue`` shift -- so this
    is the class to port ``torchvision.transforms.ColorJitter`` code to. It is not actively maintained; prefer
    :class:`ColorJiggle` for new code.

    .. image:: _static/img/ColorJitter.png

    See the Convention block on :class:`~kornia.augmentation.IntensityAugmentationBase2D`.

    Args:
        brightness: The brightness factor to apply.
        contrast: The contrast factor to apply.
        saturation: The saturation factor to apply.
        hue: The hue factor to apply.
        same_on_batch: apply the same transformation across the batch.
        p: probability of applying the transformation.
        keepdim: whether to keep the output shape the same as input (True) or broadcast it
                 to the batch form (False).
        order: a fixed application order, as indices into (brightness, contrast, saturation, hue); a subset
          applies only those, and a repeated index raises ``ValueError``. ``None`` (the default) draws a random
          order on every call. A fixed order makes the transform ``torch.compile`` fullgraph-safe for RGB inputs.
          The parameter generator still draws an ``order`` entry into ``_params``, and with a fixed order that
          entry is ignored, including on replay.
    Shape:
        - Input: :math:`(C, H, W)` or :math:`(B, C, H, W)`, Optional: :math:`(B, 3, 3)`
        - Output: :math:`(B, C, H, W)`

    Convention:
        - see :class:`ColorJiggle` for how the two classes relate. This class takes the brightness factor as
          drawn, a multiplier whose identity is ``1``, and a scalar ``brightness`` above ``1`` draws from
          ``[0, 1 + brightness]``, as torchvision does.
        - a step is skipped when every factor in the batch equals its guard value -- ``1`` for contrast and
          saturation, ``0`` for hue and, as the #4785 warning below states, for brightness -- so a skipped step
          accepts any channel count. Otherwise the step runs on the whole batch, and the brightness, contrast and
          (three-channel) saturation steps clamp the whole batch into ``[0, 1]``. A fixed ``order`` without
          index ``0`` skips the brightness step.

    .. warning::
        The brightness step is skipped for a factor of ``0`` instead of the neutral ``1``: a batch whose factors
        are all ``0`` comes back unchanged instead of black, and ``ColorJitter(0, 0, 0, 0)`` clamps an
        out-of-range input. Tracked in
        `#4785 <https://github.com/kornia/kornia/issues/4785>`_.

    .. warning::
        Because the brightness, contrast and saturation steps clamp, an all-negative input can come back as an
        all-zero image, depending on the draw and the order. Tracked in
        `#4430 <https://github.com/kornia/kornia/issues/4430>`_.

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
            if len(order) != len(set(order)):
                raise ValueError(f"`order` must not repeat an index; each adjustment applies at most once. Got {order}")
        self._fixed_order: Optional[Tuple[int, ...]] = order
        # torch.cond raises where Dynamo is unavailable (torch 2.5.1 on Python 3.13), so a fixed order keeps
        # the Python dispatch there. Checked here because Dynamo cannot trace the check inside forward.
        self._cond_fn = _apply_cond if order is not None and torch._dynamo.is_dynamo_supported() else None

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
        # The same dispatch as ColorJiggle: a fixed order on an RGB input runs torch.cond in eager and compiled
        # mode, every other call the Python guards, which accept any channel count for a skipped step.
        steps: _Steps = (self._brightness_fn, self._contrast_fn, self._saturation_fn, self._adjust_hue_turns)
        return _dispatch_color_steps(input, params, self._fixed_order, self._cond_fn, _NEUTRAL, steps)

    def _adjust_hue_turns(self, input: torch.Tensor, factor: torch.Tensor) -> torch.Tensor:
        return self._hue_fn(input, factor * 2 * pi)

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
        self._record_compile(
            ["_cond_fn", "_brightness_fn", "_contrast_fn", "_saturation_fn", "_hue_fn"],
            {
                "fullgraph": fullgraph,
                "dynamic": dynamic,
                "backend": backend,
                "mode": mode,
                "options": options,
                "disable": disable,
            },
        )
        # A fixed order on an RGB input runs every step through the torch.cond dispatcher, which is compiled
        # as one graph; the four helpers serve the random order and non-RGB inputs.
        if self._cond_fn is not None:
            self._cond_fn = torch.compile(
                self._cond_fn,
                fullgraph=fullgraph,
                dynamic=dynamic,
                backend=backend,
                mode=mode,
                options=options,
                disable=disable,
            )
        self._brightness_fn = torch.compile(
            self._brightness_fn,
            fullgraph=fullgraph,
            dynamic=dynamic,
            backend=backend,
            mode=mode,
            options=options,
            disable=disable,
        )
        self._contrast_fn = torch.compile(
            self._contrast_fn,
            fullgraph=fullgraph,
            dynamic=dynamic,
            backend=backend,
            mode=mode,
            options=options,
            disable=disable,
        )
        self._saturation_fn = torch.compile(
            self._saturation_fn,
            fullgraph=fullgraph,
            dynamic=dynamic,
            backend=backend,
            mode=mode,
            options=options,
            disable=disable,
        )
        self._hue_fn = torch.compile(
            self._hue_fn,
            fullgraph=fullgraph,
            dynamic=dynamic,
            backend=backend,
            mode=mode,
            options=options,
            disable=disable,
        )
        return self
