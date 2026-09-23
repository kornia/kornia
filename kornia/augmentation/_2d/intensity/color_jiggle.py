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
from torch.distributions import Distribution

from kornia.augmentation import random_generator as rg
from kornia.augmentation._2d.intensity.base import IntensityAugmentationBase2D
from kornia.constants import pi
from kornia.enhance import adjust_brightness, adjust_contrast, adjust_hue, adjust_saturation


def _contiguous_output(output: torch.Tensor) -> torch.Tensor:
    return output.contiguous()


def _identity(input: torch.Tensor, factor: torch.Tensor) -> torch.Tensor:
    # torch.cond rejects an output that aliases an input, so a neutral factor returns a copy. Every
    # branch returns a contiguous tensor: their forward and backward metadata must agree under Inductor,
    # which rejects preserve_format branches for channels-last and transposed inputs.
    return input.contiguous().clone()


def _adjust_brightness(input: torch.Tensor, factor: torch.Tensor) -> torch.Tensor:
    return _contiguous_output(adjust_brightness(input, factor - 1))


def _adjust_contrast(input: torch.Tensor, factor: torch.Tensor) -> torch.Tensor:
    return _contiguous_output(adjust_contrast(input, factor))


def _adjust_saturation(input: torch.Tensor, factor: torch.Tensor) -> torch.Tensor:
    return _contiguous_output(adjust_saturation(input, factor))


def _adjust_hue(input: torch.Tensor, factor: torch.Tensor) -> torch.Tensor:
    return _contiguous_output(adjust_hue(input, factor * 2 * pi))


def _apply_transform_cond(
    index: int,
    input: torch.Tensor,
    brightness: torch.Tensor,
    contrast: torch.Tensor,
    saturation: torch.Tensor,
    hue: torch.Tensor,
) -> torch.Tensor:
    if index == 0:
        return torch.cond((brightness - 1 != 0).any(), _adjust_brightness, _identity, (input, brightness))
    if index == 1:
        return torch.cond((contrast != 1).any(), _adjust_contrast, _identity, (input, contrast))
    if index == 2:
        return torch.cond((saturation != 1).any(), _adjust_saturation, _identity, (input, saturation))
    return torch.cond((hue != 0).any(), _adjust_hue, _identity, (input, hue))


class ColorJiggle(IntensityAugmentationBase2D):
    r"""Apply a random transformation to the brightness, contrast, saturation and hue of a torch.Tensor image.

    .. image:: _static/img/ColorJiggle.png

    See the Convention block on :class:`~kornia.augmentation.IntensityAugmentationBase2D`.

    Args:
        p: probability of applying the transformation.
        brightness: The brightness factor to apply.
        contrast: The contrast factor to apply.
        saturation: The saturation factor to apply.
        hue: The hue factor to apply.
        same_on_batch: apply the same transformation across the batch.
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
        - the steps use kornia's own primitives: brightness is additive (the drawn factor is re-based to
          ``factor - 1`` for :func:`kornia.enhance.adjust_brightness`, as :class:`RandomBrightness` does),
          contrast scales the raw values (:func:`kornia.enhance.adjust_contrast`), saturation scales the HSV
          saturation (:func:`kornia.enhance.adjust_saturation`) and hue shifts by turns of the hue circle
          (:func:`kornia.enhance.adjust_hue`). :class:`ColorJitter` uses torchvision's blend formulas for the first
          three instead, so the two classes give different outputs for the same factors.
        - with matching effective bounds and the default CPU sampler, this class and :class:`ColorJitter` draw the
          same factors and the same ``order`` from the same seed. The drawn ``order`` is shared by the whole
          batch; without a fixed ``order`` argument, an ``order`` tensor passed as a forward keyword or in
          replayed ``params`` replaces it.
        - ``brightness`` is bounded to ``[0, 2]`` in both argument forms: ``(0.0, 3.0)`` and the scalar ``1.5``,
          whose implied range reaches ``2.5``, raise at construction, where :class:`ColorJitter` accepts either.
        - a step is skipped when every drawn factor in the batch is neutral, so ``ColorJiggle(0, 0, 0, 0)`` is
          the identity for any input, and the channel count only has to suit the steps that run: saturation and
          hue need three channels, a brightness- or contrast-only configuration accepts any. The brightness and
          contrast steps clamp into ``[0, 1]``; saturation and hue do not clamp the RGB result.

    .. warning::
        Because the brightness and contrast steps clamp, an all-negative input can come back as an all-zero
        image, depending on the drawn factors and on which steps run. Tracked in
        `#4430 <https://github.com/kornia/kornia/issues/4430>`_.

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
        self._param_generator = rg.ColorJiggleGenerator(brightness, contrast, saturation, hue)
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
        self._cond_dispatch = order is not None and torch._dynamo.is_dynamo_supported()

    def apply_transform(
        self,
        input: torch.Tensor,
        params: Dict[str, torch.Tensor],
        flags: Dict[str, Any],
        transform: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # A fixed order runs the same torch.cond dispatcher in eager and compiled mode. Every torch.cond
        # branch is traced, including branches that are not selected at runtime, so the dispatcher is
        # restricted to RGB inputs: tracing the hue/saturation branches would otherwise reject the neutral
        # one- and four-channel configurations accepted by the Python dispatch below.
        if self._cond_dispatch and input.shape[-3] == 3:
            # An eager torch.cond enters Dynamo, whose one-time setup calls
            # ``Distribution.set_default_validate_args(False)`` process-wide; restore the caller's setting.
            validate_args = Distribution._validate_args
            try:
                jittered = input
                for idx in self._fixed_order:
                    jittered = _apply_transform_cond(
                        idx,
                        jittered,
                        params["brightness_factor"],
                        params["contrast_factor"],
                        params["saturation_factor"],
                        params["hue_factor"],
                    )
            finally:
                if Distribution._validate_args != validate_args:
                    Distribution.set_default_validate_args(validate_args)
            return jittered

        transforms = [
            lambda img: (
                adjust_brightness(img, params["brightness_factor"] - 1)
                if (params["brightness_factor"] - 1 != 0).any()
                else img
            ),
            lambda img: (
                adjust_contrast(img, params["contrast_factor"]) if (params["contrast_factor"] != 1).any() else img
            ),
            lambda img: (
                adjust_saturation(img, params["saturation_factor"]) if (params["saturation_factor"] != 1).any() else img
            ),
            lambda img: adjust_hue(img, params["hue_factor"] * 2 * pi) if (params["hue_factor"] != 0).any() else img,
        ]

        jittered = input
        order = self._fixed_order if self._fixed_order is not None else params["order"].tolist()
        for idx in order:
            t = transforms[idx]
            jittered = t(jittered)

        return jittered
