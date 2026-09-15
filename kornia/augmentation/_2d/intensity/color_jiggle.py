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

from typing import Any, Dict, List, Optional, Tuple, Union

import torch

from kornia.augmentation import random_generator as rg
from kornia.augmentation._2d.intensity.base import IntensityAugmentationBase2D
from kornia.constants import pi
from kornia.enhance import adjust_brightness, adjust_contrast, adjust_hue, adjust_saturation


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
    Shape:
        - Input: :math:`(C, H, W)` or :math:`(B, C, H, W)`, Optional: :math:`(B, 3, 3)`
        - Output: :math:`(B, C, H, W)`

    Convention:
        - this class and :class:`ColorJitter` draw the same factor values and the same random application
          ``order`` from the same seed when their effective sampling bounds match and both modules stay on
          the CPU. A scalar ``brightness > 1`` does not match: this class draws from ``[0, 2]`` while
          :class:`ColorJitter` draws from ``[0, 1 + brightness]``. Off the CPU the ``order`` diverges,
          because this class draws it on the sampler device where :class:`ColorJitter` always draws it on
          the CPU; and this class returns its factors in the dtype of its constructor arguments
          (``float32`` for Python floats) where :class:`ColorJitter` keeps the sampler dtype. Only
          :class:`ColorJitter` takes an ``order`` constructor argument that replaces its sampled order with a
          fixed one. The classes use different primitives for three adjustments:
          :func:`kornia.enhance.adjust_brightness` against
          :func:`kornia.enhance.adjust_brightness_accumulative`,
          :func:`kornia.enhance.adjust_contrast` against
          :func:`kornia.enhance.adjust_contrast_with_mean_subtraction`, and
          :func:`kornia.enhance.adjust_saturation` against
          :func:`kornia.enhance.adjust_saturation_with_gray_subtraction`. Both call
          :func:`kornia.enhance.adjust_hue`.
        - the brightness factor is re-based as :class:`RandomBrightness` re-bases it:
          ``factor - 1`` is what reaches :func:`kornia.enhance.adjust_brightness`.
        - ``ColorJiggle(0, 0, 0, 0)`` is the identity, including for values outside ``[0, 1]``. A hue-only
          configuration can also return values outside that interval.

    .. warning::
        In ``float16`` a black pixel reaching the saturation or hue step comes back as NaN, because both
        steps go through ``rgb_to_hsv``, whose ``eps`` underflows there. The hue step does the same for any
        pixel whose largest channel is ``0``. Tracked in
        `#4560 <https://github.com/kornia/kornia/issues/4560>`_.

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
    ) -> None:
        super().__init__(p=p, same_on_batch=same_on_batch, keepdim=keepdim)
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
        self._param_generator = rg.ColorJiggleGenerator(brightness, contrast, saturation, hue)

    def apply_transform(
        self,
        input: torch.Tensor,
        params: Dict[str, torch.Tensor],
        flags: Dict[str, Any],
        transform: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
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
        for idx in params["order"].tolist():
            t = transforms[idx]
            jittered = t(jittered)

        return jittered
