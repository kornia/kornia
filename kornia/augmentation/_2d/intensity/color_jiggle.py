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

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple, Union

import torch

from kornia.augmentation import random_generator as rg
from kornia.augmentation._2d.intensity.base import IntensityAugmentationBase2D
from kornia.color import hsv_to_rgb, rgb_to_hsv

# 2*pi constant used for hue scaling; computed once at module load.
_TWO_PI: float = 2.0 * math.pi


class ColorJiggle(IntensityAugmentationBase2D):
    r"""Apply a random transformation to the brightness, contrast, saturation and hue of a torch.Tensor image.

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
        params: Dict[str, Any],
        flags: Dict[str, Any],
        transform: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        brightness_factor: torch.Tensor = params["brightness_factor"]  # (B,)
        contrast_factor: torch.Tensor = params["contrast_factor"]  # (B,)
        saturation_factor: torch.Tensor = params["saturation_factor"]  # (B,)
        hue_factor: torch.Tensor = params["hue_factor"]  # (B,), in [-0.5, 0.5]

        # Pre-compute deltas once to avoid repeated temporary tensors in the loop.
        brightness_delta = brightness_factor - 1.0  # additive; 0 = no change
        hue_shift = hue_factor * _TWO_PI  # radians; 0 = no change

        do_brightness = bool(brightness_delta.any())
        do_contrast = bool((contrast_factor != 1.0).any())
        do_saturation = bool((saturation_factor != 1.0).any())
        do_hue = bool(hue_shift.any())

        # Build broadcasted factor tensors once here so each op below skips the reshape+cast.
        dtype = input.dtype
        device = input.device

        b_vec = brightness_delta.to(dtype=dtype, device=device).view(-1, 1, 1, 1) if do_brightness else None
        c_vec = contrast_factor.to(dtype=dtype, device=device).view(-1, 1, 1, 1) if do_contrast else None
        s_vec = saturation_factor.to(dtype=dtype, device=device).view(-1, 1, 1, 1) if do_saturation else None
        h_vec = hue_shift.to(dtype=dtype, device=device).view(-1, 1, 1, 1) if do_hue else None

        # Pending HSV adjustments: we accumulate sat and hue deltas and delay the HSV roundtrip
        # until we hit a non-HSV op or reach the end of the loop.  This fuses consecutive
        # saturation + hue operations (or hue + saturation) into a single rgb_to_hsv / hsv_to_rgb
        # pair, saving one full colorspace roundtrip when both are active.
        pending_s: Optional[torch.Tensor] = None
        pending_h: Optional[torch.Tensor] = None

        def flush_hsv(img: torch.Tensor) -> torch.Tensor:
            nonlocal pending_s, pending_h
            if pending_s is None and pending_h is None:
                return img
            img_hsv = rgb_to_hsv(img)
            h_ch = img_hsv[:, 0:1, :, :]
            s_ch = img_hsv[:, 1:2, :, :]
            v_ch = img_hsv[:, 2:3, :, :]
            if pending_s is not None:
                s_ch = (s_ch * pending_s).clamp_(0.0, 1.0)
            if pending_h is not None:
                h_ch = torch.fmod(h_ch + pending_h, _TWO_PI)
            result = hsv_to_rgb(torch.cat([h_ch, s_ch, v_ch], dim=1))
            pending_s = None
            pending_h = None
            return result

        cloned = False  # track whether we already cloned for in-place safety
        jittered = input

        for idx in params["order"].tolist():
            if idx == 0:  # brightness (additive)
                if not do_brightness:
                    continue
                jittered = flush_hsv(jittered)
                if not cloned:
                    jittered = jittered.clone()
                    cloned = True
                jittered.add_(b_vec).clamp_(0.0, 1.0)  # type: ignore[arg-type]

            elif idx == 1:  # contrast (multiplicative)
                if not do_contrast:
                    continue
                jittered = flush_hsv(jittered)
                if not cloned:
                    jittered = jittered.clone()
                    cloned = True
                jittered.mul_(c_vec).clamp_(0.0, 1.0)  # type: ignore[arg-type]

            elif idx == 2:  # saturation — accumulate into pending HSV state
                if not do_saturation:
                    continue
                pending_s = s_vec if pending_s is None else pending_s * s_vec  # type: ignore[operator]

            else:  # idx == 3 — hue, accumulate into pending HSV state
                if not do_hue:
                    continue
                pending_h = h_vec if pending_h is None else pending_h + h_vec  # type: ignore[operator]

        # Flush any remaining HSV work.
        jittered = flush_hsv(jittered)

        return jittered
