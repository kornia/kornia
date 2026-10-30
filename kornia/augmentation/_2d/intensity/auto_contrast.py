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

import warnings
from typing import Any, Dict, Optional

import torch

from kornia.augmentation._2d.intensity.base import IntensityAugmentationBase2D
from kornia.enhance import normalize_min_max


def _warn_clip_output_deprecated(clip_output: Optional[bool]) -> None:
    if clip_output is not None:
        warnings.warn(
            "`clip_output` of `RandomAutoContrast` is deprecated since kornia 0.9.1 and has no effect: "
            "`kornia.enhance.normalize_min_max` already returns values in [0, 1]. It will be removed in a "
            "future release; stop passing it, and pass `same_on_batch`, `p` and `keepdim` by keyword.",
            DeprecationWarning,
            stacklevel=3,
        )


class RandomAutoContrast(IntensityAugmentationBase2D):
    r"""Apply a random auto-contrast of a torch.Tensor image.

    See the Convention block on :class:`~kornia.augmentation.IntensityAugmentationBase2D`.

    Args:
        p: probability of applying the transformation.
        clip_output: deprecated since 0.9.1 and ignored; passing it emits a :class:`DeprecationWarning`,
          and it will be removed in a future release. :func:`kornia.enhance.normalize_min_max` already maps
          each sample's channels onto ``[0, 1]``. See `#4436 <https://github.com/kornia/kornia/issues/4436>`_.
        same_on_batch: apply the same transformation across the batch.
        keepdim: whether to keep the output shape the same as input (True) or broadcast it
                 to the batch form (False).
    Shape:
        - Input: ``(C, H, W)`` or ``(B, C, H, W)``
        - Output: ``(B, C, H, W)``

    Convention:
        - this operation is range-independent: it does not clamp the input into ``[0, 1]``.
        - the output is :func:`kornia.enhance.normalize_min_max` of the input: each channel of each sample is
          rescaled by its own minimum and maximum as ``(x - min) / (max - min + 1e-6)``. This is a rescale, not
          a clamp, so an input outside ``[0, 1]`` is mapped into range rather than clipped, and a constant
          channel comes back as zeros. Empty batches preserve their shape, dtype and device.

    """

    def __init__(
        self, clip_output: Optional[bool] = None, same_on_batch: bool = False, p: float = 1.0, keepdim: bool = False
    ) -> None:
        _warn_clip_output_deprecated(clip_output)
        super().__init__(p=p, same_on_batch=same_on_batch, keepdim=keepdim)

        self.clip_output = True if clip_output is None else clip_output

    def apply_transform(
        self,
        input: torch.Tensor,
        params: Dict[str, torch.Tensor],
        flags: Dict[str, Any],
        transform: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return normalize_min_max(input)
