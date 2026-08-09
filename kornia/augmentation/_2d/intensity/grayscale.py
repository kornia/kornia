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

from typing import Any, Dict, Optional

import torch
from torch import Tensor

from kornia.augmentation._2d.intensity.base import IntensityAugmentationBase2D
from kornia.color import rgb_to_grayscale
from kornia.core.check import KORNIA_CHECK


class RandomGrayscale(IntensityAugmentationBase2D):
    r"""Apply random transformation to Grayscale according to a probability p value.

    .. image:: _static/img/RandomGrayscale.png

    Works for multispectral imagery too (e.g. satellite data with 4-13+ bands): for a non-RGB
    channel count the grayscale is the weighted average across *all* channels, broadcast back to
    the input channel count. This makes the augmentation usable outside the 3-channel RGB regime.

    Args:
        rgb_weights: Per-channel weights applied when reducing to grayscale — one weight per input
            channel (three, for the usual RGB case). If ``None``, RGB inputs use the standard
            luminance weights and multispectral inputs weight every band equally. The weights
            should sum to one.
        p: probability of the image to be transformed to grayscale.
        same_on_batch: apply the same transformation across the batch.
        keepdim: whether to keep the output shape the same as input (True) or broadcast it
                        to the batch form (False).

    Shape:
        - Input: :math:`(C, H, W)` or :math:`(B, C, H, W)`, Optional: :math:`(B, 3, 3)`
        - Output: :math:`(B, C, H, W)`

    .. note::
        For 3-channel RGB inputs this uses :func:`kornia.color.rgb_to_grayscale`; multispectral
        inputs use a weighted channel average.

    Examples:
        >>> rng = torch.manual_seed(0)
        >>> inputs = torch.randn((1, 3, 3, 3))
        >>> aug = RandomGrayscale(p=1.0)
        >>> aug(inputs)
        tensor([[[[-1.1344, -0.1330,  0.1517],
                  [-0.0791,  0.6711, -0.1413],
                  [-0.1717, -0.9023,  0.0819]],
        <BLANKLINE>
                 [[-1.1344, -0.1330,  0.1517],
                  [-0.0791,  0.6711, -0.1413],
                  [-0.1717, -0.9023,  0.0819]],
        <BLANKLINE>
                 [[-1.1344, -0.1330,  0.1517],
                  [-0.0791,  0.6711, -0.1413],
                  [-0.1717, -0.9023,  0.0819]]]])

    To apply the exact augmenation again, you may take the advantage of the previous parameter state:
        >>> input = torch.randn(1, 3, 32, 32)
        >>> aug = RandomGrayscale(p=1.0)
        >>> (aug(input) == aug(input, params=aug._params)).all()
        tensor(True)

    """

    def __init__(
        self, rgb_weights: Optional[Tensor] = None, same_on_batch: bool = False, p: float = 0.1, keepdim: bool = False
    ) -> None:
        super().__init__(p=p, same_on_batch=same_on_batch, keepdim=keepdim)
        self.rgb_weights = rgb_weights

    def apply_transform(
        self, input: Tensor, params: Dict[str, Tensor], flags: Dict[str, Any], transform: Optional[Tensor] = None
    ) -> Tensor:
        num_channels = input.shape[-3]
        lead = (1,) * (input.ndim - 3)  # leading (batch) dims to broadcast/repeat over
        if num_channels == 3:
            # RGB luminance path — byte-identical to the previous behaviour.
            gray = rgb_to_grayscale(input, rgb_weights=self.rgb_weights)
        else:
            # Multispectral (e.g. satellite, 4-13+ bands): weighted average over all channels,
            # using ``rgb_weights`` (one per channel) or equal weights when unset.
            weights = self.rgb_weights
            if weights is None:
                weights = torch.full((num_channels,), 1.0 / num_channels, device=input.device, dtype=input.dtype)
            else:
                KORNIA_CHECK(
                    weights.numel() == num_channels,
                    f"rgb_weights needs one weight per channel: got {weights.numel()} for {num_channels}.",
                )
                weights = weights.to(input)
            gray = (input * weights.view(*lead, num_channels, 1, 1)).sum(-3, keepdim=True)
        # Broadcast the single grayscale channel back to the input channel count (a contiguous copy).
        return gray.repeat(*lead, num_channels, 1, 1)
