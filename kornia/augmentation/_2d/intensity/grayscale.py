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
from kornia.augmentation.utils import _transform_input
from kornia.color import rgb_to_grayscale


class RandomGrayscale(IntensityAugmentationBase2D):
    r"""Apply random transformation to Grayscale according to a probability p value.

    .. image:: _static/img/RandomGrayscale.png

    Args:
        rgb_weights: Weights that will be applied on each channel (RGB).
            The sum of the weights should add up to one.
        p: probability of the image to be transformed to grayscale.
        same_on_batch: apply the same transformation across the batch.
        keepdim: whether to keep the output shape the same as input (True) or broadcast it
                        to the batch form (False).

    Shape:
        - Input: :math:`(C, H, W)` or :math:`(B, C, H, W)`, Optional: :math:`(B, 3, 3)`
        - Output: :math:`(B, C, H, W)`

    .. note::
        This function internally uses :func:`kornia.color.rgb_to_grayscale`.

    .. note::
        A minimal-overhead fast forward path is taken automatically when called
        with a single plain ``Tensor`` (no boxes/masks/keypoints, no replay
        ``params=``, no kwargs) and ``p`` is deterministic (``0.0`` or ``1.0``).
        For boxes/masks/keypoints/replay the standard chain is preserved.

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

    # The legacy ``_fast_image_only_apply`` opt-in is disabled — the aggressive
    # forward override below is strictly faster.
    _supports_fast_image_only_path: bool = False

    def __init__(
        self, rgb_weights: Optional[Tensor] = None, same_on_batch: bool = False, p: float = 0.1, keepdim: bool = False
    ) -> None:
        super().__init__(p=p, same_on_batch=same_on_batch, keepdim=keepdim)
        self.rgb_weights = rgb_weights

    @torch.no_grad()
    def forward(self, *args: Any, **kwargs: Any) -> Tensor:
        # Aggressive fast path: completely bypass the framework chain for the
        # simple "single image tensor, deterministic p" call.
        if (
            len(args) == 1
            and isinstance(args[0], torch.Tensor)
            and not kwargs
            and self.p_batch == 1.0
            and not self.same_on_batch
            and not self.keepdim
            and self.p in (0.0, 1.0)
        ):
            x = args[0]
            d = x.dim()
            if d == 3:
                x = x.unsqueeze(0)
                d = 4
            if d == 4:
                b = x.shape[0]
                self._params = {
                    "batch_prob": torch.full((b,), bool(self.p > 0.5), dtype=torch.bool),
                    "forward_input_shape": torch.tensor(x.shape, dtype=torch.long),
                }
                eye = torch.eye(3, device=x.device, dtype=x.dtype)
                self._transform_matrix = eye.unsqueeze(0).expand(b, 3, 3)
                if self.p == 0.0:
                    return x
                # ``apply_transform`` returns a (B, 3, H, W) tensor where the
                # grayscale value is broadcast across the 3 channels.
                gray = rgb_to_grayscale(x, rgb_weights=self.rgb_weights)
                return gray.expand_as(x).contiguous()
        return super().forward(*args, **kwargs)

    def apply_transform(
        self, input: Tensor, params: Dict[str, Tensor], flags: Dict[str, Any], transform: Optional[Tensor] = None
    ) -> Tensor:
        # Make sure it returns (*, 3, H, W)
        grayscale = torch.ones_like(input)
        grayscale[:] = rgb_to_grayscale(input, rgb_weights=self.rgb_weights)
        return grayscale
