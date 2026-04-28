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

from typing import Any, Dict, Optional, Union

import torch
from torch import Tensor

from kornia.augmentation._2d.intensity.base import IntensityAugmentationBase2D
from kornia.enhance import invert


class RandomInvert(IntensityAugmentationBase2D):
    r"""Invert the tensor images values randomly.

    .. image:: _static/img/RandomInvert.png

    Args:
        max_val: The expected maximum value in the input tensor. The shape has to
          according to the input tensor shape, or at least has to work with broadcasting.
        same_on_batch: apply the same transformation across the batch.
        p: probability of applying the transformation.
        keepdim: whether to keep the output shape the same as input (True) or broadcast it
                 to the batch form (False).
    .. note::
        This function internally uses :func:`kornia.enhance.invert`.

    .. note::
        A minimal-overhead fast forward path is taken automatically when called
        with a single plain ``Tensor`` (no boxes/masks/keypoints, no replay
        ``params=``, no kwargs) and ``p`` is deterministic (``0.0`` or ``1.0``).
        For boxes/masks/keypoints/replay the standard chain is preserved.

    Examples:
        >>> rng = torch.manual_seed(0)
        >>> img = torch.rand(1, 1, 5, 5)
        >>> inv = RandomInvert()
        >>> inv(img)
        tensor([[[[0.4963, 0.7682, 0.0885, 0.1320, 0.3074],
                  [0.6341, 0.4901, 0.8964, 0.4556, 0.6323],
                  [0.3489, 0.4017, 0.0223, 0.1689, 0.2939],
                  [0.5185, 0.6977, 0.8000, 0.1610, 0.2823],
                  [0.6816, 0.9152, 0.3971, 0.8742, 0.4194]]]])

    To apply the exact augmenation again, you may take the advantage of the previous parameter state:
        >>> input = torch.randn(1, 3, 32, 32)
        >>> aug = RandomInvert(p=1.)
        >>> (aug(input) == aug(input, params=aug._params)).all()
        tensor(True)

    """

    # The legacy ``_fast_image_only_apply`` opt-in is disabled — the aggressive
    # forward override below is strictly faster.
    _supports_fast_image_only_path: bool = False

    def __init__(
        self,
        max_val: Union[float, Tensor] = 1.0,
        same_on_batch: bool = False,
        p: float = 0.5,
        keepdim: bool = False,
    ) -> None:
        super().__init__(p=p, same_on_batch=same_on_batch, p_batch=1.0, keepdim=keepdim)
        self.flags = {"max_val": max_val}

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
                max_val = torch.as_tensor(self.flags["max_val"], device=x.device, dtype=x.dtype)
                return max_val - x
        return super().forward(*args, **kwargs)

    def apply_transform(
        self, input: Tensor, params: Dict[str, Tensor], flags: Dict[str, Any], transform: Optional[Tensor] = None
    ) -> Tensor:
        return invert(input, torch.as_tensor(flags["max_val"], device=input.device, dtype=input.dtype))
