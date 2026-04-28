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

from typing import Any, Dict, Optional, Tuple, Union

import torch

from kornia.augmentation import random_generator as rg
from kornia.augmentation._2d.intensity.base import IntensityAugmentationBase2D
from kornia.augmentation.utils import _transform_input
from kornia.enhance import posterize


class RandomPosterize(IntensityAugmentationBase2D):
    r"""Posterize given torch.Tensor image or a batch of torch.Tensor images randomly.

    .. image:: _static/img/RandomPosterize.png

    Args:
        p: probability of applying the transformation.
        bits: Integer that ranged from (0, 8], in which 0 gives black image and 8 gives the original.
            If int x, bits will be generated from (x, 8) then convert to int.
            If tuple (x, y), bits will be generated from (x, y) then convert to int.
        same_on_batch: apply the same transformation across the batch.
        keepdim: whether to keep the output shape the same as input (True) or broadcast it
                 to the batch form (False).

    Shape:
        - Input: :math:`(C, H, W)` or :math:`(B, C, H, W)`, Optional: :math:`(B, 3, 3)`
        - Output: :math:`(B, C, H, W)`

    .. note::
        This function internally uses :func:`kornia.enhance.posterize`.

    .. note::
        A minimal-overhead fast forward path is taken automatically when called
        with a single plain ``Tensor`` (no boxes/masks/keypoints, no replay
        ``params=``, no kwargs) and ``p`` is deterministic (``0.0`` or ``1.0``).
        For boxes/masks/keypoints/replay the standard chain is preserved.

    Examples:
        >>> rng = torch.manual_seed(0)
        >>> input = torch.rand(1, 1, 5, 5)
        >>> posterize = RandomPosterize(3., p=1.)
        >>> posterize(input)
        tensor([[[[0.4863, 0.7529, 0.0784, 0.1255, 0.2980],
                  [0.6275, 0.4863, 0.8941, 0.4549, 0.6275],
                  [0.3451, 0.3922, 0.0157, 0.1569, 0.2824],
                  [0.5176, 0.6902, 0.8000, 0.1569, 0.2667],
                  [0.6745, 0.9098, 0.3922, 0.8627, 0.4078]]]])

    To apply the exact augmenation again, you may take the advantage of the previous parameter state:
        >>> input = torch.randn(1, 3, 32, 32)
        >>> aug = RandomPosterize(3., p=1.)
        >>> (aug(input) == aug(input, params=aug._params)).all()
        tensor(True)

    """

    # The legacy ``_fast_image_only_apply`` opt-in is disabled — the aggressive
    # forward override below is strictly faster.
    _supports_fast_image_only_path: bool = False

    def __init__(
        self,
        bits: Union[float, Tuple[float, float], torch.Tensor] = 3,
        same_on_batch: bool = False,
        p: float = 0.5,
        keepdim: bool = False,
    ) -> None:
        super().__init__(p=p, same_on_batch=same_on_batch, keepdim=keepdim)
        # TODO: the generator should receive the device
        self._param_generator = rg.PosterizeGenerator(bits)

    @torch.no_grad()
    def forward(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        # Aggressive fast path: bypass the framework chain for the simple
        # "single image tensor, deterministic p" call.
        if (
            len(args) == 1
            and isinstance(args[0], torch.Tensor)
            and not kwargs
            and self.p_batch == 1.0
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
                eye = torch.eye(3, device=x.device, dtype=x.dtype)
                self._transform_matrix = eye.unsqueeze(0).expand(b, 3, 3)
                if self.p == 0.0:
                    self._params = {
                        "batch_prob": torch.zeros(b, dtype=torch.bool),
                        "forward_input_shape": torch.tensor(x.shape, dtype=torch.long),
                    }
                    return x
                params = self._param_generator(torch.Size((b, *x.shape[1:])), self.same_on_batch)
                self._params = dict(params)
                self._params["batch_prob"] = torch.ones(b, dtype=torch.bool)
                self._params["forward_input_shape"] = torch.tensor(x.shape, dtype=torch.long)
                return posterize(x, params["bits_factor"].to(x.device))
        return super().forward(*args, **kwargs)

    def apply_transform(
        self,
        input: torch.Tensor,
        params: Dict[str, torch.Tensor],
        flags: Dict[str, Any],
        transform: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return posterize(input, params["bits_factor"].to(input.device))
