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

from typing import Any, Optional

import torch
from torch import Tensor

from kornia.augmentation._2d.intensity.base import IntensityAugmentationBase2D
from kornia.augmentation.utils import _transform_input
from kornia.enhance import normalize


class Normalize(IntensityAugmentationBase2D):
    r"""Normalize tensor images with mean and standard deviation.

    .. math::
        \text{input[channel] = (input[channel] - mean[channel]) / std[channel]}

    Where `mean` is :math:`(M_1, ..., M_n)` and `std` :math:`(S_1, ..., S_n)` for `n` channels,

    Args:
        mean: Mean for each channel.
        std: Standard deviations for each channel.
        p: probability of applying the transformation.
        keepdim: whether to keep the output shape the same as input (True) or broadcast it
                 to the batch form (False).

    Return:
        Normalised tensor with same size as input :math:`(*, C, H, W)`.

    Examples:
        >>> norm = Normalize(mean=torch.zeros(4), std=torch.ones(4))
        >>> x = torch.rand(1, 4, 3, 3)
        >>> out = norm(x)
        >>> out.shape
        torch.Size([1, 4, 3, 3])

    """

    def __init__(
        self,
        mean: Tensor | tuple[float, ...] | list[float] | float,
        std: Tensor | tuple[float, ...] | list[float] | float,
        p: float = 1.0,
        keepdim: bool = False,
    ) -> None:
        super().__init__(p=p, same_on_batch=True, keepdim=keepdim)
        if isinstance(mean, (int, float)):
            mean = torch.tensor([mean])

        if isinstance(std, (int, float)):
            std = torch.tensor([std])

        if isinstance(mean, (tuple, list)):
            mean = torch.tensor(mean)

        if isinstance(std, (tuple, list)):
            std = torch.tensor(std)

        # Public flags retained for backwards-compat (some external code introspects).
        self.flags = {"mean": mean, "std": std}

        # Fast-path buffers: pre-shaped for broadcast over (B, C, H, W) input.
        # 1-D mean/std reshape to (1, C, 1, 1); higher-rank passes through.
        # Registered as non-persistent buffers so .to() / .cuda() / .float() handle them.
        if mean.dim() == 1:
            mean_b = mean.view(1, -1, 1, 1)
            std_b = std.view(1, -1, 1, 1)
        else:
            mean_b = mean
            std_b = std
        self.register_buffer("_mean_b", mean_b, persistent=False)
        self.register_buffer("_std_b", std_b, persistent=False)

    # The legacy ``_fast_image_only_apply`` opt-in is disabled — the aggressive
    # forward override below is strictly faster.
    _supports_fast_image_only_path: bool = False

    @torch.no_grad()
    def forward(self, *args: Any, **kwargs: Any) -> Tensor:
        # Aggressive fast path: completely bypass the framework chain for the
        # simple "single image tensor, deterministic p" call.  ``Normalize`` is
        # always called with ``same_on_batch=True`` and the per-sample mask is
        # uniform across the batch, so this branch is the common case.
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
                self._params = {
                    "batch_prob": torch.full((b,), bool(self.p > 0.5), dtype=torch.bool),
                    "forward_input_shape": torch.tensor(x.shape, dtype=torch.long),
                }
                eye = torch.eye(3, device=x.device, dtype=x.dtype)
                self._transform_matrix = eye.unsqueeze(0).expand(b, 3, 3)
                if self.p == 0.0:
                    return x
                mean = self._mean_b
                std = self._std_b
                # Mirror ``kornia.enhance.normalize`` channel-shape validation
                # so users who pass mismatched mean/std still get the expected
                # ValueError on the fast path.
                mean_c = mean.shape[1] if mean.dim() == 4 else (mean.shape[0] if mean.dim() else 1)
                if mean_c != 1 and mean_c != x.shape[1]:
                    raise ValueError(
                        f"mean length and number of channels do not match. Got {mean.shape} and {x.shape}."
                    )
                std_c = std.shape[1] if std.dim() == 4 else (std.shape[0] if std.dim() else 1)
                if std_c != 1 and std_c != x.shape[1]:
                    raise ValueError(
                        f"std length and number of channels do not match. Got {std.shape} and {x.shape}."
                    )
                if mean.dtype != x.dtype or mean.device != x.device:
                    mean = mean.to(device=x.device, dtype=x.dtype)
                    std = std.to(device=x.device, dtype=x.dtype)
                return (x - mean) / std
        return super().forward(*args, **kwargs)

    def apply_transform(
        self, input: Tensor, params: dict[str, Tensor], flags: dict[str, Any], transform: Optional[Tensor] = None
    ) -> Tensor:
        # Fast path: bypass kornia.enhance.normalize's ONNX/validation overhead and
        # the per-call torch.as_tensor device cast. Buffers are pre-shaped for broadcast.
        mean = self._mean_b
        std = self._std_b
        if mean.dtype != input.dtype or mean.device != input.device:
            mean = mean.to(device=input.device, dtype=input.dtype)
            std = std.to(device=input.device, dtype=input.dtype)
        return (input - mean) / std
