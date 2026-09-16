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
from kornia.enhance import normalize


class Normalize(IntensityAugmentationBase2D):
    r"""Normalize tensor images with mean and standard deviation.

    See the Convention block on :class:`~kornia.augmentation.IntensityAugmentationBase2D`.

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

    Convention:
        - ``mean`` and ``std`` accept a float, a per-channel sequence or tensor, or a per-sample ``(B, C)``
          tensor; a length that is neither ``1`` nor the channel count raises.
        - ``p`` gates the whole batch rather than each sample: the constructor hard-codes
          ``same_on_batch=True``, which collapses the per-sample draw to a single one, and it takes no
          ``same_on_batch`` argument of its own.
        - the statistics live in ``flags`` rather than in a buffer, so ``state_dict()`` is empty and
          ``Module.to(...)`` leaves their device and dtype alone.
        - :class:`Denormalize` built with the same float, sequence or tensor ``mean`` and ``std`` inverts this
          class, up to float rounding, when both apply -- each draws its own ``p`` gate. An ``int`` statistic,
          which this class accepts, makes :class:`Denormalize` raise on the forward pass
          (`#4573 <https://github.com/kornia/kornia/issues/4573>`_).
        - the result is not clamped: moving an image out of ``[0, 1]`` is what this class is for.

    .. warning::
        A non-contiguous input -- anything :meth:`torch.Tensor.view` cannot reshape, such as a
        ``transpose``d or ``permute``d view -- raises a raw ``RuntimeError: view size is not compatible
        with input tensor's size and stride`` from :func:`kornia.enhance.normalize`, naming neither this
        class nor the fix (``.contiguous()``). Tracked in
        `#4577 <https://github.com/kornia/kornia/issues/4577>`_.

    .. note::
        This function internally uses :func:`kornia.enhance.normalize`.

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

        self.flags = {"mean": mean, "std": std}

    def apply_transform(
        self, input: Tensor, params: dict[str, Tensor], flags: dict[str, Any], transform: Optional[Tensor] = None
    ) -> Tensor:
        mean: Tensor = flags["mean"]
        std: Tensor = flags["std"]
        if torch.onnx.is_in_onnx_export():
            if mean.dim() == 1:
                mean = mean.view(1, -1)
            if std.dim() == 1:
                std = std.view(1, -1)
        return normalize(input, mean, std)
