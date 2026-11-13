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
from torch import Tensor

from kornia.augmentation._2d.intensity.base import IntensityAugmentationBase2D
from kornia.augmentation.utils.helpers import _exact_in_float64, _float_numbers
from kornia.enhance import denormalize


class Denormalize(IntensityAugmentationBase2D):
    r"""Restore image channels with supplied mean and standard deviation.

    ``output = Denormalize(mean, std)(input)`` returns ``input * std + mean``.

    See the Convention block on :class:`~kornia.augmentation.IntensityAugmentationBase2D`.

    .. math::
        \text{input[channel] = (input[channel] * std[channel]) + mean[channel]}

    Where `mean` is :math:`(M_1, \ldots, M_n)` and `std` :math:`(S_1, \ldots, S_n)` for `n` channels,

    Args:
        mean: Mean for each channel.
        std: Standard deviations for each channel.
        p: probability of applying the transformation.
        keepdim: whether to keep the output shape the same as input (True) or broadcast it
                 to the batch form (False).

    Return:
        Denormalised tensor with same size as input ``(*, C, H, W)``.

    Convention:
        - this operation is range-independent: it does not clamp the input into ``[0, 1]``.
        - the ``mean`` and ``std`` forms, the batch-wide ``p`` gate and the storage in ``flags`` are those of
          :class:`Normalize`.
        - this class inverts :class:`Normalize` built with the same ``mean`` and nonzero ``std``, up to
          floating-point rounding, when both transforms apply and their intermediate values remain finite.
        - the result is not clamped: it is ``input * std + mean`` whatever range the input is in.

    .. note::
        This function internally uses :func:`kornia.enhance.denormalize`.

    See Also:
        - :class:`Normalize`: subtracts mean and divides by standard deviation.
        - :class:`RandomAutoContrast`: derives a rescaling from each image's extrema.

    Examples:
        >>> norm = Denormalize(mean=torch.zeros(1, 4), std=torch.ones(1, 4))
        >>> x = torch.rand(1, 4, 3, 3)
        >>> out = norm(x)
        >>> out.shape
        torch.Size([1, 4, 3, 3])

    """

    def __init__(
        self,
        mean: Union[Tensor, Tuple[float], List[float], float],
        std: Union[Tensor, Tuple[float], List[float], float],
        p: float = 1.0,
        keepdim: bool = False,
    ) -> None:
        super().__init__(p=p, same_on_batch=True, keepdim=keepdim)
        # The numbers themselves, for a float64 image: the flags hold them rounded to float32.
        self._mean_numbers = _float_numbers(mean)
        self._std_numbers = _float_numbers(std)
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
        self, input: Tensor, params: Dict[str, Tensor], flags: Dict[str, Any], transform: Optional[Tensor] = None
    ) -> Tensor:
        mean, std = flags["mean"], flags["std"]
        if input.dtype == torch.float64:  # the numbers are absent from an older pickle
            mean = _exact_in_float64(mean, self.__dict__.get("_mean_numbers"), input.device)
            std = _exact_in_float64(std, self.__dict__.get("_std_numbers"), input.device)
        return denormalize(input, mean, std)
