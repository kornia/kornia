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

from typing import Dict, Optional, Tuple

import torch
from torch.distributions import Bernoulli

from kornia.augmentation.random_generator.base import RandomGeneratorBase
from kornia.augmentation.utils import _adapted_sampling, _common_param_check
from kornia.core import Tensor, tensor

__all__ = ["ProbabilityGenerator"]


class ProbabilityGenerator(RandomGeneratorBase):
    r"""Generate random probabilities for a batch of inputs.

    Args:
        p: probability to generate an 1-d binary mask. Default value is 0.5.

    Returns:
        A dict of parameters to be passed for transformation.
            - probs (Tensor): element-wise probabilities with a shape of (B,).

    Note:
        The generated random numbers are not reproducible across different devices and dtypes. By default,
        the parameters will be generated on CPU in float32. This can be changed by calling
        ``self.set_rng_device_and_dtype(device="cuda", dtype=torch.float64)``.

    """

    def __init__(self, p: float = 0.5) -> None:
        super().__init__()
        self.p = p

    def __repr__(self) -> str:
        repr = f"p={self.p}"
        return repr

    def make_samplers(self, device: torch.device, dtype: torch.dtype) -> None:
        p = torch.tensor(float(self.p), device=device, dtype=dtype)
        self.sampler = Bernoulli(p)

    def forward(self, batch_shape: Tuple[int, ...], same_on_batch: bool = False) -> Dict[str, Tensor]:
        batch_size = batch_shape[0]
        probs_mask: Tensor = _adapted_sampling((batch_size,), self.sampler, same_on_batch).bool()
        return {"probs": probs_mask}


def random_prob_generator(
    batch_size: int,
    p: float = 0.5,
    same_on_batch: bool = False,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    r"""Generate random probabilities for a batch of inputs.

    Args:
        batch_size (int): the number of images.
        p (float): probability to generate an 1-d binary mask. Default value is 0.5.
        same_on_batch (bool): apply the same transformation across the batch. Default: False.
        device (torch.device): the device on which the random numbers will be generated. Default: cpu.
        dtype (torch.dtype): the data type of the generated random numbers. Default: float32.

    Returns:
        Tensor: parameters to be passed for transformation.
            - probs (Tensor): element-wise probabilities with a shape of (B,).

    Note:
        The generated random numbers are not reproducible across different devices and dtypes.

    """
    if device is None:
        device = torch.device("cpu")
    _common_param_check(batch_size, same_on_batch)
    if not isinstance(p, (int, float)) or p > 1 or p < 0:
        raise TypeError(f"The probability should be a float number within [0, 1]. Got {type(p)}.")

    _bernoulli = Bernoulli(tensor(float(p), device=device, dtype=dtype))
    probs_mask: Tensor = _adapted_sampling((batch_size,), _bernoulli, same_on_batch).bool()

    return probs_mask
