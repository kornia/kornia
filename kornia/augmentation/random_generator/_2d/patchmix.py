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

from typing import Dict, Optional, Tuple, Union

import torch
from torch.distributions import Beta, Uniform

from kornia.augmentation.random_generator.base import RandomGeneratorBase
from kornia.augmentation.utils import _adapted_sampling, _common_param_check
from kornia.core.utils import _extract_device_dtype

__all__ = ["PatchMixGenerator"]


class PatchMixGenerator(RandomGeneratorBase):
    r"""Generate patchmix indexes and lambdas for a batch of inputs.

    Args:
        alpha (float): hyperparameter for generating cut size from beta distribution.
        patch_size (int): size of the patch to be swapped.
        p (float): probability of applying patchmix.

    Returns:
        params Dict[str, torch.Tensor]: parameters to be passed for transformation.
            - mix_pairs (torch.Tensor): element-wise probabilities with a shape of (B).
            - patch_coords (torch.Tensor): top-left coordinates of the patch (B, 2).
            - lam (torch.Tensor): mixing parameter (B).

    Note:
        The generated random numbers are not reproducible across different devices and dtypes. By default,
        the parameters will be generated on CPU in float32. This can be changed by calling
        ``self.set_rng_device_and_dtype(device="cuda", dtype=torch.float64)``.
    """

    def __init__(
        self,
        alpha: float = 1.0,
        patch_size: int = 16,
        p: float = 1.0,
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.patch_size = patch_size
        self.p = p

    def __repr__(self) -> str:
        repr = f"alpha={self.alpha}, patch_size={self.patch_size}, p={self.p}"
        return repr

    def make_samplers(self, device: torch.device, dtype: torch.dtype) -> None:
        self.beta_sampler = Beta(
            torch.tensor(self.alpha, device=device, dtype=dtype),
            torch.tensor(self.alpha, device=device, dtype=dtype),
        )
        self.rand_sampler = Uniform(
            torch.tensor(0.0, device=device, dtype=dtype),
            torch.tensor(1.0, device=device, dtype=dtype),
        )
        self.pair_sampler = Uniform(
            torch.tensor(0.0, device=device, dtype=dtype),
            torch.tensor(1.0, device=device, dtype=dtype),
        )

    def forward(self, batch_shape: Tuple[int, ...], same_on_batch: bool = False) -> Dict[str, torch.Tensor]:
        batch_size = batch_shape[0]
        height = batch_shape[-2]
        width = batch_shape[-1]

        _device, _dtype = _extract_device_dtype([self.alpha])
        _common_param_check(batch_size, same_on_batch)

        if batch_size == 0:
            return {
                "mix_pairs": torch.zeros([0], device=_device, dtype=torch.long),
                "patch_coords": torch.zeros([0, 2], device=_device, dtype=torch.long),
                "lam": torch.zeros([0], device=_device, dtype=_dtype),
            }

        with torch.no_grad():
            mix_pairs: torch.Tensor = (
                _adapted_sampling((batch_size,), self.pair_sampler, same_on_batch)
                .to(device=_device, dtype=_dtype)
                .argsort(dim=0)
            )
            # Sample lam from beta distribution
            lam = _adapted_sampling((batch_size,), self.beta_sampler, same_on_batch).to(device=_device, dtype=_dtype)

            # Sample patch coordinates
            # height - patch_size + 1
            max_y = height - self.patch_size
            max_x = width - self.patch_size

            y = (_adapted_sampling((batch_size,), self.rand_sampler, same_on_batch) * (max_y + 1)).floor().to(torch.long)
            x = (_adapted_sampling((batch_size,), self.rand_sampler, same_on_batch) * (max_x + 1)).floor().to(torch.long)

        return {
            "mix_pairs": mix_pairs.to(device=_device, dtype=torch.long),
            "patch_coords": torch.stack([x, y], dim=1).to(device=_device, dtype=torch.long),
            "lam": lam,
        }
