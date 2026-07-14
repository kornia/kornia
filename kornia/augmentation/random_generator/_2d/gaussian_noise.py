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

import torch
from torch.distributions import Normal

from kornia.augmentation.random_generator.base import RandomGeneratorBase
from kornia.augmentation.utils import _common_param_check

__all__ = ["GaussianNoiseGenerator"]


class GaussianNoiseGenerator(RandomGeneratorBase):
    r"""Generate random Gaussian noise tensors for a batch of images.

    Args:
        mean: Mean of the Gaussian distribution.
        std: Standard deviation of the Gaussian distribution.

    Returns:
        A dictionary containing the pre-generated noise tensor.
            - ``gaussian_noise``: Float tensor of noise values with a shape of ``(B, C, H, W)``
              (or ``(1, C, H, W)`` when ``same_on_batch=True``, to be expanded in ``apply_transform``).

    Note:
        The generated random numbers are not reproducible across different devices and dtypes. By default,
        the parameters will be generated on CPU. This can be changed by calling
        ``self.set_rng_device_and_dtype(device="cuda", dtype=torch.float64)``.

    """

    def __init__(self, mean: float, std: float) -> None:
        super().__init__()
        self.mean = mean
        self.std = std

    def __repr__(self) -> str:
        return f"mean={self.mean}, std={self.std}"

    def make_samplers(self, device: torch.device, dtype: torch.dtype) -> None:
        """Store device and dtype for the Normal distribution (created on demand in forward)."""
        # Normal is created fresh in forward() so device/dtype are always consistent.
        # No persistent sampler is needed beyond what the base class tracks.

    def forward(self, batch_shape: tuple[int, ...], same_on_batch: bool = False) -> dict[str, torch.Tensor]:
        """Generate a Gaussian noise tensor matching the input batch shape.

        Args:
            batch_shape: Shape of the input batch ``(B, C, H, W)``.
            same_on_batch: If ``True``, generate a single noise map ``(1, C, H, W)`` shared
                across all batch items. The augmentation's ``apply_transform`` is responsible
                for expanding it to the full batch size.

        Returns:
            Dictionary with key ``"gaussian_noise"`` containing the noise tensor.
        """
        batch_size, C, H, W = batch_shape
        _common_param_check(batch_size, same_on_batch)

        device = self.device if self.device is not None else torch.device("cpu")
        dtype = self.dtype if self.dtype is not None else torch.get_default_dtype()

        dist = Normal(
            torch.tensor(self.mean, device=device, dtype=dtype),
            torch.tensor(self.std, device=device, dtype=dtype),
        )

        # When same_on_batch, generate a single (1, C, H, W) map — apply_transform will expand it.
        sample_B = 1 if same_on_batch else batch_size
        gaussian_noise = dist.rsample((sample_B, C, H, W))

        return {"gaussian_noise": gaussian_noise}
