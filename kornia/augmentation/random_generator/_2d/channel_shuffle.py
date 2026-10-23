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

from kornia.augmentation.random_generator.base import RandomGeneratorBase, UniformDistribution
from kornia.augmentation.utils import _adapted_rsampling, _common_param_check

__all__ = ["ChannelShuffleGenerator"]


class ChannelShuffleGenerator(RandomGeneratorBase):
    r"""Generate random channel permutation indices for a batch of images.

    Returns:
        A dictionary containing the channel permutation indices.
            - ``channels``: Long tensor of permuted channel indices with a shape of ``(B, C)``.

    Note:
        The generated random numbers are not reproducible across different devices and dtypes. By default,
        the parameters will be generated on CPU. This can be changed by calling
        ``self.set_rng_device_and_dtype(device="cuda", dtype=torch.float64)``.

    """

    def __repr__(self) -> str:
        return ""

    def make_samplers(self, device: torch.device, dtype: torch.dtype) -> None:
        """Create a Uniform(0, 1) sampler used to generate random channel scores for argsort shuffling."""
        self.shuffle_sampler = UniformDistribution(
            torch.tensor(0.0, device=device, dtype=dtype),
            torch.tensor(1.0, device=device, dtype=dtype),
            validate_args=False,
        )

    def forward(self, batch_shape: tuple[int, ...], same_on_batch: bool = False) -> dict[str, torch.Tensor]:
        """Generate channel permutation indices.

        Args:
            batch_shape: Shape of the input batch ``(B, C, H, W)``.
            same_on_batch: If ``True``, the same permutation is applied to all items in the batch.

        Returns:
            Dictionary with key ``"channels"`` mapping to a ``(B, C)`` long tensor of permuted indices.
        """
        batch_size, C = batch_shape[0], batch_shape[1]
        _common_param_check(batch_size, same_on_batch)

        # Sample uniform scores of shape (B, C); _adapted_rsampling handles same_on_batch by
        # generating (1, C) and repeating across the batch dimension when same_on_batch=True.
        scores = _adapted_rsampling((batch_size, C), self.shuffle_sampler, same_on_batch)
        channels = scores.argsort(dim=1).to(dtype=torch.long)
        return {"channels": channels}
