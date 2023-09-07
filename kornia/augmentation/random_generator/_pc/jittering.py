from typing import Dict, Tuple

import torch
from torch.distributions import Normal

from kornia.augmentation.random_generator.base import RandomGeneratorBase
from kornia.augmentation.utils import _adapted_sampling
from kornia.core import Tensor

__all__ = ["JitteringGeneratorPC"]


class JitteringGeneratorPC(RandomGeneratorBase):
    r"""Jitters point clouds coordiantes by a 0-mean Gaussian.

    Args:
        jitter_scale: standard deviation for Gaussian distribution.

    Returns:
        A dict of parameters to be passed for transformation.
            - jitter (Tensor): element-wise probabilities with a shape of (B, N, 3).
    """

    def __init__(self, jitter_scale: float = 0.01) -> None:
        super().__init__()
        self.jitter_scale = jitter_scale

    def __repr__(self) -> str:
        repr = f"jitter_scale={self.jitter_scale}"
        return repr

    def make_samplers(self, device: torch.device, dtype: torch.dtype) -> None:
        loc = torch.tensor(0.0, device=device, dtype=dtype)
        scale = torch.tensor(self.jitter_scale, device=device, dtype=dtype)
        self.sampler = Normal(loc, scale)

    def forward(self, batch_shape: Tuple[int, ...], same_on_batch: bool = False) -> Dict[str, Tensor]:
        batch_size, N = batch_shape[0], batch_shape[1]
        jitter: Tensor = _adapted_sampling((batch_size, N, 3), self.sampler, same_on_batch)
        return {"jitter": jitter}
