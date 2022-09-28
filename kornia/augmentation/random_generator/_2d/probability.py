from __future__ import annotations

import torch
from torch.distributions import Bernoulli

from kornia.augmentation.random_generator.base import RandomGeneratorBase
from kornia.augmentation.utils import _adapted_sampling, _common_param_check


class ProbabilityGenerator(RandomGeneratorBase):
    r"""Generate random probabilities for a batch of inputs.

    Args:
        p: probability to generate an 1-d binary mask. Default value is 0.5.

    Returns:
        A dict of parameters to be passed for transformation.
            - probs (torch.Tensor): element-wise probabilities with a shape of (B,).

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

    def forward(self, batch_shape: torch.Size, same_on_batch: bool = False) -> dict[str, torch.Tensor]:  # type:ignore
        batch_size = batch_shape[0]
        probs_mask: torch.Tensor = _adapted_sampling((batch_size,), self.sampler, same_on_batch).bool()
        return dict(probs=probs_mask)


def random_prob_generator(
    batch_size: int,
    p: float = 0.5,
    same_on_batch: bool = False,
    device: torch.device = torch.device('cpu'),
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    r"""Generate random probabilities for a batch of inputs.

    Args:
        batch_size (int): the number of images.
        p (float): probability to generate an 1-d binary mask. Default value is 0.5.
        same_on_batch (bool): apply the same transformation across the batch. Default: False.
        device (torch.device): the device on which the random numbers will be generated. Default: cpu.
        dtype (torch.dtype): the data type of the generated random numbers. Default: float32.

    Returns:
        torch.Tensor: parameters to be passed for transformation.
            - probs (torch.Tensor): element-wise probabilities with a shape of (B,).

    Note:
        The generated random numbers are not reproducible across different devices and dtypes.
    """
    _common_param_check(batch_size, same_on_batch)
    if not isinstance(p, (int, float)) or p > 1 or p < 0:
        raise TypeError(f"The probability should be a float number within [0, 1]. Got {type(p)}.")

    _bernoulli = Bernoulli(torch.tensor(float(p), device=device, dtype=dtype))
    probs_mask: torch.Tensor = _adapted_sampling((batch_size,), _bernoulli, same_on_batch).bool()

    return probs_mask
