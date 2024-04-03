from typing import Dict, Tuple, Union

import torch
from torch import Tensor

from kornia.augmentation.random_generator.base import (
    RandomGeneratorBase,
    UniformDistribution,
)
from kornia.augmentation.utils import (
    _adapted_rsampling,
    _common_param_check,
    _joint_range_check,
)

__all__ = ["RandomGaussianBlurGenerator"]


class RandomGaussianBlurGenerator(RandomGeneratorBase):
    r"""Generate random gaussian blur parameters for a batch of images.

    Args:
        sigma: The range to uniformly sample the standard deviation for the Gaussian kernel.

    Returns:
        A dict of parameters to be passed for transformation.
            - sigma: element-wise standard deviation with a shape of (B,).

    Note:
        The generated random numbers are not reproducible across different devices and dtypes. By default,
        the parameters will be generated on CPU in float32. This can be changed by calling
        ``self.set_rng_device_and_dtype(device="cuda", dtype=torch.float64)``.
    """

    def __init__(self, sigma: Union[Tuple[float, float], Tensor] = (0.1, 2.0)) -> None:
        super().__init__()
        if sigma[1] < sigma[0]:
            raise TypeError(f"sigma_max should be higher than sigma_min: {sigma} passed.")

        self.sigma = sigma
        self.sigma_sampler: UniformDistribution

    def __repr__(self) -> str:
        repr_buf = f"sigma={self.sigma}"
        return repr_buf

    def make_samplers(self, device: torch.device, dtype: torch.dtype) -> None:
        if not isinstance(self.sigma, (torch.Tensor)):
            sigma = torch.tensor(self.sigma, device=device, dtype=dtype)
        else:
            sigma = self.sigma.to(device=device, dtype=dtype)

        _joint_range_check(sigma, "sigma", (0, float("inf")))

        self.sigma_sampler = UniformDistribution(sigma[0], sigma[1], validate_args=False)

    def forward(self, batch_shape: Tuple[int, ...], same_on_batch: bool = False) -> Dict[str, Tensor]:
        batch_size = batch_shape[0]
        _common_param_check(batch_size, same_on_batch)
        sigma = _adapted_rsampling((batch_size,), self.sigma_sampler, same_on_batch)
        return {"sigma": sigma}
