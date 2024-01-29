from __future__ import annotations

import torch

from kornia.augmentation.random_generator.base import RandomGeneratorBase, UniformDistribution
from kornia.augmentation.utils import _adapted_rsampling, _adapted_uniform, _common_param_check, _range_bound
from kornia.core import Tensor
from kornia.utils import _extract_device_dtype


class SaltAndPepperGenerator(RandomGeneratorBase):
    r"""Generate random Salt and Pepper noise parameters for a batch of images.

    Args:
        amount: A tuple representing the range for the amount of noise to apply.
        salt_and_pepper: A tuple representing the range for the ratio of Salt and Pepper noise.

    Returns:
        A dictionary of parameters to be passed for transformation.
            - amount_factor: Element-wise factors determining the amount of noise with a shape of (B,).
            - salt_and_pepper_factor: Element-wise factors determining the ratio of Salt and Pepper noise
                with a shape of (B,).
            - mask_salt: Binary masks (bool) indicating the presence of Salt noise with a shape of (B, C, H, W).
            - mask_pepper: Binary masks (bool) indicating the presence of Pepper noise with a shape of (B, C, H, W).

    Note:
        The generated random numbers are not reproducible across different devices and dtypes. By default,
        the parameters will be generated on CPU. This can be changed by calling
        ``self.set_rng_device_and_dtype(device="cuda", dtype=torch.float64)``.
    """

    def __init__(
        self,
        amount: tuple[float, float],
        salt_and_pepper: tuple[float, float],
    ) -> None:
        super().__init__()
        self.amount = amount
        self.salt_and_pepper = salt_and_pepper

    def __repr__(self) -> str:
        r"""Return a string representation of the object."""
        repr = f"amount={self.amount}, salt_and_pepper={self.salt_and_pepper}"
        return repr

    def make_samplers(self, device: torch.device, dtype: torch.dtype) -> None:
        r"""Create samplers for generating random noise parameters."""
        amount = _range_bound(
            self.amount,
            "amount",
        ).to(device, dtype)
        salt_and_pepper = _range_bound(
            self.salt_and_pepper,
            "salt_and_pepper",
        ).to(device, dtype)

        self.amount_sampler = UniformDistribution(amount[0], amount[1], validate_args=False)
        self.salt_and_pepper_sampler = UniformDistribution(salt_and_pepper[0], salt_and_pepper[1], validate_args=False)

    def forward(self, batch_shape: tuple[int, ...], same_on_batch: bool = False) -> dict[str, Tensor]:
        r"""Generate Salt and Pepper noise masks for a batch of images."""
        batch_size, C, H, W = batch_shape
        _common_param_check(batch_size, same_on_batch)
        _device, _dtype = _extract_device_dtype([self.amount, self.salt_and_pepper])

        amount_factor = _adapted_rsampling((batch_size,), self.amount_sampler, same_on_batch).to(
            device=_device, dtype=_dtype
        )
        salt_and_pepper_factor = _adapted_rsampling((batch_size,), self.salt_and_pepper_sampler, same_on_batch).to(
            device=_device, dtype=_dtype
        )

        ## Generate noise masks.
        mask_noise = _adapted_uniform(
            (batch_size, 1, H, W), low=0.0, high=1.0, same_on_batch=same_on_batch
        ) < amount_factor.view(batch_size, 1, 1, 1)
        mask_salt = _adapted_uniform(
            (batch_size, 1, H, W), low=0.0, high=1.0, same_on_batch=same_on_batch
        ) < salt_and_pepper_factor.view(batch_size, 1, 1, 1)

        # If the number of channels is greater than one (3), replicate the generated mask for each channel.
        if C > 1:
            mask_noise = mask_noise.repeat(1, C, 1, 1)
            mask_salt = mask_salt.repeat(1, C, 1, 1)
        mask_pepper = (~mask_salt & mask_noise).to(device=_device)
        mask_salt = (mask_salt & mask_noise).to(device=_device)

        return {
            "amount_factor": amount_factor,
            "salt_and_pepper_factor": salt_and_pepper_factor,
            "mask_salt": mask_salt,
            "mask_pepper": mask_pepper,
        }
