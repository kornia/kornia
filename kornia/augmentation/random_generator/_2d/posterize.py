from __future__ import annotations

import torch
from torch.distributions import Uniform

from kornia.augmentation.random_generator.base import RandomGeneratorBase
from kornia.augmentation.utils import _adapted_rsampling, _adapted_uniform, _common_param_check, _joint_range_check
from kornia.utils.helpers import _deprecated, _extract_device_dtype


class PosterizeGenerator(RandomGeneratorBase):
    r"""Generate random posterize parameters for a batch of images.

    Args:
        bits: Integer that ranged from (0, 8], in which 0 gives black image and 8 gives the original.
            If int x, bits will be generated from (x, 8).
            If tuple (x, y), bits will be generated from (x, y).

    Returns:
        A dict of parameters to be passed for transformation.
            - bits_factor (torch.Tensor): element-wise bit factors with a shape of (B,).

    Note:
        The generated random numbers are not reproducible across different devices and dtypes. By default,
        the parameters will be generated on CPU in float32. This can be changed by calling
        ``self.set_rng_device_and_dtype(device="cuda", dtype=torch.float64)``.
    """

    def __init__(self, bits: int | tuple[int, int] | torch.Tensor) -> None:
        super().__init__()
        self.bits = bits

    def __repr__(self) -> str:
        repr = f"bits={self.bits}"
        return repr

    def make_samplers(self, device: torch.device, dtype: torch.dtype) -> None:
        bits = torch.as_tensor(self.bits, device=device, dtype=dtype)
        if len(bits.size()) == 0:
            bits = bits.repeat(2)
            bits[1] = 8
        elif not (len(bits.size()) == 1 and bits.size(0) == 2):
            raise ValueError(f"'bits' shall be either a scalar or a length 2 tensor. Got {bits}.")
        _joint_range_check(bits, 'bits', (0, 8))
        self.bit_sampler = Uniform(bits[0], bits[1], validate_args=False)

    def forward(self, batch_shape: torch.Size, same_on_batch: bool = False) -> dict[str, torch.Tensor]:  # type:ignore
        batch_size = batch_shape[0]
        _common_param_check(batch_size, same_on_batch)
        _device, _ = _extract_device_dtype([self.bits if isinstance(self.bits, torch.Tensor) else None])
        bits_factor = _adapted_rsampling((batch_size,), self.bit_sampler, same_on_batch)
        return dict(bits_factor=bits_factor.to(device=_device, dtype=torch.int32))


@_deprecated()
def random_posterize_generator(
    batch_size: int,
    bits: torch.Tensor = torch.tensor([3, 5]),
    same_on_batch: bool = False,
    device: torch.device = torch.device('cpu'),
    dtype: torch.dtype = torch.float32,
) -> dict[str, torch.Tensor]:
    r"""Generate random posterize parameters for a batch of images.

    Args:
        batch_size (int): the number of images.
        bits (int or tuple): Takes in an integer tuple tensor that ranged from 0 ~ 8. Default value is [3, 5].
        same_on_batch (bool): apply the same transformation across the batch. Default: False.
        device (torch.device): the device on which the random numbers will be generated. Default: cpu.
        dtype (torch.dtype): the data type of the generated random numbers. Default: float32.

    Returns:
        params Dict[str, torch.Tensor]: parameters to be passed for transformation.
            - bits_factor (torch.Tensor): element-wise bit factors with a shape of (B,).

    Note:
        The generated random numbers are not reproducible across different devices and dtypes.
    """
    _common_param_check(batch_size, same_on_batch)
    _joint_range_check(bits, 'bits', (0, 8))
    bits_factor = _adapted_uniform(
        (batch_size,), bits[0].to(device=device, dtype=dtype), bits[1].to(device=device, dtype=dtype), same_on_batch
    ).int()

    return dict(bits_factor=bits_factor.to(device=bits.device, dtype=torch.int32))
