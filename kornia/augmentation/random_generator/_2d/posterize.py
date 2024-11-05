from typing import Dict, Tuple, Union

import torch

from kornia.augmentation.random_generator.base import RandomGeneratorBase, UniformDistribution
from kornia.augmentation.utils import _adapted_rsampling, _common_param_check, _joint_range_check
from kornia.core import Tensor, as_tensor
from kornia.utils.helpers import _extract_device_dtype

__all__ = ["PosterizeGenerator"]


class PosterizeGenerator(RandomGeneratorBase):
    r"""Generate random posterize parameters for a batch of images.

    Args:
        bits: floats that ranged from (0, 8], in which 0 gives black image and 8 gives the original.
            If float x, bits will be generated from (x, 8).
            If tuple (x, y), bits will be generated from (x, y).

    Returns:
        A dict of parameters to be passed for transformation.
            - bits_factor (Tensor): element-wise bit factors with a shape of (B,).

    Note:
        The generated random numbers are not reproducible across different devices and dtypes. By default,
        the parameters will be generated on CPU in float32. This can be changed by calling
        ``self.set_rng_device_and_dtype(device="cuda", dtype=torch.float64)``.
    """

    def __init__(self, bits: Union[float, Tuple[float, float], Tensor]) -> None:
        super().__init__()
        self.bits_factor = bits

    def __repr__(self) -> str:
        repr = f"bits={self.bits_factor}"
        return repr

    def make_samplers(self, device: torch.device, dtype: torch.dtype) -> None:
        bits = as_tensor(self.bits_factor, device=device, dtype=dtype)
        if len(bits.size()) == 0:
            bits = bits.repeat(2)
            bits[1] = 8
        elif not (len(bits.size()) == 1 and bits.size(0) == 2):
            raise ValueError(f"'bits' shall be either a scalar or a length 2 tensor. Got {bits}.")
        _joint_range_check(bits, "bits", (0, 8))
        self.bit_sampler = UniformDistribution(bits[0], bits[1], validate_args=False)

    def forward(self, batch_shape: Tuple[int, ...], same_on_batch: bool = False) -> Dict[str, Tensor]:
        batch_size = batch_shape[0]
        _common_param_check(batch_size, same_on_batch)
        _device, _ = _extract_device_dtype([self.bits_factor if isinstance(self.bits_factor, Tensor) else None])
        bits_factor = _adapted_rsampling((batch_size,), self.bit_sampler, same_on_batch)
        return {"bits_factor": bits_factor.round().to(device=_device, dtype=torch.int32)}
