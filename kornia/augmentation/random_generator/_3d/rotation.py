from typing import Dict, Tuple, Union

import torch

from kornia.augmentation.random_generator.base import RandomGeneratorBase, UniformDistribution
from kornia.augmentation.utils import _adapted_rsampling, _common_param_check, _tuple_range_reader
from kornia.core import Tensor
from kornia.utils.helpers import _extract_device_dtype


class RotationGenerator3D(RandomGeneratorBase):
    r"""Get parameters for ``rotate`` for a random 3D rotate transform.

    Args:
        degrees: Range of yaw (x-axis), pitch (y-axis), roll (z-axis) to select from.
            If degrees is a number, then yaw, pitch, roll will be generated from the range of (-degrees, +degrees).
            If degrees is a tuple of (min, max), then yaw, pitch, roll will be generated from the range of (min, max).
            If degrees is a list of floats [a, b, c], then yaw, pitch, roll will be generated from (-a, a), (-b, b)
            and (-c, c).
            If degrees is a list of tuple ((a, b), (m, n), (x, y)), then yaw, pitch, roll will be generated from
            (a, b), (m, n) and (x, y).
            Set to 0 to deactivate rotations.

    Returns:
        A dict of parameters to be passed for transformation.
            - yaw (Tensor): element-wise rotation yaws with a shape of (B,).
            - pitch (Tensor): element-wise rotation pitches with a shape of (B,).
            - roll (Tensor): element-wise rotation rolls with a shape of (B,).

    Note:
        The generated random numbers are not reproducible across different devices and dtypes. By default,
        the parameters will be generated on CPU in float32. This can be changed by calling
        ``self.set_rng_device_and_dtype(device="cuda", dtype=torch.float64)``.
    """

    def __init__(
        self,
        degrees: Union[
            Tensor,
            float,
            Tuple[float, float, float],
            Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]],
        ],
    ) -> None:
        super().__init__()
        self.degrees = degrees

    def __repr__(self) -> str:
        repr = f"degrees={self.degrees}"
        return repr

    def make_samplers(self, device: torch.device, dtype: torch.dtype) -> None:
        degrees = _tuple_range_reader(self.degrees, 3, device, dtype)
        self.yaw_sampler = UniformDistribution(degrees[0][0], degrees[0][1], validate_args=False)
        self.pitch_sampler = UniformDistribution(degrees[1][0], degrees[1][1], validate_args=False)
        self.roll_sampler = UniformDistribution(degrees[2][0], degrees[2][1], validate_args=False)

    def forward(self, batch_shape: Tuple[int, ...], same_on_batch: bool = False) -> Dict[str, Tensor]:
        batch_size = batch_shape[0]
        _common_param_check(batch_size, same_on_batch)
        _device, _dtype = _extract_device_dtype([self.degrees])

        return {
            "yaw": _adapted_rsampling((batch_size,), self.yaw_sampler, same_on_batch).to(device=_device, dtype=_dtype),
            "pitch": _adapted_rsampling((batch_size,), self.pitch_sampler, same_on_batch).to(
                device=_device, dtype=_dtype
            ),
            "roll": _adapted_rsampling((batch_size,), self.roll_sampler, same_on_batch).to(
                device=_device, dtype=_dtype
            ),
        }
