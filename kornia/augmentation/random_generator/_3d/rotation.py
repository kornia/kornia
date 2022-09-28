from __future__ import annotations

import torch
from torch.distributions import Uniform

from kornia.augmentation.random_generator.base import RandomGeneratorBase
from kornia.augmentation.utils import _adapted_rsampling, _adapted_uniform, _common_param_check, _tuple_range_reader
from kornia.utils.helpers import _deprecated, _extract_device_dtype


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
            - yaw (torch.Tensor): element-wise rotation yaws with a shape of (B,).
            - pitch (torch.Tensor): element-wise rotation pitches with a shape of (B,).
            - roll (torch.Tensor): element-wise rotation rolls with a shape of (B,).

    Note:
        The generated random numbers are not reproducible across different devices and dtypes. By default,
        the parameters will be generated on CPU in float32. This can be changed by calling
        ``self.set_rng_device_and_dtype(device="cuda", dtype=torch.float64)``.
    """

    def __init__(
        self,
        degrees: (
            torch.Tensor
            | float
            | tuple[float, float, float]
            | tuple[tuple[float, float], tuple[float, float], tuple[float, float]]
        ),
    ) -> None:
        super().__init__()
        self.degrees = degrees

    def __repr__(self) -> str:
        repr = f"degrees={self.degrees}"
        return repr

    def make_samplers(self, device: torch.device, dtype: torch.dtype) -> None:
        degrees = _tuple_range_reader(self.degrees, 3, device, dtype)
        self.yaw_sampler = Uniform(degrees[0][0], degrees[0][1], validate_args=False)
        self.pitch_sampler = Uniform(degrees[1][0], degrees[1][1], validate_args=False)
        self.roll_sampler = Uniform(degrees[2][0], degrees[2][1], validate_args=False)

    def forward(self, batch_shape: torch.Size, same_on_batch: bool = False) -> dict[str, torch.Tensor]:  # type:ignore
        batch_size = batch_shape[0]
        _common_param_check(batch_size, same_on_batch)
        _device, _dtype = _extract_device_dtype([self.degrees])

        return dict(
            yaw=_adapted_rsampling((batch_size,), self.yaw_sampler, same_on_batch).to(device=_device, dtype=_dtype),
            pitch=_adapted_rsampling((batch_size,), self.pitch_sampler, same_on_batch).to(device=_device, dtype=_dtype),
            roll=_adapted_rsampling((batch_size,), self.roll_sampler, same_on_batch).to(device=_device, dtype=_dtype),
        )


@_deprecated(replace_with=RotationGenerator3D.__name__)
def random_rotation_generator3d(
    batch_size: int,
    degrees: torch.Tensor,
    same_on_batch: bool = False,
    device: torch.device = torch.device('cpu'),
    dtype: torch.dtype = torch.float32,
) -> dict[str, torch.Tensor]:
    r"""Get parameters for ``rotate`` for a random rotate transform.

    Args:
        batch_size (int): the tensor batch size.
        degrees (torch.Tensor): Ranges of degrees (3, 2) for yaw, pitch and roll.
        same_on_batch (bool): apply the same transformation across the batch. Default: False.
        device (torch.device): the device on which the random numbers will be generated. Default: cpu.
        dtype (torch.dtype): the data type of the generated random numbers. Default: float32.

    Returns:
        params Dict[str, torch.Tensor]: parameters to be passed for transformation.
            - yaw (torch.Tensor): element-wise rotation yaws with a shape of (B,).
            - pitch (torch.Tensor): element-wise rotation pitches with a shape of (B,).
            - roll (torch.Tensor): element-wise rotation rolls with a shape of (B,).
    """
    if degrees.shape != torch.Size([3, 2]):
        raise AssertionError(f"'degrees' must be the shape of (3, 2). Got {degrees.shape}.")
    _device, _dtype = _extract_device_dtype([degrees])
    degrees = degrees.to(device=device, dtype=dtype)
    yaw = _adapted_uniform((batch_size,), degrees[0][0], degrees[0][1], same_on_batch)
    pitch = _adapted_uniform((batch_size,), degrees[1][0], degrees[1][1], same_on_batch)
    roll = _adapted_uniform((batch_size,), degrees[2][0], degrees[2][1], same_on_batch)

    return dict(
        yaw=yaw.to(device=_device, dtype=_dtype),
        pitch=pitch.to(device=_device, dtype=_dtype),
        roll=roll.to(device=_device, dtype=_dtype),
    )
