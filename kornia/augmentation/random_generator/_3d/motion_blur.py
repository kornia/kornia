from __future__ import annotations

from typing import Dict, Tuple, Union

import torch
from torch.distributions import Uniform

from kornia.augmentation.random_generator.base import RandomGeneratorBase
from kornia.augmentation.utils import (
    _adapted_rsampling,
    _adapted_uniform,
    _common_param_check,
    _joint_range_check,
    _range_bound,
    _tuple_range_reader,
)
from kornia.utils.helpers import _deprecated, _extract_device_dtype


class MotionBlurGenerator3D(RandomGeneratorBase):
    r"""Get parameters for motion blur.

    Args:
        kernel_size: motion kernel size (odd and positive).
            If int, the kernel will have a fixed size.
            If Tuple[int, int], it will randomly generate the value from the range batch-wisely.
        angle: angle of the motion blur in degrees (anti-clockwise rotation).
            If float, it will generate the value from (-angle, angle).
        direction: forward/backward direction of the motion blur.
            Lower values towards -1.0 will point the motion blur towards the back (with angle provided via angle),
            while higher values towards 1.0 will point the motion blur forward. A value of 0.0 leads to a
            uniformly (but still angled) motion blur.
            If float, it will generate the value from (-direction, direction).
            If Tuple[int, int], it will randomly generate the value from the range.

    Returns:
        A dict of parameters to be passed for transformation.
            - ksize_factor (torch.Tensor): element-wise kernel size factors with a shape of (B,).
            - angle_factor (torch.Tensor): element-wise angle factors with a shape of (B,).
            - direction_factor (torch.Tensor): element-wise direction factors with a shape of (B,).

    Note:
        The generated random numbers are not reproducible across different devices and dtypes. By default,
        the parameters will be generated on CPU in float32. This can be changed by calling
        ``self.set_rng_device_and_dtype(device="cuda", dtype=torch.float64)``.
    """

    def __init__(
        self,
        kernel_size: int | tuple[int, int],
        angle: (
            torch.Tensor
            | float
            | tuple[float, float, float]
            | tuple[tuple[float, float], tuple[float, float], tuple[float, float]]
        ),
        direction: torch.Tensor | float | tuple[float, float],
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.angle = angle
        self.direction = direction

    def __repr__(self) -> str:
        repr = f"kernel_size={self.kernel_size}, angle={self.angle}, direction={self.direction}"
        return repr

    def make_samplers(self, device: torch.device, dtype: torch.dtype) -> None:
        angle: torch.Tensor = _tuple_range_reader(self.angle, 3, device=device, dtype=dtype)
        direction = _range_bound(self.direction, 'direction', center=0.0, bounds=(-1, 1)).to(device=device, dtype=dtype)
        if isinstance(self.kernel_size, int):
            if not (self.kernel_size >= 3 and self.kernel_size % 2 == 1):
                raise AssertionError(f"`kernel_size` must be odd and greater than 3. Got {self.kernel_size}.")
            self.ksize_sampler = Uniform(self.kernel_size // 2, self.kernel_size // 2, validate_args=False)
        elif isinstance(self.kernel_size, tuple):
            # kernel_size is fixed across the batch
            if len(self.kernel_size) != 2:
                raise AssertionError(f"`kernel_size` must be (2,) if it is a tuple. Got {self.kernel_size}.")
            self.ksize_sampler = Uniform(self.kernel_size[0] // 2, self.kernel_size[1] // 2, validate_args=False)
        else:
            raise TypeError(f"Unsupported type: {type(self.kernel_size)}")

        self.yaw_sampler = Uniform(angle[0][0], angle[0][1], validate_args=False)
        self.pitch_sampler = Uniform(angle[1][0], angle[1][1], validate_args=False)
        self.roll_sampler = Uniform(angle[2][0], angle[2][1], validate_args=False)
        self.direction_sampler = Uniform(direction[0], direction[1], validate_args=False)

    def forward(self, batch_shape: torch.Size, same_on_batch: bool = False) -> dict[str, torch.Tensor]:  # type:ignore
        batch_size = batch_shape[0]
        _common_param_check(batch_size, same_on_batch)
        # self.ksize_factor.expand((batch_size, -1))
        _device, _dtype = _extract_device_dtype([self.angle, self.direction])
        yaw_factor = _adapted_rsampling((batch_size,), self.yaw_sampler, same_on_batch)
        pitch_factor = _adapted_rsampling((batch_size,), self.pitch_sampler, same_on_batch)
        roll_factor = _adapted_rsampling((batch_size,), self.roll_sampler, same_on_batch)
        angle_factor = torch.stack([yaw_factor, pitch_factor, roll_factor], dim=1)

        direction_factor = _adapted_rsampling((batch_size,), self.direction_sampler, same_on_batch)
        ksize_factor = _adapted_rsampling((batch_size,), self.ksize_sampler, same_on_batch).int() * 2 + 1

        return dict(
            ksize_factor=ksize_factor.to(device=_device, dtype=torch.int32),
            angle_factor=angle_factor.to(device=_device, dtype=_dtype),
            direction_factor=direction_factor.to(device=_device, dtype=_dtype),
        )


@_deprecated(replace_with=MotionBlurGenerator3D.__name__)
def random_motion_blur_generator3d(
    batch_size: int,
    kernel_size: int | tuple[int, int],
    angle: torch.Tensor,
    direction: torch.Tensor,
    same_on_batch: bool = False,
    device: torch.device = torch.device('cpu'),
    dtype: torch.dtype = torch.float32,
) -> dict[str, torch.Tensor]:
    r"""Get parameters for motion blur.

    Args:
        batch_size (int): the tensor batch size.
        kernel_size (int or (int, int)): motion kernel size (odd and positive) or range.
        angle (torch.Tensor): yaw, pitch and roll range of the motion blur in degrees :math:`(3, 2)`.
        direction (torch.Tensor): forward/backward direction of the motion blur.
            Lower values towards -1.0 will point the motion blur towards the back (with
            angle provided via angle), while higher values towards 1.0 will point the motion
            blur forward. A value of 0.0 leads to a uniformly (but still angled) motion blur.
        same_on_batch (bool): apply the same transformation across the batch. Default: False.
        device (torch.device): the device on which the random numbers will be generated. Default: cpu.
        dtype (torch.dtype): the data type of the generated random numbers. Default: float32.

    Returns:
        params Dict[str, torch.Tensor]: parameters to be passed for transformation.
            - ksize_factor (torch.Tensor): element-wise kernel size factors with a shape of (B,).
            - angle_factor (torch.Tensor): element-wise center with a shape of (B,).
            - direction_factor (torch.Tensor): element-wise scales with a shape of (B,).

    Note:
        The generated random numbers are not reproducible across different devices and dtypes.
    """
    _device, _dtype = _extract_device_dtype([angle, direction])
    _joint_range_check(direction, 'direction', (-1, 1))
    if isinstance(kernel_size, int):
        if not (kernel_size >= 3 and kernel_size % 2 == 1):
            raise AssertionError(f"`kernel_size` must be odd and greater than 3. Got {kernel_size}.")
        ksize_factor = torch.tensor([kernel_size] * batch_size, device=device, dtype=dtype).int()
    elif isinstance(kernel_size, tuple):
        if not (len(kernel_size) == 2 and kernel_size[0] >= 3 and kernel_size[0] <= kernel_size[1]):
            raise AssertionError(f"`kernel_size` must be greater than 3. Got range {kernel_size}.")
        # kernel_size is fixed across the batch
        ksize_factor = (
            _adapted_uniform((batch_size,), kernel_size[0] // 2, kernel_size[1] // 2, same_on_batch=True).int() * 2 + 1
        )
    else:
        raise TypeError(f"Unsupported type: {type(kernel_size)}")

    if angle.shape != torch.Size([3, 2]):
        raise AssertionError(f"'angle' must be the shape of (3, 2). Got {angle.shape}.")
    angle = angle.to(device=device, dtype=dtype)
    yaw = _adapted_uniform((batch_size,), angle[0][0], angle[0][1], same_on_batch)
    pitch = _adapted_uniform((batch_size,), angle[1][0], angle[1][1], same_on_batch)
    roll = _adapted_uniform((batch_size,), angle[2][0], angle[2][1], same_on_batch)
    angle_factor = torch.stack([yaw, pitch, roll], dim=1)

    direction = direction.to(device=device, dtype=dtype)
    direction_factor = _adapted_uniform((batch_size,), direction[0], direction[1], same_on_batch)

    return dict(
        ksize_factor=ksize_factor.to(device=_device),
        angle_factor=angle_factor.to(device=_device, dtype=_dtype),
        direction_factor=direction_factor.to(device=_device, dtype=_dtype),
    )
