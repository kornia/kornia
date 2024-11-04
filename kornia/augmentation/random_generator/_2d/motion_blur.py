from typing import Dict, Tuple, Union

import torch

from kornia.augmentation.random_generator.base import RandomGeneratorBase, UniformDistribution
from kornia.augmentation.utils import _adapted_rsampling, _common_param_check, _range_bound
from kornia.core import Tensor
from kornia.utils.helpers import _extract_device_dtype

__all__ = ["MotionBlurGenerator"]


class MotionBlurGenerator(RandomGeneratorBase):
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
            - ksize_factor (Tensor): element-wise kernel size factors with a shape of (B,).
            - angle_factor (Tensor): element-wise angle factors with a shape of (B,).
            - direction_factor (Tensor): element-wise direction factors with a shape of (B,).

    Note:
        The generated random numbers are not reproducible across different devices and dtypes. By default,
        the parameters will be generated on CPU in float32. This can be changed by calling
        ``self.set_rng_device_and_dtype(device="cuda", dtype=torch.float64)``.
    """

    def __init__(
        self,
        kernel_size: Union[int, Tuple[int, int]],
        angle: Union[Tensor, float, Tuple[float, float]],
        direction: Union[Tensor, float, Tuple[float, float]],
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.angle = angle
        self.direction = direction

    def __repr__(self) -> str:
        repr = f"kernel_size={self.kernel_size}, angle={self.angle}, direction={self.direction}"
        return repr

    def make_samplers(self, device: torch.device, dtype: torch.dtype) -> None:
        angle = _range_bound(self.angle, "angle", center=0.0, bounds=(-360, 360)).to(device=device, dtype=dtype)
        direction = _range_bound(self.direction, "direction", center=0.0, bounds=(-1, 1)).to(device=device, dtype=dtype)
        if isinstance(self.kernel_size, int):
            if not (self.kernel_size >= 3 and self.kernel_size % 2 == 1):
                raise AssertionError(f"`kernel_size` must be odd and greater than 3. Got {self.kernel_size}.")
            self.ksize_sampler = UniformDistribution(self.kernel_size // 2, self.kernel_size // 2, validate_args=False)
        elif isinstance(self.kernel_size, tuple):
            # kernel_size is fixed across the batch
            if len(self.kernel_size) != 2:
                raise AssertionError(f"`kernel_size` must be (2,) if it is a tuple. Got {self.kernel_size}.")
            self.ksize_sampler = UniformDistribution(
                self.kernel_size[0] // 2, self.kernel_size[1] // 2, validate_args=False
            )
        else:
            raise TypeError(f"Unsupported type: {type(self.kernel_size)}")

        self.angle_sampler = UniformDistribution(angle[0], angle[1], validate_args=False)
        self.direction_sampler = UniformDistribution(direction[0], direction[1], validate_args=False)

    def forward(self, batch_shape: Tuple[int, ...], same_on_batch: bool = False) -> Dict[str, Tensor]:
        batch_size = batch_shape[0]
        _common_param_check(batch_size, same_on_batch)
        # self.ksize_factor.expand((batch_size, -1))
        _device, _dtype = _extract_device_dtype([self.angle, self.direction])
        angle_factor = _adapted_rsampling((batch_size,), self.angle_sampler, same_on_batch)
        direction_factor = _adapted_rsampling((batch_size,), self.direction_sampler, same_on_batch)
        ksize_factor = _adapted_rsampling((batch_size,), self.ksize_sampler, same_on_batch).int() * 2 + 1

        return {
            "ksize_factor": ksize_factor.to(device=_device, dtype=torch.int32),
            "angle_factor": angle_factor.to(device=_device, dtype=_dtype),
            "direction_factor": direction_factor.to(device=_device, dtype=_dtype),
        }
