from typing import Dict, Tuple, Union

import torch

from kornia.augmentation.random_generator.base import RandomGeneratorBase, UniformDistribution
from kornia.augmentation.utils import _adapted_rsampling, _common_param_check, _joint_range_check, _range_bound
from kornia.core import Tensor, as_tensor, stack, tensor
from kornia.utils.helpers import _extract_device_dtype

__all__ = ["ShearGenerator"]


class ShearGenerator(RandomGeneratorBase):
    r"""Get parameters for ``shear`` for a random shear transform.

    Args:
        shear: Range of degrees to select from.
            If float, a shear parallel to the x axis in the range (-shear, +shear) will be applied.
            If (a, b), a shear parallel to the x axis in the range (-shear, +shear) will be applied.
            If (a, b, c, d), then x-axis shear in (shear[0], shear[1]) and y-axis shear in (shear[2], shear[3])
            will be applied. Will not apply shear by default.
            If tensor, shear is a 2x2 tensor, a x-axis shear in (shear[0][0], shear[0][1]) and y-axis shear in
            (shear[1][0], shear[1][1]) will be applied. Will not apply shear by default.

    Returns:
        A dict of parameters to be passed for transformation.
            - shear_x (Tensor): element-wise x-axis shears with a shape of (B,).
            - shear_y (Tensor): element-wise y-axis shears with a shape of (B,).

    Note:
        The generated random numbers are not reproducible across different devices and dtypes. By default,
        the parameters will be generated on CPU in float32. This can be changed by calling
        ``self.set_rng_device_and_dtype(device="cuda", dtype=torch.float64)``.
    """

    def __init__(self, shear: Union[Tensor, float, Tuple[float, float], Tuple[float, float, float, float]]) -> None:
        super().__init__()
        self.shear = shear

    def __repr__(self) -> str:
        repr = f"shear={self.shear}"
        return repr

    def make_samplers(self, device: torch.device, dtype: torch.dtype) -> None:
        shear = as_tensor(self.shear, device=device, dtype=dtype)
        if shear.shape == torch.Size([2, 2]):
            _shear = shear
        else:
            _shear = stack(
                [
                    _range_bound(shear if shear.dim() == 0 else shear[:2], "shear-x", 0, (-360, 360)),
                    (
                        tensor([0, 0], device=device, dtype=dtype)
                        if shear.dim() == 0 or len(shear) == 2
                        else _range_bound(shear[2:], "shear-y", 0, (-360, 360))
                    ),
                ]
            )

        _joint_range_check(_shear[0], "shear")
        _joint_range_check(_shear[1], "shear")
        self.shear_x = _shear[0].clone()
        self.shear_y = _shear[1].clone()

        shear_x_sampler = UniformDistribution(_shear[0][0], _shear[0][1], validate_args=False)
        shear_y_sampler = UniformDistribution(_shear[1][0], _shear[1][1], validate_args=False)

        self.shear_x_sampler = shear_x_sampler
        self.shear_y_sampler = shear_y_sampler

    def forward(self, batch_shape: Tuple[int, ...], same_on_batch: bool = False) -> Dict[str, Tensor]:
        batch_size = batch_shape[0]
        height = batch_shape[-2]
        width = batch_shape[-1]

        _device, _dtype = _extract_device_dtype([self.shear])
        _common_param_check(batch_size, same_on_batch)
        if not (isinstance(width, (int,)) and isinstance(height, (int,)) and width > 0 and height > 0):
            raise AssertionError(f"`width` and `height` must be positive integers. Got {width}, {height}.")

        center: Tensor = tensor([width, height], device=_device, dtype=_dtype).view(1, 2) / 2.0 - 0.5
        center = center.expand(batch_size, -1)

        sx = _adapted_rsampling((batch_size,), self.shear_x_sampler, same_on_batch)
        sy = _adapted_rsampling((batch_size,), self.shear_y_sampler, same_on_batch)
        sx = sx.to(device=_device, dtype=_dtype)
        sy = sy.to(device=_device, dtype=_dtype)

        return {"center": center, "shear_x": sx, "shear_y": sy}
