from typing import Dict, Optional, Tuple, Union

import torch

from kornia.augmentation.random_generator.base import RandomGeneratorBase, UniformDistribution
from kornia.augmentation.utils import _adapted_rsampling, _common_param_check, _range_bound
from kornia.core import Tensor
from kornia.utils.helpers import _extract_device_dtype

__all__ = ["TranslateGenerator"]


class TranslateGenerator(RandomGeneratorBase):
    r"""Get parameters for ``translate`` for a random translate transform.

    Args:
        translate: tuple of maximum absolute fraction for horizontal
            and vertical translations. For example translate=(a, b), then horizontal shift
            is randomly sampled in the range -img_width * a < dx < img_width * a and vertical shift is
            randomly sampled in the range -img_height * b < dy < img_height * b. Will not translate by default.

    Returns:
        A dict of parameters to be passed for transformation.
            - translations (Tensor): element-wise translations with a shape of (B, 2).

    Note:
        The generated random numbers are not reproducible across different devices and dtypes. By default,
        the parameters will be generated on CPU in float32. This can be changed by calling
        ``self.set_rng_device_and_dtype(device="cuda", dtype=torch.float64)``.
    """

    def __init__(
        self,
        translate_x: Optional[Union[Tensor, Tuple[float, float]]] = None,
        translate_y: Optional[Union[Tensor, Tuple[float, float]]] = None,
    ) -> None:
        super().__init__()
        self.translate_x = translate_x
        self.translate_y = translate_y

    def __repr__(self) -> str:
        repr = f"translate_x={self.translate_x}, translate_y={self.translate_y}"
        return repr

    def make_samplers(self, device: torch.device, dtype: torch.dtype) -> None:
        self.translate_x_sampler = None
        self.translate_y_sampler = None

        if self.translate_x is not None:
            _translate_x = _range_bound(self.translate_x, "translate_x", bounds=(-1, 1), check="joint").to(
                device=device, dtype=dtype
            )

            self.translate_x_sampler = UniformDistribution(
                _translate_x[..., 0], _translate_x[..., 1], validate_args=False
            )

        if self.translate_y is not None:
            _translate_y = _range_bound(self.translate_y, "translate_y", bounds=(-1, 1), check="joint").to(
                device=device, dtype=dtype
            )

            self.translate_y_sampler = UniformDistribution(
                _translate_y[..., 0], _translate_y[..., 1], validate_args=False
            )

    def forward(self, batch_shape: Tuple[int, ...], same_on_batch: bool = False) -> Dict[str, Tensor]:
        batch_size = batch_shape[0]
        height = batch_shape[-2]
        width = batch_shape[-1]

        _device, _dtype = _extract_device_dtype([self.translate_x, self.translate_y])
        _common_param_check(batch_size, same_on_batch)
        if not (isinstance(width, (int,)) and isinstance(height, (int,)) and width > 0 and height > 0):
            raise AssertionError(f"`width` and `height` must be positive integers. Got {width}, {height}.")

        if self.translate_x_sampler is not None:
            translate_x = (
                _adapted_rsampling((batch_size,), self.translate_x_sampler, same_on_batch).to(
                    device=_device, dtype=_dtype
                )
                * width
            )
        else:
            translate_x = torch.zeros((batch_size,), device=_device, dtype=_dtype)

        if self.translate_y_sampler is not None:
            translate_y = (
                _adapted_rsampling((batch_size,), self.translate_y_sampler, same_on_batch).to(
                    device=_device, dtype=_dtype
                )
                * height
            )
        else:
            translate_y = torch.zeros((batch_size,), device=_device, dtype=_dtype)

        return {"translate_x": translate_x, "translate_y": translate_y}
