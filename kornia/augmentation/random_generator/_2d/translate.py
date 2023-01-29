from typing import Dict, Tuple, Union

import torch
from torch.distributions import Uniform

from kornia.augmentation.random_generator.base import RandomGeneratorBase
from kornia.augmentation.utils import _adapted_rsampling, _common_param_check, _range_bound
from kornia.core import Tensor, stack
from kornia.utils.helpers import _extract_device_dtype

__all__ = [
    "TranslateGenerator"
]


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

    def __init__(self, translate: Union[Tensor, Tuple[float, float]]) -> None:
        super().__init__()
        self.translate = translate

    def __repr__(self) -> str:
        repr = f"translate={self.translate}"
        return repr

    def make_samplers(self, device: torch.device, dtype: torch.dtype) -> None:
        _translate = _range_bound(
            self.translate, 'translate', bounds=(0, 1), check='singular').to(device=device, dtype=dtype)

        self.translate_x = torch.stack([-_translate[0], _translate[0]], dim=-1)
        self.translate_y = torch.stack([-_translate[1], _translate[1]], dim=-1)

        translate_x_sampler = Uniform(- self.translate_x[..., 0], self.translate_x[..., 1], validate_args=False)
        translate_y_sampler = Uniform(- self.translate_y[..., 0], self.translate_y[..., 1], validate_args=False)

        self.translate_x_sampler = translate_x_sampler
        self.translate_y_sampler = translate_y_sampler

    def forward(self, batch_shape: torch.Size, same_on_batch: bool = False) -> Dict[str, Tensor]:
        batch_size = batch_shape[0]
        height = batch_shape[-2]
        width = batch_shape[-1]

        _device, _dtype = _extract_device_dtype([self.translate])
        _common_param_check(batch_size, same_on_batch)
        if not (isinstance(width, (int,)) and isinstance(height, (int,)) and width > 0 and height > 0):
            raise AssertionError(f"`width` and `height` must be positive integers. Got {width}, {height}.")

        translate_x = _adapted_rsampling(
            (batch_size,), self.translate_x_sampler, same_on_batch).to(device=_device, dtype=_dtype) * width
        translate_y = _adapted_rsampling(
            (batch_size,), self.translate_y_sampler, same_on_batch).to(device=_device, dtype=_dtype) * height

        return dict(translate_x=translate_x, translate_y=translate_y)
