from functools import partial
from typing import Dict, List, Tuple, Union

import torch
from torch.distributions import Uniform

from kornia.augmentation.random_generator.base import RandomGeneratorBase
from kornia.augmentation.utils import _adapted_rsampling, _common_param_check, _joint_range_check, _range_bound
from kornia.core import Tensor
from kornia.utils.helpers import _extract_device_dtype


class ColorJiggleGenerator(RandomGeneratorBase):
    r"""Generate random color jiter parameters for a batch of images following OpenCV.

    Args:
        brightness: The brightness factor to apply.
        contrast: The contrast factor to apply.
        saturation: The saturation factor to apply.
        hue: The hue factor to apply.

    Returns:
        A dict of parameters to be passed for transformation.
            - brightness_factor: element-wise brightness factors with a shape of (B,).
            - contrast_factor: element-wise contrast factors with a shape of (B,).
            - hue_factor: element-wise hue factors with a shape of (B,).
            - saturation_factor: element-wise saturation factors with a shape of (B,).
            - order: applying orders of the color adjustments with a shape of (4). In which,
                0 is brightness adjustment; 1 is contrast adjustment;
                2 is saturation adjustment; 3 is hue adjustment.

    Note:
        The generated random numbers are not reproducible across different devices and dtypes. By default,
        the parameters will be generated on CPU in float32. This can be changed by calling
        ``self.set_rng_device_and_dtype(device="cuda", dtype=torch.float64)``.
    """

    def __init__(
        self,
        brightness: Union[Tensor, float, Tuple[float, float], List[float]] = 0.0,
        contrast: Union[Tensor, float, Tuple[float, float], List[float]] = 0.0,
        saturation: Union[Tensor, float, Tuple[float, float], List[float]] = 0.0,
        hue: Union[Tensor, float, Tuple[float, float], List[float]] = 0.0,
    ) -> None:
        super().__init__()
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    def __repr__(self) -> str:
        repr = (
            f"brightness={self.brightness}, contrast={self.contrast}, saturation=" f"{self.saturation}, hue={self.hue}"
        )
        return repr

    def make_samplers(self, device: torch.device, dtype: torch.dtype) -> None:
        self._brightness = _range_bound(self.brightness, 'brightness', center=1.0, bounds=(0, 2), device=device, dtype=dtype)
        self._contrast: Tensor = _range_bound(self.contrast, 'contrast', center=1.0, device=device, dtype=dtype)
        self._saturation: Tensor = _range_bound(self.saturation, 'saturation', center=1.0, device=device, dtype=dtype)
        self._hue: Tensor = _range_bound(self.hue, 'hue', bounds=(-0.5, 0.5), device=device, dtype=dtype)

        _joint_range_check(self._brightness, "brightness", (0, 2))
        _joint_range_check(self._contrast, "contrast", (0, float('inf')))
        _joint_range_check(self._hue, "hue", (-0.5, 0.5))
        _joint_range_check(self._saturation, "saturation", (0, float('inf')))

        self.randperm = partial(torch.randperm, device=device, dtype=dtype)
        self.generic_sampler = Uniform(0, 1)

    def forward(self, batch_shape: torch.Size, same_on_batch: bool = False) -> Dict[str, Tensor]:
        batch_size = batch_shape[0]
        _common_param_check(batch_size, same_on_batch)
        _device, _dtype = _extract_device_dtype([self.brightness, self.contrast, self.hue, self.saturation])

        generic_factors = _adapted_rsampling((batch_size * 4,), self.generic_sampler, same_on_batch).to(device=_device, dtype=_dtype)

        brightness_factor = (generic_factors[:batch_size] - self._brightness[0]) / (self._brightness[1] - self._brightness[0])
        
        return dict(
            brightness_factor=brightness_factor,
            contrast_factor=...,
            hue_factor=...,
            saturation_factor=...,
            order=self.randperm(4).to(device=_device, dtype=_dtype).long(),
        )
