from functools import partial
from typing import Dict, List, Tuple, Union

import torch

from kornia.augmentation.random_generator.base import RandomGeneratorBase, UniformDistribution
from kornia.augmentation.utils import _adapted_rsampling, _common_param_check, _joint_range_check, _range_bound
from kornia.core import Tensor
from kornia.utils.helpers import _extract_device_dtype

__all__ = ["ColorJiggleGenerator"]


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
        return f"brightness={self.brightness}, contrast={self.contrast}, saturation={self.saturation}, hue={self.hue}"

    def make_samplers(self, device: torch.device, dtype: torch.dtype) -> None:
        brightness = _range_bound(self.brightness, "brightness", center=1.0, bounds=(0, 2), device=device, dtype=dtype)
        contrast: Tensor = _range_bound(self.contrast, "contrast", center=1.0, device=device, dtype=dtype)
        saturation: Tensor = _range_bound(self.saturation, "saturation", center=1.0, device=device, dtype=dtype)
        hue: Tensor = _range_bound(self.hue, "hue", bounds=(-0.5, 0.5), device=device, dtype=dtype)

        _joint_range_check(brightness, "brightness", (0, 2))
        _joint_range_check(contrast, "contrast", (0, float("inf")))
        _joint_range_check(hue, "hue", (-0.5, 0.5))
        _joint_range_check(saturation, "saturation", (0, float("inf")))

        self.brightness_sampler = UniformDistribution(brightness[0], brightness[1], validate_args=False)
        self.contrast_sampler = UniformDistribution(contrast[0], contrast[1], validate_args=False)
        self.hue_sampler = UniformDistribution(hue[0], hue[1], validate_args=False)
        self.saturation_sampler = UniformDistribution(saturation[0], saturation[1], validate_args=False)
        self.randperm = partial(torch.randperm, device=device, dtype=dtype)

    def forward(self, batch_shape: Tuple[int, ...], same_on_batch: bool = False) -> Dict[str, Tensor]:
        batch_size = batch_shape[0]
        _common_param_check(batch_size, same_on_batch)
        _device, _dtype = _extract_device_dtype([self.brightness, self.contrast, self.hue, self.saturation])
        brightness_factor = _adapted_rsampling((batch_size,), self.brightness_sampler, same_on_batch)
        contrast_factor = _adapted_rsampling((batch_size,), self.contrast_sampler, same_on_batch)
        hue_factor = _adapted_rsampling((batch_size,), self.hue_sampler, same_on_batch)
        saturation_factor = _adapted_rsampling((batch_size,), self.saturation_sampler, same_on_batch)
        return {
            "brightness_factor": brightness_factor.to(device=_device, dtype=_dtype),
            "contrast_factor": contrast_factor.to(device=_device, dtype=_dtype),
            "hue_factor": hue_factor.to(device=_device, dtype=_dtype),
            "saturation_factor": saturation_factor.to(device=_device, dtype=_dtype),
            "order": self.randperm(4).to(device=_device, dtype=_dtype).long(),
        }
