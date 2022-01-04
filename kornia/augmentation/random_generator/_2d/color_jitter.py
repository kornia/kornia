from functools import partial
from typing import Dict, List, Optional, Tuple, Union

import torch
from torch.distributions import Uniform

from kornia.augmentation.random_generator.base import RandomGeneratorBase
from kornia.augmentation.utils import (
    _adapted_rsampling,
    _adapted_uniform,
    _common_param_check,
    _joint_range_check,
    _range_bound,
)
from kornia.utils.helpers import _deprecated, _extract_device_dtype


class ColorJitterGenerator(RandomGeneratorBase):
    r"""Generate random color jiter parameters for a batch of images.

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
        brightness: Union[torch.Tensor, float, Tuple[float, float], List[float]] = 0.0,
        contrast: Union[torch.Tensor, float, Tuple[float, float], List[float]] = 0.0,
        saturation: Union[torch.Tensor, float, Tuple[float, float], List[float]] = 0.0,
        hue: Union[torch.Tensor, float, Tuple[float, float], List[float]] = 0.0,
    ) -> None:
        super().__init__()
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    def __repr__(self) -> str:
        repr = f"brightness={self.brightness}, contrast={self.contrast}, saturation={self.saturation}, hue={self.hue}"
        return repr

    def make_samplers(self, device: torch.device, dtype: torch.dtype) -> None:
        brightness: torch.Tensor = _range_bound(
            self.brightness, 'brightness', center=1.0, bounds=(0, 2), device=device, dtype=dtype
        )
        contrast: torch.Tensor = _range_bound(self.contrast, 'contrast', center=1.0, device=device, dtype=dtype)
        saturation: torch.Tensor = _range_bound(self.saturation, 'saturation', center=1.0, device=device, dtype=dtype)
        hue: torch.Tensor = _range_bound(self.hue, 'hue', bounds=(-0.5, 0.5), device=device, dtype=dtype)

        _joint_range_check(brightness, "brightness", (0, 2))
        _joint_range_check(contrast, "contrast", (0, float('inf')))
        _joint_range_check(hue, "hue", (-0.5, 0.5))
        _joint_range_check(saturation, "saturation", (0, float('inf')))

        self.brightness_sampler = Uniform(brightness[0], brightness[1], validate_args=False)
        self.contrast_sampler = Uniform(contrast[0], contrast[1], validate_args=False)
        self.hue_sampler = Uniform(hue[0], hue[1], validate_args=False)
        self.saturation_sampler = Uniform(saturation[0], saturation[1], validate_args=False)
        self.randperm = partial(torch.randperm, device=device, dtype=dtype)

    def forward(self, batch_shape: torch.Size, same_on_batch: bool = False) -> Dict[str, torch.Tensor]:  # type:ignore
        batch_size = batch_shape[0]
        _common_param_check(batch_size, same_on_batch)
        _device, _dtype = _extract_device_dtype([self.brightness, self.contrast, self.hue, self.saturation])
        brightness_factor = _adapted_rsampling((batch_size,), self.brightness_sampler, same_on_batch)
        contrast_factor = _adapted_rsampling((batch_size,), self.contrast_sampler, same_on_batch)
        hue_factor = _adapted_rsampling((batch_size,), self.hue_sampler, same_on_batch)
        saturation_factor = _adapted_rsampling((batch_size,), self.saturation_sampler, same_on_batch)
        return dict(
            brightness_factor=brightness_factor.to(device=_device, dtype=_dtype),
            contrast_factor=contrast_factor.to(device=_device, dtype=_dtype),
            hue_factor=hue_factor.to(device=_device, dtype=_dtype),
            saturation_factor=saturation_factor.to(device=_device, dtype=_dtype),
            order=self.randperm(4).to(device=_device, dtype=_dtype).long(),
        )


@_deprecated(replace_with=ColorJitterGenerator.__name__)
def random_color_jitter_generator(
    batch_size: int,
    brightness: Optional[torch.Tensor] = None,
    contrast: Optional[torch.Tensor] = None,
    saturation: Optional[torch.Tensor] = None,
    hue: Optional[torch.Tensor] = None,
    same_on_batch: bool = False,
    device: torch.device = torch.device('cpu'),
    dtype: torch.dtype = torch.float32,
) -> Dict[str, torch.Tensor]:
    r"""Generate random color jiter parameters for a batch of images.

    Args:
        batch_size (int): the number of images.
        brightness (torch.Tensor, optional): Brightness factor tensor of range (a, b).
            The provided range must follow 0 <= a <= b <= 2. Default value is [0., 0.].
        contrast (torch.Tensor, optional): Contrast factor tensor of range (a, b).
            The provided range must follow 0 <= a <= b. Default value is [0., 0.].
        saturation (torch.Tensor, optional): Saturation factor tensor of range (a, b).
            The provided range must follow 0 <= a <= b. Default value is [0., 0.].
        hue (torch.Tensor, optional): Saturation factor tensor of range (a, b).
            The provided range must follow -0.5 <= a <= b < 0.5. Default value is [0., 0.].
        same_on_batch (bool): apply the same transformation across the batch. Default: False.
        device (torch.device): the device on which the random numbers will be generated. Default: cpu.
        dtype (torch.dtype): the data type of the generated random numbers. Default: float32.

    Returns:
        params Dict[str, torch.Tensor]: parameters to be passed for transformation.
            - brightness_factor (torch.Tensor): element-wise brightness factors with a shape of (B,).
            - contrast_factor (torch.Tensor): element-wise contrast factors with a shape of (B,).
            - hue_factor (torch.Tensor): element-wise hue factors with a shape of (B,).
            - saturation_factor (torch.Tensor): element-wise saturation factors with a shape of (B,).
            - order (torch.Tensor): applying orders of the color adjustments with a shape of (4). In which,
                0 is brightness adjustment; 1 is contrast adjustment;
                2 is saturation adjustment; 3 is hue adjustment.

    Note:
        The generated random numbers are not reproducible across different devices and dtypes.
    """
    _common_param_check(batch_size, same_on_batch)
    _device, _dtype = _extract_device_dtype([brightness, contrast, hue, saturation])
    brightness = torch.as_tensor([0.0, 0.0] if brightness is None else brightness, device=device, dtype=dtype)
    contrast = torch.as_tensor([0.0, 0.0] if contrast is None else contrast, device=device, dtype=dtype)
    hue = torch.as_tensor([0.0, 0.0] if hue is None else hue, device=device, dtype=dtype)
    saturation = torch.as_tensor([0.0, 0.0] if saturation is None else saturation, device=device, dtype=dtype)

    _joint_range_check(brightness, "brightness", (0, 2))
    _joint_range_check(contrast, "contrast", (0, float('inf')))
    _joint_range_check(hue, "hue", (-0.5, 0.5))
    _joint_range_check(saturation, "saturation", (0, float('inf')))

    brightness_factor = _adapted_uniform((batch_size,), brightness[0], brightness[1], same_on_batch)
    contrast_factor = _adapted_uniform((batch_size,), contrast[0], contrast[1], same_on_batch)
    hue_factor = _adapted_uniform((batch_size,), hue[0], hue[1], same_on_batch)
    saturation_factor = _adapted_uniform((batch_size,), saturation[0], saturation[1], same_on_batch)

    return dict(
        brightness_factor=brightness_factor.to(device=_device, dtype=_dtype),
        contrast_factor=contrast_factor.to(device=_device, dtype=_dtype),
        hue_factor=hue_factor.to(device=_device, dtype=_dtype),
        saturation_factor=saturation_factor.to(device=_device, dtype=_dtype),
        order=torch.randperm(4, device=_device, dtype=_dtype).long(),
    )
