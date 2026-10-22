# LICENSE HEADER MANAGED BY add-license-header
#
# Copyright 2018 Kornia Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from typing import Any, Dict, Optional, Tuple

import torch

import kornia.augmentation as K
from kornia.augmentation._2d.intensity.base import _clamp_unit_range
from kornia.augmentation._2d.intensity.brightness import RandomBrightness
from kornia.augmentation._2d.intensity.contrast import RandomContrast
from kornia.augmentation._2d.intensity.saturation import RandomSaturation
from kornia.augmentation.auto.operations.base import OperationBase
from kornia.core.utils import _to_device_nonblocking
from kornia.enhance import (
    adjust_brightness_accumulative,
    adjust_contrast_with_mean_subtraction,
    adjust_saturation_with_gray_subtraction,
)

__all__ = [
    "AutoContrast",
    "Brightness",
    "Contrast",
    "Equalize",
    "Gray",
    "HorizontalFlip",
    "Hue",
    "Invert",
    "Posterize",
    "Rotate",
    "Saturate",
    "Sharpness",
    "ShearX",
    "ShearY",
    "Solarize",
    "SolarizeAdd",
    "TranslateX",
    "TranslateY",
    "VerticalFlip",
]


class _PILBrightness(RandomBrightness):
    def apply_transform(
        self,
        input: torch.Tensor,
        params: Dict[str, torch.Tensor],
        flags: Dict[str, Any],
        transform: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        factor = _to_device_nonblocking(params["brightness_factor"], input.device, input.dtype)
        return adjust_brightness_accumulative(_clamp_unit_range(input), factor)


class _PILContrast(RandomContrast):
    def apply_transform(
        self,
        input: torch.Tensor,
        params: Dict[str, torch.Tensor],
        flags: Dict[str, Any],
        transform: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        factor = _to_device_nonblocking(params["contrast_factor"], input.device, input.dtype)
        return adjust_contrast_with_mean_subtraction(_clamp_unit_range(input), factor)


class _PILSaturation(RandomSaturation):
    def apply_transform(
        self,
        input: torch.Tensor,
        params: Dict[str, torch.Tensor],
        flags: Dict[str, Any],
        transform: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        factor = _to_device_nonblocking(params["saturation_factor"], input.device, input.dtype)
        return adjust_saturation_with_gray_subtraction(_clamp_unit_range(input), factor)


class AutoContrast(OperationBase):
    """Apply auto_contrast operation.

    Args:
        initial_probability: the initial probability. If None, the augmentation will be randomly
            applied according to he augmentation sampling range.
        temperature: retained for API compatibility.
        symmetric_megnitude: if to randomly assign the magnitude as negative or not.

    """

    def __init__(
        self, initial_probability: float = 0.5, temperature: float = 0.1, symmetric_megnitude: bool = False
    ) -> None:
        super().__init__(
            K.RandomAutoContrast(same_on_batch=False, p=initial_probability),
            initial_magnitude=None,
            temperature=temperature,
            symmetric_megnitude=symmetric_megnitude,
        )


class Brightness(OperationBase):
    """Apply PIL-style multiplicative brightness for automatic augmentation.

    Unlike :class:`~kornia.augmentation.RandomBrightness`, the factor multiplies
    the input. The policy factors match the PIL ImageEnhance convention.


    Args:
        initial_probability: the initial probability. If None, the augmentation will be randomly
            applied according to he augmentation sampling range.
        initial_magnitude: the initial magnitude.
        magnitude_range: the sampling range for random sampling and clamping the optimized magnitude.
        temperature: retained for API compatibility.
        symmetric_megnitude: if to randomly assign the magnitude as negative or not.

    """

    def __init__(
        self,
        initial_magnitude: Optional[float] = 0.5,
        initial_probability: float = 0.5,
        magnitude_range: Tuple[float, float] = (0.2, 1.8),
        temperature: float = 0.1,
        symmetric_megnitude: bool = False,
    ) -> None:
        super().__init__(
            _PILBrightness(magnitude_range, same_on_batch=False, p=initial_probability),
            initial_magnitude=[("brightness_factor", initial_magnitude)],
            temperature=temperature,
            symmetric_megnitude=symmetric_megnitude,
        )


class Contrast(OperationBase):
    """Apply PIL-style contrast by blending towards the mean grayscale value.

    Unlike :class:`~kornia.augmentation.RandomContrast`, the factor scales the
    difference from the mean rather than multiplying the input directly.


    Args:
        initial_probability: the initial probability. If None, the augmentation will be randomly
            applied according to he augmentation sampling range.
        initial_magnitude: the initial magnitude.
        magnitude_range: the sampling range for random sampling and clamping the optimized magnitude.
        temperature: retained for API compatibility.
        symmetric_megnitude: if to randomly assign the magnitude as negative or not.

    """

    def __init__(
        self,
        initial_magnitude: Optional[float] = 0.5,
        initial_probability: float = 0.5,
        magnitude_range: Tuple[float, float] = (0.2, 1.8),
        temperature: float = 0.1,
        symmetric_megnitude: bool = False,
    ) -> None:
        super().__init__(
            _PILContrast(magnitude_range, same_on_batch=False, p=initial_probability),
            initial_magnitude=[("contrast_factor", initial_magnitude)],
            temperature=temperature,
            symmetric_megnitude=symmetric_megnitude,
        )


class Hue(OperationBase):
    """Apply hue operation.

    Args:
        initial_probability: the initial probability. If None, the augmentation will be randomly
            applied according to he augmentation sampling range.
        initial_magnitude: the initial magnitude.
        magnitude_range: the sampling range for random sampling and clamping the optimized magnitude.
        temperature: retained for API compatibility.
        symmetric_megnitude: if to randomly assign the magnitude as negative or not.

    """

    def __init__(
        self,
        initial_magnitude: Optional[float] = 0.0,
        initial_probability: float = 0.5,
        magnitude_range: Tuple[float, float] = (-0.5, 0.5),
        temperature: float = 0.1,
        symmetric_megnitude: bool = False,
    ) -> None:
        super().__init__(
            K.RandomHue(magnitude_range, same_on_batch=False, p=initial_probability),
            initial_magnitude=[("hue_factor", initial_magnitude)],
            temperature=temperature,
            symmetric_megnitude=symmetric_megnitude,
        )


class Saturate(OperationBase):
    """Apply PIL-style color enhancement by blending with grayscale.

    Unlike :class:`~kornia.augmentation.RandomSaturation`, this blends RGB
    values with grayscale rather than scaling HSV saturation.


    Args:
        initial_probability: the initial probability. If None, the augmentation will be randomly
            applied according to he augmentation sampling range.
        initial_magnitude: the initial magnitude.
        magnitude_range: the sampling range for random sampling and clamping the optimized magnitude.
        temperature: retained for API compatibility.
        symmetric_megnitude: if to randomly assign the magnitude as negative or not.

    """

    def __init__(
        self,
        initial_magnitude: Optional[float] = 0.5,
        initial_probability: float = 0.5,
        magnitude_range: Tuple[float, float] = (0.2, 1.8),
        temperature: float = 0.1,
        symmetric_megnitude: bool = False,
    ) -> None:
        super().__init__(
            _PILSaturation(magnitude_range, same_on_batch=False, p=initial_probability),
            initial_magnitude=[("saturation_factor", initial_magnitude)],
            temperature=temperature,
            symmetric_megnitude=symmetric_megnitude,
        )


# TODO: Equalize cannot update probabilities yet.
class Equalize(OperationBase):
    """Apply equalize operation.

    Args:
        initial_probability: the initial probability. If None, the augmentation will be randomly
            applied according to he augmentation sampling range.
        temperature: retained for API compatibility.

    Note:
        Equalize cannot update probabilities yet.

    """

    def __init__(self, initial_probability: float = 0.5, temperature: float = 0.1) -> None:
        super().__init__(
            K.RandomEqualize(same_on_batch=False, p=initial_probability),
            initial_magnitude=None,
            temperature=temperature,
            symmetric_megnitude=False,
        )


class Gray(OperationBase):
    """Apply grayscale operation.

    Args:
        initial_probability: the initial probability. If None, the augmentation will be randomly
            applied according to he augmentation sampling range.
        temperature: retained for API compatibility.

    """

    def __init__(self, initial_probability: float = 0.5, temperature: float = 0.1) -> None:
        super().__init__(
            K.RandomGrayscale(same_on_batch=False, p=initial_probability),
            initial_magnitude=None,
            temperature=temperature,
            symmetric_megnitude=False,
        )


class Invert(OperationBase):
    """Apply invert operation.

    Args:
        initial_probability: the initial probability. If None, the augmentation will be randomly
            applied according to he augmentation sampling range.
        temperature: retained for API compatibility.

    """

    def __init__(self, initial_probability: float = 0.5, temperature: float = 0.1) -> None:
        super().__init__(
            K.RandomInvert(same_on_batch=False, p=initial_probability),
            initial_magnitude=None,
            temperature=temperature,
            symmetric_megnitude=False,
        )


class Posterize(OperationBase):
    """Apply posterize operation.

    Args:
        initial_magnitude: the initial magnitude.
        initial_probability: the initial probability. If None, the augmentation will be randomly
            applied according to he augmentation sampling range.
        magnitude_range: the sampling range for random sampling and clamping the optimized magnitude.
        temperature: retained for API compatibility.
        symmetric_megnitude: if to randomly assign the magnitude as negative or not.

    """

    @staticmethod
    def _process_magnitude(magnitude: torch.Tensor) -> torch.Tensor:
        return magnitude.long()

    def __init__(
        self,
        initial_magnitude: Optional[float] = 4.0,
        initial_probability: float = 0.5,
        magnitude_range: Tuple[float, float] = (1.0, 8.0),
        temperature: float = 0.1,
        symmetric_megnitude: bool = False,
    ) -> None:
        super().__init__(
            K.RandomPosterize(magnitude_range, same_on_batch=False, p=initial_probability),
            initial_magnitude=[("bits_factor", initial_magnitude)],
            temperature=temperature,
            symmetric_megnitude=symmetric_megnitude,
            magnitude_fn=Posterize._process_magnitude,
        )


class Solarize(OperationBase):
    """Apply solarize operation.

    Args:
        initial_magnitude: the initial magnitude.
        initial_probability: the initial probability. If None, the augmentation will be randomly
            applied according to he augmentation sampling range.
        magnitude_range: the sampling range for random sampling and clamping the optimized
        symmetric_megnitude: if to randomly assign the magnitude as negative or not.magnitude.
        temperature: retained for API compatibility.

    """

    def __init__(
        self,
        initial_magnitude: Optional[float] = 0.5,
        initial_probability: float = 0.5,
        magnitude_range: Tuple[float, float] = (0.0, 1.0),
        temperature: float = 0.1,
        symmetric_megnitude: bool = False,
    ) -> None:
        super().__init__(
            K.RandomSolarize(magnitude_range, additions=0.0, same_on_batch=False, p=initial_probability),
            initial_magnitude=[("thresholds", initial_magnitude)],
            temperature=temperature,
            symmetric_megnitude=symmetric_megnitude,
        )


class SolarizeAdd(OperationBase):
    """Apply solarize-addition operation with a fixed thresholds of 0.5.

    Args:
        initial_magnitude: the initial magnitude.
        initial_probability: the initial probability. If None, the augmentation will be randomly
            applied according to he augmentation sampling range.
        magnitude_range: the sampling range for random sampling and clamping the optimized magnitude.
        temperature: retained for API compatibility.
        symmetric_megnitude: if to randomly assign the magnitude as negative or not.

    """

    def __init__(
        self,
        initial_magnitude: Optional[float] = 0.0,
        initial_probability: float = 0.5,
        magnitude_range: Tuple[float, float] = (-0.3, 0.3),
        temperature: float = 0.1,
        symmetric_megnitude: bool = False,
    ) -> None:
        super().__init__(
            K.RandomSolarize(thresholds=0.5, additions=magnitude_range, same_on_batch=False, p=initial_probability),
            initial_magnitude=[("additions", initial_magnitude)],
            temperature=temperature,
            symmetric_megnitude=symmetric_megnitude,
        )


class Sharpness(OperationBase):
    """Apply sharpness operation.

    Args:
        initial_magnitude: the initial magnitude.
        initial_probability: the initial probability. If None, the augmentation will be randomly
            applied according to he augmentation sampling range.
        magnitude_range: the sampling range for random sampling and clamping the optimized magnitude.
        temperature: retained for API compatibility.
        symmetric_megnitude: if to randomly assign the magnitude as negative or not.

    """

    def __init__(
        self,
        initial_magnitude: Optional[float] = 0.5,
        initial_probability: float = 0.5,
        magnitude_range: Tuple[float, float] = (0.1, 1.9),
        temperature: float = 0.1,
        symmetric_megnitude: bool = False,
    ) -> None:
        super().__init__(
            K.RandomSharpness(magnitude_range, same_on_batch=False, p=initial_probability),
            initial_magnitude=[("sharpness", initial_magnitude)],
            temperature=temperature,
            symmetric_megnitude=symmetric_megnitude,
        )


class HorizontalFlip(OperationBase):
    """Apply horizontal flip operation.

    Args:
        initial_probability: the initial probability. If None, the augmentation will be randomly
            applied according to he augmentation sampling range.
        temperature: retained for API compatibility.

    """

    def __init__(self, initial_probability: float = 0.5, temperature: float = 0.1) -> None:
        super().__init__(
            K.RandomHorizontalFlip(same_on_batch=False, p=initial_probability),
            initial_magnitude=None,
            temperature=temperature,
            symmetric_megnitude=False,
        )


class VerticalFlip(OperationBase):
    """Apply vertical flip operation.

    Args:
        initial_magnitude: the initial magnitude.
        temperature: retained for API compatibility.

    """

    def __init__(self, initial_probability: float = 0.5, temperature: float = 0.1) -> None:
        super().__init__(
            K.RandomVerticalFlip(same_on_batch=False, p=initial_probability),
            initial_magnitude=None,
            temperature=temperature,
            symmetric_megnitude=False,
        )


class Rotate(OperationBase):
    """Apply rotate operation.

    Args:
        initial_magnitude: the initial magnitude.
        initial_probability: the initial probability. If None, the augmentation will be randomly
            applied according to he augmentation sampling range.
        magnitude_range: the sampling range for random sampling and clamping the optimized magnitude.
        temperature: retained for API compatibility.
        symmetric_megnitude: if to randomly assign the magnitude as negative or not.

    """

    def __init__(
        self,
        initial_magnitude: Optional[float] = 15.0,
        initial_probability: float = 0.5,
        magnitude_range: Tuple[float, float] = (0.0, 30.0),
        temperature: float = 0.1,
        symmetric_megnitude: bool = True,
    ) -> None:
        if symmetric_megnitude and magnitude_range[0] < 0:
            raise ValueError(
                f"Lower bound of {self.__class__.__name__} is a symmetric operation. "
                f"The lower bound must above 0. Got {magnitude_range[0]}."
            )
        super().__init__(
            K.RandomRotation(magnitude_range, same_on_batch=False, p=initial_probability),
            initial_magnitude=[("degrees", initial_magnitude)],
            temperature=temperature,
            symmetric_megnitude=symmetric_megnitude,
        )


class ShearX(OperationBase):
    """Apply shear operation along x-axis.

    Args:
        initial_magnitude: the initial magnitude.
        initial_probability: the initial probability. If None, the augmentation will be randomly
            applied according to he augmentation sampling range.
        magnitude_range: the sampling range for random sampling and clamping the optimized magnitude.
        temperature: retained for API compatibility.
        symmetric_megnitude: if to randomly assign the magnitude as negative or not.

    """

    @staticmethod
    def _process_magnitude(magnitude: torch.Tensor) -> torch.Tensor:
        # make it sign-agnostic
        return magnitude * 180

    def __init__(
        self,
        initial_magnitude: Optional[float] = 0.1,
        initial_probability: float = 0.5,
        magnitude_range: Tuple[float, float] = (0.0, 0.3),
        temperature: float = 0.1,
        symmetric_megnitude: bool = True,
    ) -> None:
        if symmetric_megnitude and magnitude_range[0] < 0:
            raise ValueError(
                f"Lower bound of {self.__class__.__name__} is a symmetric operation. "
                f"The lower bound must above 0. Got {magnitude_range[0]}."
            )
        super().__init__(
            K.RandomShear(magnitude_range, same_on_batch=False, p=initial_probability, align_corners=True),
            initial_magnitude=[("shear_x", initial_magnitude)],
            temperature=temperature,
            symmetric_megnitude=symmetric_megnitude,
            magnitude_fn=ShearX._process_magnitude,
        )


class ShearY(OperationBase):
    """Apply shear operation along y-axis.

    Args:
        initial_magnitude: the initial magnitude.
        initial_probability: the initial probability. If None, the augmentation will be randomly
            applied according to he augmentation sampling range.
        magnitude_range: the sampling range for random sampling and clamping the optimized magnitude.
        temperature: retained for API compatibility.
        symmetric_megnitude: if to randomly assign the magnitude as negative or not.

    """

    @staticmethod
    def _process_magnitude(magnitude: torch.Tensor) -> torch.Tensor:
        # make it sign-agnostic
        return magnitude * 180

    def __init__(
        self,
        initial_magnitude: Optional[float] = 0.1,
        initial_probability: float = 0.5,
        magnitude_range: Tuple[float, float] = (0.0, 0.3),
        temperature: float = 0.1,
        symmetric_megnitude: bool = True,
    ) -> None:
        if symmetric_megnitude and magnitude_range[0] < 0:
            raise ValueError(
                f"Lower bound of {self.__class__.__name__} is a symmetric operation. "
                f"The lower bound must above 0. Got {magnitude_range[0]}."
            )
        super().__init__(
            K.RandomShear(
                (0.0, 0.0, magnitude_range[0], magnitude_range[1]),
                same_on_batch=False,
                p=initial_probability,
                align_corners=True,
            ),
            initial_magnitude=[("shear_y", initial_magnitude)],
            temperature=temperature,
            symmetric_megnitude=symmetric_megnitude,
            magnitude_fn=ShearY._process_magnitude,
        )


class TranslateX(OperationBase):
    """Apply translate operation along x-axis.

    Args:
        initial_magnitude: the initial magnitude.
        initial_probability: the initial probability. If None, the augmentation will be randomly
            applied according to he augmentation sampling range.
        magnitude_range: the sampling range for random sampling and clamping the optimized magnitude.
        temperature: retained for API compatibility.
        symmetric_megnitude: if to randomly assign the magnitude as negative or not.

    """

    def __init__(
        self,
        initial_magnitude: Optional[float] = 0.2,
        initial_probability: float = 0.5,
        magnitude_range: Tuple[float, float] = (0.0, 0.5),
        temperature: float = 0.1,
        symmetric_megnitude: bool = True,
    ) -> None:
        if symmetric_megnitude and magnitude_range[0] < 0:
            raise ValueError(
                f"Lower bound of {self.__class__.__name__} is a symmetric operation. "
                f"The lower bound must above 0. Got {magnitude_range[0]}."
            )
        super().__init__(
            K.RandomTranslate(magnitude_range, same_on_batch=False, p=initial_probability, align_corners=True),
            initial_magnitude=[("translate_x", initial_magnitude)],
            temperature=temperature,
            symmetric_megnitude=symmetric_megnitude,
        )


class TranslateY(OperationBase):
    """Apply translate operation along y-axis.

    Args:
        initial_magnitude: the initial magnitude.
        initial_probability: the initial probability. If None, the augmentation will be randomly
            applied according to he augmentation sampling range.
        magnitude_range: the sampling range for random sampling and clamping the optimized magnitude.
        temperature: retained for API compatibility.
        symmetric_megnitude: if to randomly assign the magnitude as negative or not.

    """

    def __init__(
        self,
        initial_magnitude: Optional[float] = 0.2,
        initial_probability: float = 0.5,
        magnitude_range: Tuple[float, float] = (0.0, 0.5),
        temperature: float = 0.1,
        symmetric_megnitude: bool = True,
    ) -> None:
        if symmetric_megnitude and magnitude_range[0] < 0:
            raise ValueError(
                f"Lower bound of {self.__class__.__name__} is a symmetric operation. "
                f"The lower bound must above 0. Got {magnitude_range[0]}."
            )
        super().__init__(
            K.RandomTranslate(None, magnitude_range, same_on_batch=False, p=initial_probability, align_corners=True),
            initial_magnitude=[("translate_y", initial_magnitude)],
            temperature=temperature,
            symmetric_megnitude=symmetric_megnitude,
        )
