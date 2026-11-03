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
"""Implement color and geometric operations used by automatic augmentation policies.

Which operation to use:

- :class:`AutoContrast`, :class:`Equalize`, :class:`Contrast`, :class:`Brightness`, :class:`Hue`,
  :class:`Saturate` and :class:`Gray` change image tone or color.
- :class:`Invert`, :class:`Posterize`, :class:`Solarize`, :class:`SolarizeAdd` and :class:`Sharpness` change values
  or local appearance.
- :class:`HorizontalFlip`, :class:`VerticalFlip`, :class:`Rotate`, :class:`ShearX`, :class:`ShearY`,
  :class:`TranslateX` and :class:`TranslateY` change image geometry.
"""

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


def _fraction_to_pixels(mag: Optional[torch.Tensor], size: int) -> Optional[torch.Tensor]:
    # RandomTranslate's generator samples a fraction of the image and stores it in pixels, so a magnitude that
    # replaces the draw has to be converted the same way.
    return None if mag is None else mag * size


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
        """Apply the sampled PIL-style enhancement factor to the input."""
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
        """Apply the sampled PIL-style enhancement factor to the input."""
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
        """Apply the sampled PIL-style enhancement factor to the input."""
        factor = _to_device_nonblocking(params["saturation_factor"], input.device, input.dtype)
        return adjust_saturation_with_gray_subtraction(_clamp_unit_range(input), factor)


class AutoContrast(OperationBase):
    """Stretch each image channel's occupied range to the full value range.

    ``output = AutoContrast(probability)(input)`` stretches each selected channel's occupied range in a floating
    ``(B, C, H, W)`` batch toward ``[0, 1]`` . Use :class:`Contrast` to adjust contrast around the grayscale mean
    instead.

    Args:
        initial_probability: probability of applying the operation to each image.
        temperature: retained for API compatibility.
        symmetric_megnitude: if to randomly assign the magnitude as negative or not.

    See Also:
        - :class:`~kornia.augmentation.auto.operations.ops.Contrast` : changes contrast by blending toward the grayscale
          mean.

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

    ``output = Brightness(magnitude, probability)(input)`` multiplies selected images in a floating ``(B, C, H, W)``
    batch by the magnitude and keeps the shape. Inputs are clamped into ``[0, 1]`` before adjustment, as are outputs;
    use :class:`~kornia.augmentation.RandomBrightness` for additive brightness.

    Args:
        initial_probability: probability of applying the operation to each image.
        initial_magnitude: the initial magnitude.
        magnitude_range: the sampling range for random sampling and clamping the optimized magnitude.
        temperature: retained for API compatibility.
        symmetric_megnitude: if to randomly assign the magnitude as negative or not.

    See Also:
        - :class:`~kornia.augmentation.auto.operations.ops.Contrast`: changes contrast rather than brightness.

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
    """Apply PIL-style contrast by blending each image towards its mean grayscale value.

    ``output = Contrast(magnitude, probability)(input)`` scales selected floating images ``(B, C, H, W)`` around their
    mean. RGB inputs use the grayscale mean; other channel counts use the mean over all channels and pixels. Selected
    inputs and outputs are clamped into ``[0, 1]`` ; use :class:`~kornia.augmentation.RandomContrast` for direct
    multiplication instead.

    Args:
        initial_probability: probability of applying the operation to each image.
        initial_magnitude: the initial magnitude.
        magnitude_range: the sampling range for random sampling and clamping the optimized magnitude.
        temperature: retained for API compatibility.
        symmetric_megnitude: if to randomly assign the magnitude as negative or not.

    See Also:
        - :class:`~kornia.augmentation.auto.operations.ops.Brightness`: changes brightness rather than contrast.

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
    """Shift hue by a policy-controlled amount.

    ``output = Hue(magnitude, probability)(input)`` shifts hue in a floating RGB batch ``(B, 3, H, W)`` and keeps its
    shape. The magnitude is a fraction of a full hue turn. Selected inputs and outputs are clamped to ``[0, 1]`` ; use
    :class:`Saturate` to change saturation instead.

    Args:
        initial_probability: probability of applying the operation to each image.
        initial_magnitude: the initial magnitude.
        magnitude_range: the sampling range for random sampling and clamping the optimized magnitude.
        temperature: retained for API compatibility.
        symmetric_megnitude: if to randomly assign the magnitude as negative or not.

    See Also:
        - :class:`~kornia.augmentation.auto.operations.ops.Saturate`: adjusts saturation rather than hue.

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

    ``output = Saturate(magnitude, probability)(input)`` blends selected floating images ``(B, C, H, W)`` with their
    grayscale versions. One or three channels are accepted; grayscale inputs only undergo the ``[0, 1]`` input/output
    clamps. Use :class:`Hue` to shift hue instead.

    Unlike :class:`~kornia.augmentation.RandomSaturation`, this blends RGB
    values with grayscale rather than scaling HSV saturation.

    Args:
        initial_probability: probability of applying the operation to each image.
        initial_magnitude: the initial magnitude.
        magnitude_range: the sampling range for random sampling and clamping the optimized magnitude.
        temperature: retained for API compatibility.
        symmetric_megnitude: if to randomly assign the magnitude as negative or not.

    See Also:
        - :class:`~kornia.augmentation.auto.operations.ops.Hue`: shifts hue rather than saturation.

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
    """Equalize image histograms when selected by a policy probability gate.

    ``output = Equalize(probability)(input)`` equalizes the histograms of selected images in a floating ``(B, C, H, W)``
    batch and keeps its shape. Selected inputs and outputs are clamped to ``[0, 1]`` . Use :class:`AutoContrast` to
    stretch channel ranges without histogram equalization.

    Args:
        initial_probability: probability of applying the operation to each image.
        temperature: retained for API compatibility.

    Note:
        Equalize cannot update probabilities yet.

    See Also:
        - :class:`~kornia.augmentation.auto.operations.ops.AutoContrast`: stretches ranges without equalization.

    """

    def __init__(self, initial_probability: float = 0.5, temperature: float = 0.1) -> None:
        super().__init__(
            K.RandomEqualize(same_on_batch=False, p=initial_probability),
            initial_magnitude=None,
            temperature=temperature,
            symmetric_megnitude=False,
        )


class Gray(OperationBase):
    """Convert selected images to grayscale while retaining their channel layout.

    ``output = Gray(probability)(input)`` replaces each selected floating image ``(B, C, H, W)`` with a channel average
    broadcast back to all channels. RGB inputs use luminance weights; other nonempty channel counts use equal weights.
    Values are not clamped; use :class:`Saturate` for a continuous blend toward grayscale.

    Args:
        initial_probability: probability of applying the operation to each image.
        temperature: retained for API compatibility.

    See Also:
        - :class:`~kornia.augmentation.auto.operations.ops.Saturate`: adjusts saturation continuously.

    """

    def __init__(self, initial_probability: float = 0.5, temperature: float = 0.1) -> None:
        super().__init__(
            K.RandomGrayscale(same_on_batch=False, p=initial_probability),
            initial_magnitude=None,
            temperature=temperature,
            symmetric_megnitude=False,
        )


class Invert(OperationBase):
    """Invert selected image values around the operation's supported range.

    ``output = Invert(probability)(input)`` replaces selected values with ``1 - input`` in a floating ``(B, C, H, W)``
    batch. It preserves the shape and does not clamp values; use :class:`Solarize` for thresholded inversion.

    Args:
        initial_probability: probability of applying the operation to each image.
        temperature: retained for API compatibility.

    See Also:
        - :class:`~kornia.augmentation.auto.operations.ops.Solarize`: inverts values above a threshold.

    """

    def __init__(self, initial_probability: float = 0.5, temperature: float = 0.1) -> None:
        super().__init__(
            K.RandomInvert(same_on_batch=False, p=initial_probability),
            initial_magnitude=None,
            temperature=temperature,
            symmetric_megnitude=False,
        )


class Posterize(OperationBase):
    """Reduce selected images to a policy-controlled number of bits.

    ``output = Posterize(magnitude, probability)(input)`` reduces the retained bits of selected floating
    ``(B, C, H, W)`` images. Selected inputs are clamped to ``[0, 1]`` and the magnitude is truncated to an integer bit
    count; use :class:`Solarize` for thresholded inversion instead.

    Args:
        initial_magnitude: the initial magnitude.
        initial_probability: probability of applying the operation to each image.
        magnitude_range: the sampling range for random sampling and clamping the optimized magnitude.
        temperature: retained for API compatibility.
        symmetric_megnitude: if to randomly assign the magnitude as negative or not.

    See Also:
        - :class:`~kornia.augmentation.auto.operations.ops.Solarize`: applies thresholded inversion.

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
    """Invert values at or above a policy-controlled solarization threshold.

    ``output = Solarize(magnitude, probability)(input)`` inverts values at or above the magnitude's threshold in
    selected floating ``(B, C, H, W)`` images after clamping inputs to ``[0, 1]`` . Use :class:`SolarizeAdd` to add an
    intensity offset before inversion.

    Args:
        initial_magnitude: the initial magnitude.
        initial_probability: probability of applying the operation to each image.
        magnitude_range: the range for sampling and clamping the magnitude.
        symmetric_megnitude: whether to assign a random sign to the magnitude.
        temperature: retained for API compatibility.

    See Also:
        - :class:`~kornia.augmentation.auto.operations.ops.Invert`: inverts all values rather than a thresholded set.

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
    """Add an intensity offset, then solarize at a sampled threshold.

    ``output = SolarizeAdd(magnitude, probability)(input)`` clamps selected floating images ``(B, C, H, W)`` into
    ``[0, 1]`` , adds the magnitude to every pixel, clamps again, then inverts values at or above a sampled threshold.
    Thresholds are drawn from ``[0, 1]`` per selected image; use :class:`Solarize` to control the threshold with the
    magnitude instead.

    Args:
        initial_magnitude: the initial magnitude.
        initial_probability: probability of applying the operation to each image.
        magnitude_range: the sampling range for random sampling and clamping the optimized magnitude.
        temperature: retained for API compatibility.
        symmetric_megnitude: if to randomly assign the magnitude as negative or not.

    See Also:
        - :class:`~kornia.augmentation.auto.operations.ops.Solarize`: controls the threshold without an offset.

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
    """Blend selected images with a smoothed copy at a policy-controlled factor.

    ``output = Sharpness(magnitude, probability)(input)`` blends selected floating ``(B, C, H, W)`` images with a
    smoothed copy. A factor of zero selects the smoothed image, one keeps the input, and larger factors sharpen;
    selected inputs and outputs are clamped into ``[0, 1]`` . Height and width must be at least three pixels when the
    operation runs.

    Args:
        initial_magnitude: the initial magnitude.
        initial_probability: probability of applying the operation to each image.
        magnitude_range: the sampling range for random sampling and clamping the optimized magnitude.
        temperature: retained for API compatibility.
        symmetric_megnitude: if to randomly assign the magnitude as negative or not.

    See Also:
        - :class:`~kornia.augmentation.auto.operations.ops.Contrast`: changes contrast rather than sharpness.

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
    """Reverse image columns with a policy probability gate.

    ``output = HorizontalFlip(probability)(input)`` reverses columns of selected floating ``(B, C, H, W)`` images,
    preserving shape and pixel values. Use :class:`VerticalFlip` to reverse rows.

    Args:
        initial_probability: probability of applying the operation to each image.
        temperature: retained for API compatibility.

    See Also:
        - :class:`~kornia.augmentation.auto.operations.ops.VerticalFlip`: reverses image rows.

    """

    def __init__(self, initial_probability: float = 0.5, temperature: float = 0.1) -> None:
        super().__init__(
            K.RandomHorizontalFlip(same_on_batch=False, p=initial_probability),
            initial_magnitude=None,
            temperature=temperature,
            symmetric_megnitude=False,
        )


class VerticalFlip(OperationBase):
    """Reverse image rows with a policy probability gate.

    ``output = VerticalFlip(probability)(input)`` reverses rows of selected floating ``(B, C, H, W)`` images, preserving
    shape and pixel values. Use :class:`HorizontalFlip` to reverse columns.

    Args:
        initial_probability: probability of applying the operation to each image.
        temperature: retained for API compatibility.

    See Also:
        - :class:`~kornia.augmentation.auto.operations.ops.HorizontalFlip`: reverses image columns.

    """

    def __init__(self, initial_probability: float = 0.5, temperature: float = 0.1) -> None:
        super().__init__(
            K.RandomVerticalFlip(same_on_batch=False, p=initial_probability),
            initial_magnitude=None,
            temperature=temperature,
            symmetric_megnitude=False,
        )


class Rotate(OperationBase):
    """Rotate selected images by a policy-controlled angle.

    ``output = Rotate(magnitude, probability)(input)`` rotates selected floating ``(B, C, H, W)`` images about their
    centers, keeping the output size. The magnitude is in degrees with a random sign by default; use :class:`ShearX` or
    :class:`ShearY` for shearing.

    Args:
        initial_magnitude: the initial magnitude.
        initial_probability: probability of applying the operation to each image.
        magnitude_range: the sampling range for random sampling and clamping the optimized magnitude.
        temperature: retained for API compatibility.
        symmetric_megnitude: if to randomly assign the magnitude as negative or not.

    See Also:
        - :class:`~kornia.augmentation.auto.operations.ops.ShearX`: shears along the x axis.
        - :class:`~kornia.augmentation.auto.operations.ops.ShearY`: shears along the y axis.

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
    """Shear selected images along the x axis by a policy-controlled amount.

    ``output = ShearX(magnitude, probability)(input)`` shears selected floating ``(B, C, H, W)`` images along x, keeping
    the output size. The magnitude is a shear factor with a random sign by default; use :class:`ShearY` for y-axis
    shearing.

    The magnitude is the shear factor, as in PIL and torchvision, whether it is sampled from ``magnitude_range``,
    learned, or passed to ``forward_parameters``. It is mapped to the wrapped augmentation's degrees with
    ``rad2deg(atan(magnitude))``, so a magnitude of ``0.3`` shears by about ``16.7`` degrees.

    Args:
        initial_magnitude: the initial magnitude, a shear factor.
        initial_probability: probability of applying the operation to each image.
        magnitude_range: the sampling range for random sampling and clamping the optimized magnitude.
        temperature: retained for API compatibility.
        symmetric_megnitude: if to randomly assign the magnitude as negative or not.

    See Also:
        - :class:`~kornia.augmentation.auto.operations.ops.ShearY`: shears along the y axis.

    """

    @staticmethod
    def _process_magnitude(magnitude: torch.Tensor) -> torch.Tensor:
        # the magnitude is the shear factor; map it to degrees as torchvision
        # does (atan keeps the sign)
        return torch.atan(magnitude) * 57.29577951308232  # 180/pi; aten::rad2deg is not ONNX-exportable

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
    """Shear selected images along the y axis by a policy-controlled amount.

    ``output = ShearY(magnitude, probability)(input)`` shears selected floating ``(B, C, H, W)`` images along y, keeping
    the output size. The magnitude is a shear factor with a random sign by default; use :class:`ShearX` for x-axis
    shearing.

    The magnitude is the shear factor, as in PIL and torchvision, whether it is sampled from ``magnitude_range``,
    learned, or passed to ``forward_parameters``. It is mapped to the wrapped augmentation's degrees with
    ``rad2deg(atan(magnitude))``, so a magnitude of ``0.3`` shears by about ``16.7`` degrees.

    Args:
        initial_magnitude: the initial magnitude, a shear factor.
        initial_probability: probability of applying the operation to each image.
        magnitude_range: the sampling range for random sampling and clamping the optimized magnitude.
        temperature: retained for API compatibility.
        symmetric_megnitude: if to randomly assign the magnitude as negative or not.

    See Also:
        - :class:`~kornia.augmentation.auto.operations.ops.ShearX`: shears along the x axis.

    """

    @staticmethod
    def _process_magnitude(magnitude: torch.Tensor) -> torch.Tensor:
        # the magnitude is the shear factor; map it to degrees as torchvision
        # does (atan keeps the sign)
        return torch.atan(magnitude) * 57.29577951308232  # 180/pi; aten::rad2deg is not ONNX-exportable

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
    """Translate selected images along the x axis by a policy-controlled amount.

    ``output = TranslateX(magnitude, probability)(input)`` shifts selected floating ``(B, C, H, W)`` images
    horizontally, keeping the output size. The magnitude is a fraction of image width with a random sign by default; use
    :class:`TranslateY` for vertical shifts.

    The magnitude is a fraction of the image width, whether it is sampled from ``magnitude_range``,
    learned, or passed to ``forward_parameters``; the wrapped augmentation translates by that fraction
    of the width in pixels.

    Args:
        initial_magnitude: the initial magnitude, a fraction of the image width.
        initial_probability: probability of applying the operation to each image.
        magnitude_range: the sampling range for random sampling and clamping the optimized magnitude.
        temperature: retained for API compatibility.
        symmetric_megnitude: if to randomly assign the magnitude as negative or not.

    See Also:
        - :class:`~kornia.augmentation.auto.operations.ops.TranslateY`: translates along the y axis.

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

    def forward_parameters(
        self, batch_shape: torch.Size, mag: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Sample parameters, converting a set magnitude from a fraction of the width to pixels.

        Args:
            batch_shape: Input batch shape used for sampling.
            mag: Optional magnitude override, a fraction of the image width.

        Returns:
            Parameter dictionary consumed by :meth:`forward`.
        """
        mag = _fraction_to_pixels(self.magnitude if mag is None else mag, batch_shape[-1])
        return super().forward_parameters(batch_shape, mag)


class TranslateY(OperationBase):
    """Translate selected images along the y axis by a policy-controlled amount.

    ``output = TranslateY(magnitude, probability)(input)`` shifts selected floating ``(B, C, H, W)`` images vertically,
    keeping the output size. The magnitude is a fraction of image height with a random sign by default; use
    :class:`TranslateX` for horizontal shifts.

    The magnitude is a fraction of the image height, whether it is sampled from ``magnitude_range``,
    learned, or passed to ``forward_parameters``; the wrapped augmentation translates by that fraction
    of the height in pixels.

    Args:
        initial_magnitude: the initial magnitude, a fraction of the image height.
        initial_probability: probability of applying the operation to each image.
        magnitude_range: the sampling range for random sampling and clamping the optimized magnitude.
        temperature: retained for API compatibility.
        symmetric_megnitude: if to randomly assign the magnitude as negative or not.

    See Also:
        - :class:`~kornia.augmentation.auto.operations.ops.TranslateX`: translates along the x axis.

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

    def forward_parameters(
        self, batch_shape: torch.Size, mag: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Sample parameters, converting a set magnitude from a fraction of the height to pixels.

        Args:
            batch_shape: Input batch shape used for sampling.
            mag: Optional magnitude override, a fraction of the image height.

        Returns:
            Parameter dictionary consumed by :meth:`forward`.
        """
        mag = _fraction_to_pixels(self.magnitude if mag is None else mag, batch_shape[-2])
        return super().forward_parameters(batch_shape, mag)
