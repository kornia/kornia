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

"""AutoAugment operation wrapper."""

from typing import Tuple

import torch

from kornia.augmentation.auto.operations import (
    AutoContrast,
    Brightness,
    Contrast,
    Equalize,
    Invert,
    OperationBase,
    Posterize,
    Rotate,
    Saturate,
    Sharpness,
    ShearX,
    ShearY,
    Solarize,
    TranslateX,
    TranslateY,
)


def _magnitude_bin(magnitudes: torch.Tensor, magnitude: int, name: str) -> Tuple[float, float]:
    """Return the magnitude range that bin ``magnitude`` selects from ``magnitudes``.

    A bin indexes two adjacent points, so the last usable bin is two short of the end.
    Indexing directly would let a negative bin wrap silently onto a reversed range, and
    an over-large one surface an ``IndexError`` naming an internal tensor rather than the
    policy entry the caller wrote.

    Args:
        magnitudes: The op's magnitude scale.
        magnitude: Bin index into that scale.
        name: Operation name, for the error message.

    Returns:
        The ``(low, high)`` range for that bin.

    Raises:
        ValueError: If ``magnitude`` is not a usable bin index.
    """
    last_bin = magnitudes.shape[0] - 2
    if not 0 <= magnitude <= last_bin:
        raise ValueError(f"Expect magnitude bin for `{name}` in [0, {last_bin}]. Got {magnitude}.")
    return magnitudes[magnitude].item(), magnitudes[magnitude + 1].item()


def shear_x(probability: float, magnitude: int) -> OperationBase:
    """Return ShearX op."""
    magnitudes = torch.linspace(-0.3, 0.3, 11) * 180.0
    return ShearX(
        None,
        probability,
        magnitude_range=_magnitude_bin(magnitudes, magnitude, "shear_x"),
        symmetric_megnitude=False,
    )


def shear_y(probability: float, magnitude: int) -> OperationBase:
    """Return ShearY op."""
    magnitudes = torch.linspace(-0.3, 0.3, 11) * 180.0
    return ShearY(
        None,
        probability,
        magnitude_range=_magnitude_bin(magnitudes, magnitude, "shear_y"),
        symmetric_megnitude=False,
    )


def translate_x(probability: float, magnitude: int) -> OperationBase:
    """Return TranslateX op."""
    magnitudes = torch.linspace(-0.5, 0.5, 11)
    return TranslateX(
        None,
        probability,
        magnitude_range=_magnitude_bin(magnitudes, magnitude, "translate_x"),
        symmetric_megnitude=False,
    )


def translate_y(probability: float, magnitude: int) -> OperationBase:
    """Return TranslateY op."""
    magnitudes = torch.linspace(-0.5, 0.5, 11)
    return TranslateY(
        None,
        probability,
        magnitude_range=_magnitude_bin(magnitudes, magnitude, "translate_y"),
        symmetric_megnitude=False,
    )


def rotate(probability: float, magnitude: int) -> OperationBase:
    """Return rotate op."""
    magnitudes = torch.linspace(-30, 30, 11)
    return Rotate(
        None,
        probability,
        magnitude_range=_magnitude_bin(magnitudes, magnitude, "rotate"),
        symmetric_megnitude=False,
    )


def auto_contrast(probability: float, _: int) -> OperationBase:
    """Return AutoConstrast op."""
    return AutoContrast(probability)


def invert(probability: float, _: int) -> OperationBase:
    """Return invert op."""
    return Invert(probability)


def equalize(probability: float, _: int) -> OperationBase:
    """Return equalize op."""
    return Equalize(probability)


def solarize(probability: float, magnitude: int) -> OperationBase:
    """Return solarize op."""
    magnitudes = torch.linspace(0, 255, 11) / 255.0
    return Solarize(None, probability, magnitude_range=_magnitude_bin(magnitudes, magnitude, "solarize"))


def posterize(probability: float, magnitude: int) -> OperationBase:
    """Return posterize op."""
    magnitudes = torch.linspace(4, 8, 11)
    return Posterize(None, probability, magnitude_range=_magnitude_bin(magnitudes, magnitude, "posterize"))


def contrast(probability: float, magnitude: int) -> OperationBase:
    """Return contrast op."""
    magnitudes = torch.linspace(0.1, 1.9, 11)
    return Contrast(None, probability, magnitude_range=_magnitude_bin(magnitudes, magnitude, "contrast"))


def brightness(probability: float, magnitude: int) -> OperationBase:
    """Return brightness op."""
    magnitudes = torch.linspace(0.1, 1.9, 11)
    return Brightness(None, probability, magnitude_range=_magnitude_bin(magnitudes, magnitude, "brightness"))


def sharpness(probability: float, magnitude: int) -> OperationBase:
    """Return sharpness op."""
    magnitudes = torch.linspace(0.1, 1.9, 11)
    return Sharpness(None, probability, magnitude_range=_magnitude_bin(magnitudes, magnitude, "sharpness"))


def color(probability: float, magnitude: int) -> OperationBase:
    """Return color op."""
    magnitudes = torch.linspace(0.1, 1.9, 11)
    return Saturate(None, probability, magnitude_range=_magnitude_bin(magnitudes, magnitude, "color"))
