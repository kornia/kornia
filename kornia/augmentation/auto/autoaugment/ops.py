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
from kornia.core import linspace


def shear_x(probability: float, magnitude: int) -> OperationBase:
    magnitudes = linspace(-0.3, 0.3, 11) * 180.0
    return ShearX(
        None,
        probability,
        magnitude_range=(magnitudes[magnitude].item(), magnitudes[magnitude + 1].item()),
        symmetric_megnitude=False,
    )


def shear_y(probability: float, magnitude: int) -> OperationBase:
    magnitudes = linspace(-0.3, 0.3, 11) * 180.0
    return ShearY(
        None,
        probability,
        magnitude_range=(magnitudes[magnitude].item(), magnitudes[magnitude + 1].item()),
        symmetric_megnitude=False,
    )


def translate_x(probability: float, magnitude: int) -> OperationBase:
    magnitudes = linspace(-0.5, 0.5, 11)
    return TranslateX(
        None,
        probability,
        magnitude_range=(magnitudes[magnitude].item(), magnitudes[magnitude + 1].item()),
        symmetric_megnitude=False,
    )


def translate_y(probability: float, magnitude: int) -> OperationBase:
    magnitudes = linspace(-0.5, 0.5, 11)
    return TranslateY(
        None,
        probability,
        magnitude_range=(magnitudes[magnitude].item(), magnitudes[magnitude + 1].item()),
        symmetric_megnitude=False,
    )


def rotate(probability: float, magnitude: int) -> OperationBase:
    magnitudes = linspace(-30, 30, 11)
    return Rotate(
        None,
        probability,
        magnitude_range=(magnitudes[magnitude].item(), magnitudes[magnitude + 1].item()),
        symmetric_megnitude=False,
    )


def auto_contrast(probability: float, _: int) -> OperationBase:
    return AutoContrast(probability)


def invert(probability: float, _: int) -> OperationBase:
    return Invert(probability)


def equalize(probability: float, _: int) -> OperationBase:
    return Equalize(probability)


def solarize(probability: float, magnitude: int) -> OperationBase:
    magnitudes = linspace(0, 255, 11) / 255.0
    return Solarize(None, probability, magnitude_range=(magnitudes[magnitude].item(), magnitudes[magnitude + 1].item()))


def posterize(probability: float, magnitude: int) -> OperationBase:
    magnitudes = linspace(4, 8, 11)
    return Posterize(
        None, probability, magnitude_range=(magnitudes[magnitude].item(), magnitudes[magnitude + 1].item())
    )


def contrast(probability: float, magnitude: int) -> OperationBase:
    magnitudes = linspace(0.1, 1.9, 11)
    return Contrast(None, probability, magnitude_range=(magnitudes[magnitude].item(), magnitudes[magnitude + 1].item()))


def brightness(probability: float, magnitude: int) -> OperationBase:
    magnitudes = linspace(0.1, 1.9, 11)
    return Brightness(
        None, probability, magnitude_range=(magnitudes[magnitude].item(), magnitudes[magnitude + 1].item())
    )


def sharpness(probability: float, magnitude: int) -> OperationBase:
    magnitudes = linspace(0.1, 1.9, 11)
    return Sharpness(
        None, probability, magnitude_range=(magnitudes[magnitude].item(), magnitudes[magnitude + 1].item())
    )


def color(probability: float, magnitude: int) -> OperationBase:
    magnitudes = linspace(0.1, 1.9, 11)
    return Saturate(None, probability, magnitude_range=(magnitudes[magnitude].item(), magnitudes[magnitude + 1].item()))
