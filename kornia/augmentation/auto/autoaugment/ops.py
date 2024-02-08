"""AutoAugment operation wrapper."""
from kornia.augmentation.auto.operations import AutoContrast
from kornia.augmentation.auto.operations import Brightness
from kornia.augmentation.auto.operations import Contrast
from kornia.augmentation.auto.operations import Equalize
from kornia.augmentation.auto.operations import Invert
from kornia.augmentation.auto.operations import OperationBase
from kornia.augmentation.auto.operations import Posterize
from kornia.augmentation.auto.operations import Rotate
from kornia.augmentation.auto.operations import Saturate
from kornia.augmentation.auto.operations import Sharpness
from kornia.augmentation.auto.operations import ShearX
from kornia.augmentation.auto.operations import ShearY
from kornia.augmentation.auto.operations import Solarize
from kornia.augmentation.auto.operations import TranslateX
from kornia.augmentation.auto.operations import TranslateY
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
