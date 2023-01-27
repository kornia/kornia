"""RandAugment operation wrapper."""
from kornia.augmentation.auto.operations import OperationBase
from kornia.augmentation.auto.operations import (
    Brightness,
    Contrast,
    Hue,
    Saturate,
    Equalize,
    Gray,
    Invert,
    Posterize,
    Solarize,
    SolarizeAdd,
    Sharpness,
    HorizontalFlip,
    VerticalFlip,
    Rotate
)


def shear_x(min_mag: float, max_mag: float) -> OperationBase:
    raise NotImplementedError


def shear_y(min_mag: float, max_mag: float) -> OperationBase:
    raise NotImplementedError


def translate_x(min_mag: float, max_mag: float) -> OperationBase:
    raise NotImplementedError


def translate_y(min_mag: float, max_mag: float) -> OperationBase:
    raise NotImplementedError


def rotate(min_mag: float, max_mag: float) -> OperationBase:
    return Rotate(None, 1., magnitude_range=(min_mag, max_mag))


def auto_contrast(min_mag: float, max_mag: float) -> OperationBase:
    raise NotImplementedError


def invert(min_mag: float, max_mag: float) -> OperationBase:
    return Invert(1.)


def equalize(min_mag: float, max_mag: float) -> OperationBase:
    return Equalize(1.)


def solarize(min_mag: float, max_mag: float) -> OperationBase:
    return Solarize(None, 1., magnitude_range=(min_mag, max_mag))


def solarize_add(min_mag: float, max_mag: float) -> OperationBase:
    return SolarizeAdd(None, 1., magnitude_range=(min_mag, max_mag))


def posterize(min_mag: float, max_mag: float) -> OperationBase:
    return Posterize(None, 1., magnitude_range=(min_mag, max_mag))


def contrast(min_mag: float, max_mag: float) -> OperationBase:
    return Contrast(None, 1., magnitude_range=(min_mag, max_mag))


def brightness(min_mag: float, max_mag: float) -> OperationBase:
    return Brightness(None, 1., magnitude_range=(min_mag, max_mag))


def sharpness(min_mag: float, max_mag: float) -> OperationBase:
    return Sharpness(None, 1., magnitude_range=(min_mag, max_mag))


def color(min_mag: float, max_mag: float) -> OperationBase:
    raise NotImplementedError
