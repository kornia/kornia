"""RandAugment operation wrapper."""
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
from kornia.augmentation.auto.operations import SolarizeAdd
from kornia.augmentation.auto.operations import TranslateX
from kornia.augmentation.auto.operations import TranslateY


def shear_x(min_mag: float, max_mag: float) -> OperationBase:
    if min_mag != -max_mag:
        raise ValueError(
            f"{ShearX.__name__} is a symmetric operation that `- min_mag == max_mag`. Got [{min_mag}, {max_mag}]"
        )
    return ShearX(None, 1.0, magnitude_range=(0.0, max_mag), symmetric_megnitude=True)


def shear_y(min_mag: float, max_mag: float) -> OperationBase:
    if min_mag != -max_mag:
        raise ValueError(
            f"{ShearY.__name__} is a symmetric operation that `- min_mag == max_mag`. Got [{min_mag}, {max_mag}]"
        )
    return ShearY(None, 1.0, magnitude_range=(0.0, max_mag), symmetric_megnitude=True)


def translate_x(min_mag: float, max_mag: float) -> OperationBase:
    if min_mag != -max_mag:
        raise ValueError(
            f"{TranslateX.__name__} is a symmetric operation that `- min_mag == max_mag`. Got [{min_mag}, {max_mag}]"
        )
    return TranslateX(None, 1.0, magnitude_range=(0.0, max_mag), symmetric_megnitude=True)


def translate_y(min_mag: float, max_mag: float) -> OperationBase:
    if min_mag != -max_mag:
        raise ValueError(
            f"{TranslateY.__name__} is a symmetric operation that `- min_mag == max_mag`. Got [{min_mag}, {max_mag}]"
        )
    return TranslateY(None, 1.0, magnitude_range=(0.0, max_mag), symmetric_megnitude=True)


def rotate(min_mag: float, max_mag: float) -> OperationBase:
    if min_mag != -max_mag:
        raise ValueError(
            f"{Rotate.__name__} is a symmetric operation that `- min_mag == max_mag`. Got [{min_mag}, {max_mag}]"
        )
    return Rotate(None, 1.0, magnitude_range=(0.0, max_mag), symmetric_megnitude=True)


def auto_contrast(min_mag: float, max_mag: float) -> OperationBase:
    return AutoContrast(1.0)


def invert(min_mag: float, max_mag: float) -> OperationBase:
    return Invert(1.0)


def equalize(min_mag: float, max_mag: float) -> OperationBase:
    return Equalize(1.0)


def solarize(min_mag: float, max_mag: float) -> OperationBase:
    return Solarize(None, 1.0, magnitude_range=(min_mag, max_mag))


def solarize_add(min_mag: float, max_mag: float) -> OperationBase:
    return SolarizeAdd(None, 1.0, magnitude_range=(min_mag, max_mag))


def posterize(min_mag: float, max_mag: float) -> OperationBase:
    return Posterize(None, 1.0, magnitude_range=(min_mag, max_mag))


def contrast(min_mag: float, max_mag: float) -> OperationBase:
    return Contrast(None, 1.0, magnitude_range=(min_mag, max_mag))


def brightness(min_mag: float, max_mag: float) -> OperationBase:
    return Brightness(None, 1.0, magnitude_range=(min_mag, max_mag))


def sharpness(min_mag: float, max_mag: float) -> OperationBase:
    return Sharpness(None, 1.0, magnitude_range=(min_mag, max_mag))


def color(min_mag: float, max_mag: float) -> OperationBase:
    return Saturate(None, 1.0, magnitude_range=(min_mag, max_mag))
