"""AutoAugment operation wrapper."""
import torch

from kornia.augmentation.auto.operations import OperationBase
from kornia.augmentation.auto.operations import (
    Brightness,
    Contrast,
    Equalize,
    Invert,
    Posterize,
    Solarize,
    Sharpness,
    Rotate,
    ShearX,
    ShearY,
    TranslateX,
    TranslateY,
)


def shear_x(probability: float, magnitude: float) -> OperationBase:
    magnitudes = torch.linspace(-0.3, 0.3, 11) * 180.
    return ShearX(
        None, probability, magnitude_range=(magnitudes[magnitude].item(), magnitudes[magnitude + 1].item()))


def shear_y(probability: float, magnitude: float) -> OperationBase:
    magnitudes = torch.linspace(-0.3, 0.3, 11) * 180.
    return ShearY(
        None, probability, magnitude_range=(magnitudes[magnitude].item(), magnitudes[magnitude + 1].item()))


def translate_x(probability: float, magnitude: float) -> OperationBase:
    magnitudes = torch.linspace(-0.5, 0.5, 11)
    return TranslateX(
        None, probability, magnitude_range=(magnitudes[magnitude].item(), magnitudes[magnitude + 1].item()))


def translate_y(probability: float, magnitude: float) -> OperationBase:
    magnitudes = torch.linspace(-0.5, 0.5, 11)
    return TranslateY(
        None, probability, magnitude_range=(magnitudes[magnitude].item(), magnitudes[magnitude + 1].item()))


def rotate(probability: float, magnitude: float) -> OperationBase:
    magnitudes = torch.linspace(-30, 30, 11)
    return Rotate(
        None, probability, magnitude_range=(magnitudes[magnitude].item(), magnitudes[magnitude + 1].item()))


def auto_contrast(probability: float, magnitude: float) -> OperationBase:
    raise NotImplementedError


def invert(probability: float, _: float) -> OperationBase:
    return Invert(probability)


def equalize(probability: float, _: float) -> OperationBase:
    return Equalize(probability)


def solarize(probability: float, magnitude: float) -> OperationBase:
    magnitudes = torch.linspace(0, 255, 11) / 255.
    return Solarize(None, probability, magnitude_range=(magnitudes[magnitude].item(), magnitudes[magnitude + 1].item()))


def posterize(probability: float, magnitude: float) -> OperationBase:
    magnitudes = torch.linspace(4, 8, 11)
    return Posterize(
        None, probability, magnitude_range=(magnitudes[magnitude].item(), magnitudes[magnitude + 1].item()))


def contrast(probability: float, magnitude: float) -> OperationBase:
    magnitudes = torch.linspace(0.1, 1.9, 11)
    return Contrast(
        None, probability, magnitude_range=(magnitudes[magnitude].item(), magnitudes[magnitude + 1].item()))


def brightness(probability: float, magnitude: float) -> OperationBase:
    magnitudes = torch.linspace(0.1, 1.9, 11)
    return Brightness(
        None, probability, magnitude_range=(magnitudes[magnitude].item(), magnitudes[magnitude + 1].item()))


def sharpness(probability: float, magnitude: float) -> OperationBase:
    magnitudes = torch.linspace(0.1, 1.9, 11)
    return Sharpness(
        None, probability, magnitude_range=(magnitudes[magnitude].item(), magnitudes[magnitude + 1].item()))


def color(probability: float, magnitude: float) -> OperationBase:
    raise NotImplementedError