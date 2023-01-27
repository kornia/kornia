from typing import Optional, Tuple

import torch

from kornia.augmentation.auto.operations.base import OperationBase
from kornia.augmentation import (
    RandomBrightness,
    RandomContrast,
    RandomHue,
    RandomSaturation,
    RandomEqualize,
    RandomGrayscale,
    RandomInvert,
    RandomPosterize,
    RandomSolarize,
    RandomSharpness,
    RandomHorizontalFlip,
    RandomVerticalFlip,
    RandomRotation,
)
from kornia.core import Tensor
from kornia.grad_estimator import STEFunction

__all__ = [
    "Brightness", "Contrast", "Hue", "Saturate", "Equalize", "Gray", "Invert", "Posterize",
    "Solarize", "SolarizeAdd", "Sharpness", "HorizontalFlip", "VerticalFlip", "Rotate",
]


class Brightness(OperationBase):
    def __init__(
        self,
        initial_magnitude: Optional[float] = 0.5,
        initial_probability: float = 0.5,
        magnitude_range: Tuple[float, float] = (0.2, 1.8),
        temperature: float = 0.1,
    ):
        super(Brightness, self).__init__(
            RandomBrightness(magnitude_range, same_on_batch=False, p=initial_probability),
            initial_magnitude=[("brightness_factor", initial_magnitude)],
            temperature=temperature,
        )


class Contrast(OperationBase):
    def __init__(
        self,
        initial_magnitude: Optional[float] = 0.5,
        initial_probability: float = 0.5,
        magnitude_range: Tuple[float, float] = (0.2, 1.8),
        temperature: float = 0.1,
    ):
        super(Contrast, self).__init__(
            RandomContrast(magnitude_range, same_on_batch=False, p=initial_probability),
            initial_magnitude=[("contrast_factor", initial_magnitude)],
            temperature=temperature,
        )


class Hue(OperationBase):
    def __init__(
        self,
        initial_magnitude: Optional[float] = 0.,
        initial_probability: float = 0.5,
        magnitude_range: Tuple[float, float] = (-.5, .5),
        temperature: float = 0.1,
    ):
        super(Hue, self).__init__(
            RandomHue(magnitude_range, same_on_batch=False, p=initial_probability),
            initial_magnitude=[("hue_factor", initial_magnitude)],
            temperature=temperature,
        )


class Saturate(OperationBase):
    def __init__(
        self,
        initial_magnitude: Optional[float] = 0.5,
        initial_probability: float = 0.5,
        magnitude_range: Tuple[float, float] = (0.2, 1.8),
        temperature: float = 0.1,
    ):
        super(Saturate, self).__init__(
            RandomSaturation(magnitude_range, same_on_batch=False, p=initial_probability),
            initial_magnitude=[("saturation_factor", initial_magnitude)],
            temperature=temperature,
        )


# TODO: Equalize cannot update probabilities yet.
class Equalize(OperationBase):
    def __init__(
        self,
        initial_probability: float = 0.5,
        temperature: float = 0.1,
    ):
        super(Equalize, self).__init__(
            RandomEqualize(same_on_batch=False, p=initial_probability),
            initial_magnitude=None,
            temperature=temperature,
            gradient_estimator=STEFunction,
        )


class Gray(OperationBase):
    def __init__(
        self,
        initial_probability: float = 0.5,
        temperature: float = 0.1,
    ):
        super(Gray, self).__init__(
            RandomGrayscale(same_on_batch=False, p=initial_probability),
            initial_magnitude=None,
            temperature=temperature,
        )


class Invert(OperationBase):
    def __init__(
        self,
        initial_probability: float = 0.5,
        temperature: float = 0.1,
    ):
        super(Invert, self).__init__(
            RandomInvert(same_on_batch=False, p=initial_probability),
            initial_magnitude=None,
            temperature=temperature,
        )


class Posterize(OperationBase):

    @staticmethod
    def _process_magnitude(magnitude: Tensor) -> Tensor:
        return magnitude.long()

    def __init__(
        self,
        initial_magnitude: Optional[float] = 4.,
        initial_probability: float = 0.5,
        magnitude_range: Tuple[float, float] = (1, 8),
        temperature: float = 0.1,
    ):
        super(Posterize, self).__init__(
            RandomPosterize(magnitude_range, same_on_batch=False, p=initial_probability),
            initial_magnitude=[("bits_factor", initial_magnitude)],
            temperature=temperature,
            magnitude_fn=Posterize._process_magnitude,
            gradient_estimator=STEFunction
        )


class Solarize(OperationBase):
    def __init__(
        self,
        initial_magnitude: Optional[float] = 0.5,
        initial_probability: float = 0.5,
        magnitude_range: Tuple[float, float] = (0., 1.),
        temperature: float = 0.1,
    ):
        super(Solarize, self).__init__(
            RandomSolarize(magnitude_range, additions=0., same_on_batch=False, p=initial_probability),
            initial_magnitude=[("thresholds", initial_magnitude)],
            temperature=temperature,
            gradient_estimator=STEFunction
        )


class SolarizeAdd(OperationBase):
    def __init__(
        self,
        initial_magnitude: Optional[float] = 0.,
        initial_probability: float = 0.5,
        magnitude_range: Tuple[float, float] = (- 0.3, 0.3),
        temperature: float = 0.1,
    ):
        super(SolarizeAdd, self).__init__(
            RandomSolarize(thresholds=0.5, additions=magnitude_range, same_on_batch=False, p=initial_probability),
            initial_magnitude=[("additions", initial_magnitude)],
            temperature=temperature,
            gradient_estimator=STEFunction
        )


class Sharpness(OperationBase):
    def __init__(
        self,
        initial_magnitude: Optional[float] = 0.5,
        initial_probability: float = 0.5,
        magnitude_range: Tuple[float, float] = (0.1, 1.9),
        temperature: float = 0.1,
    ):
        super(Sharpness, self).__init__(
            RandomSharpness(magnitude_range, same_on_batch=False, p=initial_probability),
            initial_magnitude=[("sharpness", initial_magnitude)],
            temperature=temperature,
        )


class HorizontalFlip(OperationBase):
    def __init__(
        self,
        initial_probability: float = 0.5,
        temperature: float = 0.1,
    ):
        super(HorizontalFlip, self).__init__(
            RandomHorizontalFlip(same_on_batch=False, p=initial_probability),
            initial_magnitude=None,
            temperature=temperature,
        )


class VerticalFlip(OperationBase):
    def __init__(
        self,
        initial_probability: float = 0.5,
        temperature: float = 0.1,
    ):
        super(VerticalFlip, self).__init__(
            RandomVerticalFlip(same_on_batch=False, p=initial_probability),
            initial_magnitude=None,
            temperature=temperature,
        )


class Rotate(OperationBase):

    @staticmethod
    def _process_magnitude(magnitude: Tensor) -> Tensor:
        # make it sign-agnostic
        return magnitude * (torch.randint(0, 1, (1,)) * 2 - 1).item()

    def __init__(
        self,
        initial_magnitude: Optional[float] = 15.,
        initial_probability: float = 0.5,
        magnitude_range: Tuple[float, float] = (-30., 30.),
        temperature: float = 0.1,
    ):
        super(Rotate, self).__init__(
            RandomRotation(magnitude_range, same_on_batch=False, p=initial_probability),
            initial_magnitude=[("degrees", initial_magnitude)],
            temperature=temperature,
            magnitude_fn=Rotate._process_magnitude
        )

