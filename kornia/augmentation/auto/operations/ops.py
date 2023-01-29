from typing import Optional, Tuple

from kornia.augmentation import (
    RandomBrightness,
    RandomContrast,
    RandomEqualize,
    RandomGrayscale,
    RandomHorizontalFlip,
    RandomHue,
    RandomInvert,
    RandomPosterize,
    RandomRotation,
    RandomSaturation,
    RandomSharpness,
    RandomShear,
    RandomSolarize,
    RandomTranslate,
    RandomVerticalFlip,
)
from kornia.augmentation.auto.operations.base import OperationBase
from kornia.core import Tensor
from kornia.grad_estimator import STEFunction

__all__ = [
    "Brightness",
    "Contrast",
    "Hue",
    "Saturate",
    "Equalize",
    "Gray",
    "Invert",
    "Posterize",
    "Solarize",
    "SolarizeAdd",
    "Sharpness",
    "HorizontalFlip",
    "VerticalFlip",
    "Rotate",
    "ShearX",
    "ShearY",
    "TranslateX",
    "TranslateY",
]


class Brightness(OperationBase):
    """Apply brightness operation.

    Args:
        initial_probability: the initial probability.
        initial_magnitude: the initial magnitude.
        magnitude_range: the sampling range for random sampling and clamping the optimized magnitude.
        temperature: temperature for RelaxedBernoulli distribution used during training.
    """

    def __init__(
        self,
        initial_magnitude: Optional[float] = 0.5,
        initial_probability: float = 0.5,
        magnitude_range: Tuple[float, float] = (0.2, 1.8),
        temperature: float = 0.1,
        symmetric_megnitude: bool = False,
    ):
        super().__init__(
            RandomBrightness(magnitude_range, same_on_batch=False, p=initial_probability),
            initial_magnitude=[("brightness_factor", initial_magnitude)],
            temperature=temperature,
            symmetric_megnitude=symmetric_megnitude,
        )


class Contrast(OperationBase):
    """Apply contrast operation.

    Args:
        initial_probability: the initial probability.
        initial_magnitude: the initial magnitude.
        magnitude_range: the sampling range for random sampling and clamping the optimized magnitude.
        temperature: temperature for RelaxedBernoulli distribution used during training.
    """

    def __init__(
        self,
        initial_magnitude: Optional[float] = 0.5,
        initial_probability: float = 0.5,
        magnitude_range: Tuple[float, float] = (0.2, 1.8),
        temperature: float = 0.1,
        symmetric_megnitude: bool = False,
    ):
        super().__init__(
            RandomContrast(magnitude_range, same_on_batch=False, p=initial_probability),
            initial_magnitude=[("contrast_factor", initial_magnitude)],
            temperature=temperature,
            symmetric_megnitude=symmetric_megnitude,
        )


class Hue(OperationBase):
    """Apply hue operation.

    Args:
        initial_probability: the initial probability.
        initial_magnitude: the initial magnitude.
        magnitude_range: the sampling range for random sampling and clamping the optimized magnitude.
        temperature: temperature for RelaxedBernoulli distribution used during training.
    """

    def __init__(
        self,
        initial_magnitude: Optional[float] = 0.0,
        initial_probability: float = 0.5,
        magnitude_range: Tuple[float, float] = (-0.5, 0.5),
        temperature: float = 0.1,
        symmetric_megnitude: bool = False,
    ):
        super().__init__(
            RandomHue(magnitude_range, same_on_batch=False, p=initial_probability),
            initial_magnitude=[("hue_factor", initial_magnitude)],
            temperature=temperature,
            symmetric_megnitude=symmetric_megnitude,
        )


class Saturate(OperationBase):
    """Apply saturation operation.

    Args:
        initial_probability: the initial probability.
        initial_magnitude: the initial magnitude.
        magnitude_range: the sampling range for random sampling and clamping the optimized magnitude.
        temperature: temperature for RelaxedBernoulli distribution used during training.
    """

    def __init__(
        self,
        initial_magnitude: Optional[float] = 0.5,
        initial_probability: float = 0.5,
        magnitude_range: Tuple[float, float] = (0.2, 1.8),
        temperature: float = 0.1,
        symmetric_megnitude: bool = False,
    ):
        super().__init__(
            RandomSaturation(magnitude_range, same_on_batch=False, p=initial_probability),
            initial_magnitude=[("saturation_factor", initial_magnitude)],
            temperature=temperature,
            symmetric_megnitude=symmetric_megnitude,
        )


# TODO: Equalize cannot update probabilities yet.
class Equalize(OperationBase):
    """Apply equalize operation.

    Args:
        initial_probability: the initial probability.
        temperature: temperature for RelaxedBernoulli distribution used during training.

    Note:
        Equalize cannot update probabilities yet.

    Note:
        STE gradient estimator applied for back propagation.
    """

    def __init__(self, initial_probability: float = 0.5, temperature: float = 0.1, symmetric_megnitude: bool = False):
        super().__init__(
            RandomEqualize(same_on_batch=False, p=initial_probability),
            initial_magnitude=None,
            temperature=temperature,
            symmetric_megnitude=symmetric_megnitude,
            gradient_estimator=STEFunction,
        )


class Gray(OperationBase):
    """Apply grayscale operation.

    Args:
        initial_probability: the initial probability.
        temperature: temperature for RelaxedBernoulli distribution used during training.
    """

    def __init__(self, initial_probability: float = 0.5, temperature: float = 0.1, symmetric_megnitude: bool = False):
        super().__init__(
            RandomGrayscale(same_on_batch=False, p=initial_probability),
            initial_magnitude=None,
            temperature=temperature,
            symmetric_megnitude=symmetric_megnitude,
        )


class Invert(OperationBase):
    """Apply invert operation.

    Args:
        initial_probability: the initial probability.
        temperature: temperature for RelaxedBernoulli distribution used during training.
    """

    def __init__(self, initial_probability: float = 0.5, temperature: float = 0.1, symmetric_megnitude: bool = False):
        super().__init__(
            RandomInvert(same_on_batch=False, p=initial_probability),
            initial_magnitude=None,
            temperature=temperature,
            symmetric_megnitude=symmetric_megnitude,
        )


class Posterize(OperationBase):
    """Apply posterize operation.

    Args:
        initial_magnitude: the initial magnitude.
        initial_probability: the initial probability.
        magnitude_range: the sampling range for random sampling and clamping the optimized magnitude.
        temperature: temperature for RelaxedBernoulli distribution used during training.

    Note:
        STE gradient estimator applied for back propagation.
    """

    @staticmethod
    def _process_magnitude(magnitude: Tensor) -> Tensor:
        return magnitude.long()

    def __init__(
        self,
        initial_magnitude: Optional[float] = 4.0,
        initial_probability: float = 0.5,
        magnitude_range: Tuple[float, float] = (1, 8),
        temperature: float = 0.1,
        symmetric_megnitude: bool = False,
    ):
        super().__init__(
            RandomPosterize(magnitude_range, same_on_batch=False, p=initial_probability),
            initial_magnitude=[("bits_factor", initial_magnitude)],
            temperature=temperature,
            symmetric_megnitude=symmetric_megnitude,
            magnitude_fn=Posterize._process_magnitude,
            gradient_estimator=STEFunction,
        )


class Solarize(OperationBase):
    """Apply solarize operation.

    Args:
        initial_magnitude: the initial magnitude.
        initial_probability: the initial probability.
        magnitude_range: the sampling range for random sampling and clamping the optimized magnitude.
        temperature: temperature for RelaxedBernoulli distribution used during training.

    Note:
        STE gradient estimator applied for back propagation.
    """

    def __init__(
        self,
        initial_magnitude: Optional[float] = 0.5,
        initial_probability: float = 0.5,
        magnitude_range: Tuple[float, float] = (0.0, 1.0),
        temperature: float = 0.1,
        symmetric_megnitude: bool = False,
    ):
        super().__init__(
            RandomSolarize(magnitude_range, additions=0.0, same_on_batch=False, p=initial_probability),
            initial_magnitude=[("thresholds", initial_magnitude)],
            temperature=temperature,
            symmetric_megnitude=symmetric_megnitude,
            gradient_estimator=STEFunction,
        )


class SolarizeAdd(OperationBase):
    """Apply solarize-addition operation with a fixed thresholds of 0.5.

    Args:
        initial_magnitude: the initial magnitude.
        initial_probability: the initial probability.
        magnitude_range: the sampling range for random sampling and clamping the optimized magnitude.
        temperature: temperature for RelaxedBernoulli distribution used during training.

    Note:
        STE gradient estimator applied for back propagation.
    """

    def __init__(
        self,
        initial_magnitude: Optional[float] = 0.0,
        initial_probability: float = 0.5,
        magnitude_range: Tuple[float, float] = (-0.3, 0.3),
        temperature: float = 0.1,
        symmetric_megnitude: bool = False,
    ):
        super().__init__(
            RandomSolarize(thresholds=0.5, additions=magnitude_range, same_on_batch=False, p=initial_probability),
            initial_magnitude=[("additions", initial_magnitude)],
            temperature=temperature,
            symmetric_megnitude=symmetric_megnitude,
            gradient_estimator=STEFunction,
        )


class Sharpness(OperationBase):
    """Apply sharpness operation.

    Args:
        initial_magnitude: the initial magnitude.
        initial_probability: the initial probability.
        magnitude_range: the sampling range for random sampling and clamping the optimized magnitude.
        temperature: temperature for RelaxedBernoulli distribution used during training.
    """

    def __init__(
        self,
        initial_magnitude: Optional[float] = 0.5,
        initial_probability: float = 0.5,
        magnitude_range: Tuple[float, float] = (0.1, 1.9),
        temperature: float = 0.1,
        symmetric_megnitude: bool = False,
    ):
        super().__init__(
            RandomSharpness(magnitude_range, same_on_batch=False, p=initial_probability),
            initial_magnitude=[("sharpness", initial_magnitude)],
            temperature=temperature,
            symmetric_megnitude=symmetric_megnitude,
        )


class HorizontalFlip(OperationBase):
    """Apply horizontal flip operation.

    Args:
        initial_probability: the initial probability.
        temperature: temperature for RelaxedBernoulli distribution used during training.
    """

    def __init__(self, initial_probability: float = 0.5, temperature: float = 0.1, symmetric_megnitude: bool = False):
        super().__init__(
            RandomHorizontalFlip(same_on_batch=False, p=initial_probability),
            initial_magnitude=None,
            temperature=temperature,
            symmetric_megnitude=symmetric_megnitude,
        )


class VerticalFlip(OperationBase):
    """Apply vertical flip operation.

    Args:
        initial_magnitude: the initial magnitude.
        temperature: temperature for RelaxedBernoulli distribution used during training.
    """

    def __init__(self, initial_probability: float = 0.5, temperature: float = 0.1, symmetric_megnitude: bool = False):
        super().__init__(
            RandomVerticalFlip(same_on_batch=False, p=initial_probability),
            initial_magnitude=None,
            temperature=temperature,
            symmetric_megnitude=symmetric_megnitude,
        )


class Rotate(OperationBase):
    """Apply rotate operation.

    Args:
        initial_magnitude: the initial magnitude.
        initial_probability: the initial probability.
        magnitude_range: the sampling range for random sampling and clamping the optimized magnitude.
        temperature: temperature for RelaxedBernoulli distribution used during training.
    """

    def __init__(
        self,
        initial_magnitude: Optional[float] = 15.0,
        initial_probability: float = 0.5,
        magnitude_range: Tuple[float, float] = (0.0, 30.0),
        temperature: float = 0.1,
        symmetric_megnitude: bool = True,
    ):
        if symmetric_megnitude and magnitude_range[0] < 0:
            raise ValueError(
                f"Lower bound of {self.__class__.__name__} is a symetric operation. "
                f"The lower bound must above 0. Got {magnitude_range[0]}."
            )
        super().__init__(
            RandomRotation(magnitude_range, same_on_batch=False, p=initial_probability),
            initial_magnitude=[("degrees", initial_magnitude)],
            temperature=temperature,
            symmetric_megnitude=symmetric_megnitude,
        )


class ShearX(OperationBase):
    """Apply shear operation along x-axis.

    Args:
        initial_magnitude: the initial magnitude.
        initial_probability: the initial probability.
        magnitude_range: the sampling range for random sampling and clamping the optimized magnitude.
        temperature: temperature for RelaxedBernoulli distribution used during training.
    """

    @staticmethod
    def _process_magnitude(magnitude: Tensor) -> Tensor:
        # make it sign-agnostic
        return magnitude * 180

    def __init__(
        self,
        initial_magnitude: Optional[float] = 0.0,
        initial_probability: float = 0.5,
        magnitude_range: Tuple[float, float] = (0.0, 0.3),
        temperature: float = 0.1,
        symmetric_megnitude: bool = True,
    ):
        if symmetric_megnitude and magnitude_range[0] < 0:
            raise ValueError(
                f"Lower bound of {self.__class__.__name__} is a symetric operation. "
                f"The lower bound must above 0. Got {magnitude_range[0]}."
            )
        super().__init__(
            RandomShear(magnitude_range, same_on_batch=False, p=initial_probability),
            initial_magnitude=[("shear_x", initial_magnitude)],
            temperature=temperature,
            symmetric_megnitude=symmetric_megnitude,
            magnitude_fn=ShearX._process_magnitude,
        )


class ShearY(OperationBase):
    """Apply shear operation along y-axis.

    Args:
        initial_magnitude: the initial magnitude.
        initial_probability: the initial probability.
        magnitude_range: the sampling range for random sampling and clamping the optimized magnitude.
        temperature: temperature for RelaxedBernoulli distribution used during training.
    """

    @staticmethod
    def _process_magnitude(magnitude: Tensor) -> Tensor:
        # make it sign-agnostic
        return magnitude * 180

    def __init__(
        self,
        initial_magnitude: Optional[float] = 0.0,
        initial_probability: float = 0.5,
        magnitude_range: Tuple[float, float] = (0.0, 0.3),
        temperature: float = 0.1,
        symmetric_megnitude: bool = True,
    ):
        if symmetric_megnitude and magnitude_range[0] < 0:
            raise ValueError(
                f"Lower bound of {self.__class__.__name__} is a symetric operation. "
                f"The lower bound must above 0. Got {magnitude_range[0]}."
            )
        super().__init__(
            RandomShear((0.0, 0.0, *magnitude_range), same_on_batch=False, p=initial_probability),
            initial_magnitude=[("shear_y", initial_magnitude)],
            temperature=temperature,
            symmetric_megnitude=symmetric_megnitude,
            magnitude_fn=ShearY._process_magnitude,
        )


class TranslateX(OperationBase):
    """Apply translate operation along x-axis.

    Args:
        initial_magnitude: the initial magnitude.
        initial_probability: the initial probability.
        magnitude_range: the sampling range for random sampling and clamping the optimized magnitude.
        temperature: temperature for RelaxedBernoulli distribution used during training.
    """

    def __init__(
        self,
        initial_magnitude: Optional[float] = 0.0,
        initial_probability: float = 0.5,
        magnitude_range: Tuple[float, float] = (0.0, 0.5),
        temperature: float = 0.1,
        symmetric_megnitude: bool = True,
    ):
        if symmetric_megnitude and magnitude_range[0] < 0:
            raise ValueError(
                f"Lower bound of {self.__class__.__name__} is a symetric operation. "
                f"The lower bound must above 0. Got {magnitude_range[0]}."
            )
        super().__init__(
            RandomTranslate(magnitude_range, same_on_batch=False, p=initial_probability),
            initial_magnitude=[("translate_x", initial_magnitude)],
            temperature=temperature,
            symmetric_megnitude=symmetric_megnitude,
        )


class TranslateY(OperationBase):
    """Apply translate operation along y-axis.

    Args:
        initial_magnitude: the initial magnitude.
        initial_probability: the initial probability.
        magnitude_range: the sampling range for random sampling and clamping the optimized magnitude.
        temperature: temperature for RelaxedBernoulli distribution used during training.
    """

    def __init__(
        self,
        initial_magnitude: Optional[float] = 0.0,
        initial_probability: float = 0.5,
        magnitude_range: Tuple[float, float] = (0.0, 0.5),
        temperature: float = 0.1,
        symmetric_megnitude: bool = True,
    ):
        if symmetric_megnitude and magnitude_range[0] < 0:
            raise ValueError(
                f"Lower bound of {self.__class__.__name__} is a symetric operation. "
                f"The lower bound must above 0. Got {magnitude_range[0]}."
            )
        super().__init__(
            RandomTranslate(magnitude_range, same_on_batch=False, p=initial_probability),
            initial_magnitude=[("translate_y", initial_magnitude)],
            temperature=temperature,
            symmetric_megnitude=symmetric_megnitude,
        )
