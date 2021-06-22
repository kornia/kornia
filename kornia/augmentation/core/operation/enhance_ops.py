from typing import Callable, List, Optional, Tuple, Union
from functools import partial

import torch
from torch.autograd import Function

from kornia.augmentation.core.gradient_estimator import STEFunction
from kornia.augmentation.core.sampling import DynamicSampling
from kornia.enhance import (
    adjust_brightness,
    adjust_contrast,
    adjust_hue,
    adjust_saturation,
    equalize,
)
from kornia.constants import pi
from torch.autograd.grad_mode import F

from .base import IntensityAugmentOperation, Parameters

__all__ = [
    "BrightnessAugment",
    "ColorJitter",
    "ContrastAugment",
    "EqualizeAugment",
    "HueAugment",
    "SaturationAugment",
]


class EqualizeAugment(IntensityAugmentOperation):
    """Perform equalization augmentation.

    Args:
        gradient_estimator(Function, optional): gradient estimator for this operation.
            Default is Straight-through estimator.
        p (float): probability of the image being flipped. Default value is 0.5
        same_on_batch (bool): apply the same transformation across the batch. Default: False.

    Examples:
        >>> a = EqualizeAugment(p=1.)
        >>> out = a(torch.ones(2, 3, 100, 100) * 0.5)
        >>> out.shape
        torch.Size([2, 3, 100, 100])

        # Backprop with gradients estimator
        >>> inp = torch.ones(2, 3, 100, 100, requires_grad=True) * 0.5
        >>> a = EqualizeAugment(p=1., gradient_estimator=STEFunction)
        >>> out = a(inp)
        >>> loss = (out - torch.ones(2, 3, 100, 100)).mean()
        >>> loss.backward()
        >>> inp.grad
    """

    def __init__(
        self,
        # Note: Weird that the inheritance typing is not working for Function
        gradient_estimator: Optional[Function] = STEFunction,  # type:ignore
        p: float = 0.5,
        same_on_batch: bool = False,
    ):
        super().__init__(
            torch.tensor(p),
            torch.tensor(1.0),
            sampler_list=[],
            gradient_estimator=gradient_estimator,
            same_on_batch=same_on_batch,
        )

    def apply_transform(self, input: torch.Tensor, _: List[torch.Tensor]) -> torch.Tensor:
        return equalize(input)


class BrightnessAugment(IntensityAugmentOperation):
    """Perform brightness adjustment augmentation.

    Args:
        sampler (List[Union[Tuple[float, float], DynamicSampling]]): sampler for sampling brightness
            parameter to perform the transformation. If a tuple (a, b), it will sample from (a, b) uniformly.
            Otherwise, it will sample from the pointed sampling distribution. Default is (-0.7, 0.7).
        gradient_estimator(Function, optional): gradient estimator for this operation. Default is None.
        p (float): probability of the image being flipped. Default value is 0.5
        same_on_batch (bool): apply the same transformation across the batch. Default: False.

    Examples:
        >>> a = BrightnessAugment(p=1.)
        >>> out = a(torch.ones(2, 3, 100, 100, requires_grad=True) * 0.5)
        >>> out.shape
        torch.Size([2, 3, 100, 100])
        >>> loss = out.mean()
        >>> loss.backward()
    """

    def __init__(
        self,
        sampler: Union[Tuple[float, float], DynamicSampling] = (-0.7, 0.7),
        gradient_estimator: Optional[Function] = None,
        p: float = 0.5,
        same_on_batch: bool = False,
    ):
        super().__init__(
            torch.tensor(p),
            torch.tensor(1.0),
            sampler_list=[sampler],
            gradient_estimator=gradient_estimator,
            same_on_batch=same_on_batch,
        )

    def apply_transform(self, input: torch.Tensor, magnitude: List[torch.Tensor]) -> torch.Tensor:
        return adjust_brightness(input, magnitude[0])


class ContrastAugment(IntensityAugmentOperation):
    """Perform contrast adjustment augmentation.

    Args:
        sampler (List[Union[Tuple[float, float], DynamicSampling]]): sampler for sampling contrast
            parameter to perform the transformation. If a tuple (a, b), it will sample from (a, b) uniformly.
            Otherwise, it will sample from the pointed sampling distribution. Default is (0.3, 1.7).
        gradient_estimator(Function, optional): gradient estimator for this operation. Default is None.
        p (float): probability of the image being flipped. Default value is 0.5
        same_on_batch (bool): apply the same transformation across the batch. Default: False.

    Examples:
        >>> a = ContrastAugment(p=1.)
        >>> out = a(torch.ones(2, 3, 100, 100, requires_grad=True) * 0.5)
        >>> out.shape
        torch.Size([2, 3, 100, 100])
        >>> loss = out.mean()
        >>> loss.backward()
    """

    def __init__(
        self,
        sampler: Union[Tuple[float, float], DynamicSampling] = (0.3, 1.7),
        gradient_estimator: Optional[Function] = None,
        p: float = 0.5,
        same_on_batch: bool = False,
    ):
        super().__init__(
            torch.tensor(p),
            torch.tensor(1.0),
            sampler_list=[sampler],
            gradient_estimator=gradient_estimator,
            same_on_batch=same_on_batch,
        )

    def apply_transform(self, input: torch.Tensor, magnitude: List[torch.Tensor]) -> torch.Tensor:
        return adjust_contrast(input, magnitude[0])


class SaturationAugment(IntensityAugmentOperation):
    """Perform saturation adjustment augmentation.

    Args:
        sampler (List[Union[Tuple[float, float], DynamicSampling]]): sampler for sampling saturation
            parameter to perform the transformation. If a tuple (a, b), it will sample from (a, b) uniformly.
            Otherwise, it will sample from the pointed sampling distribution. Default is (0.3, 1.7).
        mapper(Union[Tuple[float, float], Callable]], Optional): the mapping function to map the sampled saturation
            parameter to any range. If a tuple (a, b), it will map to (a, b) by `torch.clamp` by default, in which
            ``a`` and ``b`` can be None to indicate infinity. Otherwise, it will by mapped by the provided function.
            Default is (0., None).
        gradient_estimator(Function, optional): gradient estimator for this operation. Default is None.
        p (float): probability of the image being flipped. Default value is 0.5
        same_on_batch (bool): apply the same transformation across the batch. Default: False.

    Examples:
        >>> a = SaturationAugment(p=1.)
        >>> out = a(torch.ones(2, 3, 100, 100, requires_grad=True) * 0.5)
        >>> out.shape
        torch.Size([2, 3, 100, 100])
        >>> loss = out.mean()
        >>> loss.backward()
    """

    def __init__(
        self,
        sampler: Union[Tuple[float, float], DynamicSampling] = (0.3, 1.7),
        gradient_estimator: Optional[Function] = None,
        p: float = 0.5,
        same_on_batch: bool = False,
    ):
        super().__init__(
            torch.tensor(p),
            torch.tensor(1.0),
            sampler_list=[sampler],
            gradient_estimator=gradient_estimator,
            same_on_batch=same_on_batch,
        )

    def apply_transform(self, input: torch.Tensor, magnitude: List[torch.Tensor]) -> torch.Tensor:
        return adjust_saturation(input, magnitude[0])


class HueAugment(IntensityAugmentOperation):
    """Perform hue adjustment augmentation.

    Args:
        sampler (List[Union[Tuple[float, float], DynamicSampling]]): sampler for sampling hue
            parameter to perform the transformation. If a tuple (a, b), it will sample from (a, b) uniformly.
            Otherwise, it will sample from the pointed sampling distribution. Default is (-0.3, 0.3).
        gradient_estimator(Function, optional): gradient estimator for this operation. Default is None.
        p (float): probability of the image being flipped. Default value is 0.5
        same_on_batch (bool): apply the same transformation across the batch. Default: False.

    Examples:
        >>> a = HueAugment(p=1.)
        >>> out = a(torch.ones(2, 3, 100, 100, requires_grad=True) * 0.5)
        >>> out.shape
        torch.Size([2, 3, 100, 100])
        >>> loss = out.mean()
        >>> loss.backward()
    """

    def __init__(
        self,
        sampler: Union[Tuple[float, float], DynamicSampling] = (-0.3, 0.3),
        gradient_estimator: Optional[Function] = None,
        p: float = 0.5,
        same_on_batch: bool = False,
    ):
        super().__init__(
            torch.tensor(p),
            torch.tensor(1.0),
            sampler_list=[sampler],
            gradient_estimator=gradient_estimator,
            same_on_batch=same_on_batch,
        )

    def apply_transform(self, input: torch.Tensor, magnitude: List[torch.Tensor]) -> torch.Tensor:
        return adjust_hue(input, magnitude[0] * 2 * pi)


class ColorJitter(IntensityAugmentOperation):
    r"""Applies a random transformation to the brightness, contrast, saturation and hue of a tensor image.

    Args:
        brightness_sampler (List[Union[Tuple[float, float], DynamicSampling]]): sampler for sampling brightness
            parameter to perform the transformation. If a tuple (a, b), it will sample from (a, b) uniformly.
            Otherwise, it will sample from the pointed sampling distribution. Default is (-0., 0.).
        contrast_sampler (List[Union[Tuple[float, float], DynamicSampling]]): sampler for sampling contrast
            parameter to perform the transformation. If a tuple (a, b), it will sample from (a, b) uniformly.
            Otherwise, it will sample from the pointed sampling distribution. Default is (-0., 0.).
        saturation_sampler (List[Union[Tuple[float, float], DynamicSampling]]): sampler for sampling saturation
            parameter to perform the transformation. If a tuple (a, b), it will sample from (a, b) uniformly.
            Otherwise, it will sample from the pointed sampling distribution. Default is (-0., 0.).
        hue_sampler (List[Union[Tuple[float, float], DynamicSampling]]): sampler for sampling hue
            parameter to perform the transformation. If a tuple (a, b), it will sample from (a, b) uniformly.
            Otherwise, it will sample from the pointed sampling distribution. Default is (-0., 0.).
        gradient_estimator(Function, optional): gradient estimator for this operation. Default is None.
        p (float): probability of applying the transformation. Default value is 1.
        same_on_batch (bool): apply the same transformation across the batch. Default: False.

    Shape:
        - Input: :math:`(C, H, W)` or :math:`(B, C, H, W)`, Optional: :math:`(B, 3, 3)`
        - Output: :math:`(B, C, H, W)`

    Note:
        Input tensor must be float and normalized into [0, 1] for the best differentiability support.
        Additionally, this function accepts another transformation tensor (:math:`(B, 3, 3)`), then the
        applied transformation will be merged int to the input transformation tensor and returned.

    Examples:
        >>> rng = torch.manual_seed(0)
        >>> inputs = torch.ones(1, 3, 3, 3)
        >>> aug = ColorJitter((0.9, 1.1), (0.9, 1.1), (0.9, 1.1), (-0.1, 0.1), p=1.)
        >>> aug(inputs)
        tensor([[[[0.9264, 0.9264, 0.9264],
                  [0.9264, 0.9264, 0.9264],
                  [0.9264, 0.9264, 0.9264]],
        <BLANKLINE>
                 [[0.9264, 0.9264, 0.9264],
                  [0.9264, 0.9264, 0.9264],
                  [0.9264, 0.9264, 0.9264]],
        <BLANKLINE>
                 [[0.9264, 0.9264, 0.9264],
                  [0.9264, 0.9264, 0.9264],
                  [0.9264, 0.9264, 0.9264]]]])
    """

    def __init__(
        self,
        brightness_sampler: Union[Tuple[float, float], DynamicSampling] = (1., 1.),
        contrast_sampler: Union[Tuple[float, float], DynamicSampling] = (0., 0.),
        saturation_sampler: Union[Tuple[float, float], DynamicSampling] = (0., 0.),
        hue_sampler: Union[Tuple[float, float], DynamicSampling] = (0., 0.),
        gradient_estimator: Optional[Function] = None,
        p: float = 0.5,
        same_on_batch: bool = False,
    ) -> None:
        super().__init__(
            torch.tensor(p),
            torch.tensor(1.0),
            sampler_list=[brightness_sampler, contrast_sampler, saturation_sampler, hue_sampler],
            gradient_estimator=gradient_estimator,
            same_on_batch=same_on_batch,
        )

    def generate_parameters(self, input: torch.Tensor) -> Parameters:
        p = super().generate_parameters(input)
        p.magnitudes.append(
            torch.randperm(4, device=input.device, dtype=input.dtype)[None].repeat(input.shape[0], 1).long()
        )
        return p

    def apply_transform(self, input: torch.Tensor, magnitude: List[torch.Tensor]) -> torch.Tensor:
        # transforms = [
        #     partial(adjust_brightness, brightness_factor=magnitude[0] - 1),
        #     partial(adjust_contrast, contrast_factor=magnitude[1]),
        #     partial(adjust_saturation, saturation_factor=magnitude[2]),
        #     partial(adjust_hue, magnitude[3] * 2 * pi),
        # ]
        transforms = [
            lambda img: adjust_brightness(img, magnitude[0]),
            lambda img: adjust_contrast(img, magnitude[1]),
            lambda img: adjust_saturation(img, magnitude[2]),
            lambda img: adjust_hue(img, magnitude[3] * 2 * pi),
        ]

        jittered = input
        for idx in magnitude[-1][0].tolist():
            t = transforms[idx]
            jittered = t(jittered)
        return jittered
