from typing import Optional, Callable, Dict, Union, Tuple, List, cast

import torch
import torch.nn as nn
from torch.autograd import Function

from kornia.enhance import (
    equalize,
    adjust_brightness
)
from kornia.augmentation.core.sampling import (
    DynamicSampling,
)
from kornia.augmentation.core.gradient_estimator import (
    STEFunction
)
from .base import IntensityAugmentOperation


class EqualizeAugment(IntensityAugmentOperation):
    """Perform equalization augmentation.

    Examples:
        >>> a = EqualizeAugment(1.)
        >>> out = a(torch.ones(2, 3, 100, 100) * 0.5)
        >>> out.shape
        torch.Size([2, 3, 100, 100])

        # Backprop with gradients estimator
        >>> inp = torch.ones(2, 3, 100, 100, requires_grad=True) * 0.5
        >>> a = EqualizeAugment(1., gradients_estimator=STEFunction)
        >>> out = a(inp)
        >>> loss = (out - torch.ones(2, 3, 100, 100)).mean()
        >>> loss.backward()
        >>> inp.grad
    """
    def __init__(
        self, p: float = 0.5, same_on_batch: bool = False,
        gradients_estimator: Optional[Function] = STEFunction  # type:ignore
        # Note: Weird that the inheritance typing is not working for Function
    ):
        super().__init__(
            torch.tensor(p), torch.tensor(1.), sampler=None, mapper=None, gradients_estimator=gradients_estimator,
            same_on_batch=same_on_batch
        )

    def apply_transform(self, input: torch.Tensor, _: List[torch.Tensor]) -> torch.Tensor:
        return equalize(input)


class BrightnessAugment(IntensityAugmentOperation):
    """Perform brightness adjustment augmentation.

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
        sampler: Union[Tuple[float, float], DynamicSampling] = (0.3, 0.7),
        mapper: Optional[Callable] = None, p: float = 0.5,
        same_on_batch: bool = False, mode: str = 'bilinear', align_corners: bool = True,
        gradients_estimator: Optional[Function] = None
    ):
        super().__init__(
            torch.tensor(p), torch.tensor(1.), sampler=[sampler], mapper=None if mapper is None else [mapper],
            gradients_estimator=gradients_estimator, same_on_batch=same_on_batch
        )
        self.mode = mode
        self.align_corners = align_corners

    def apply_transform(self, input: torch.Tensor, magnitude: List[torch.Tensor]) -> torch.Tensor:
        return adjust_brightness(input, magnitude[0])
