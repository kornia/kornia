from typing import Optional, Callable, Dict, Union, Tuple, List, cast

import torch
import torch.nn as nn
from torch.autograd import Function

from kornia.enhance import (
    equalize
)
from .smart_sampling import (
    SmartSampling,
    SmartUniform,
)
from .gradient_estimator import (
    STEFunction
)
from .operation_base import IntensityAugmentOperation


class Equalize(IntensityAugmentOperation):
    """
    >>> a = Equalize(1.)
    >>> out = a(torch.ones(2, 3, 100, 100) * 0.5)
    >>> out.shape
    torch.Size([2, 3, 100, 100])

    # Backprop with gradients estimator
    >>> inp = torch.ones(2, 3, 100, 100, requires_grad=True) * 0.5
    >>> a = Equalize(1., gradients_estimator=STEFunction)
    >>> out = a(inp)
    >>> loss = (out - torch.ones(2, 3, 100, 100)).mean()
    >>> loss.backward()
    >>> inp.grad
    """
    def __init__(
        self, p: float = 0.5, same_on_batch: bool = False,
        gradients_estimator: Optional[Function] = STEFunction
    ):
        super().__init__(
            torch.tensor(p), torch.tensor(1.), sampler=None, mapper=None, gradients_estimator=gradients_estimator,
            same_on_batch=same_on_batch
        )

    def apply_transform(self, input: torch.Tensor, _: Optional[torch.Tensor]) -> torch.Tensor:
        return equalize(input)
