from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function


class STEFunction(Function):
    """StraightThrough Estimator
    """
    @staticmethod
    def forward(ctx, input: torch.Tensor, output: torch.Tensor) -> torch.Tensor:
        ctx.shape = output.shape
        return output

    @staticmethod
    def backward(ctx, grad_in: torch.Tensor) -> Tuple[None, torch.Tensor]:
        return None, F.hardtanh(grad_in.sum_to_size(ctx.shape))  # Avoid gradients exploding


class StraightThroughEstimator(nn.Module):
    def __init__(self):
        super(StraightThroughEstimator, self).__init__()

    def forward(self, input: torch.Tensor, output: torch.Tensor) -> torch.Tensor:
        output = STEFunction.apply(input, output)
        return output
