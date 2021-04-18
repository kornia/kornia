from typing import Tuple

import torch
from torch.autograd import Function


# https://github.com/moskomule/dda/blob/3ffe718e253a77ecb8b4e638d851f0d3d248c111/dda/functional.py#L56
class StraightThroughEstimator(Function):
    """ StraightThrough Estimator
    """
    @staticmethod
    def forward(ctx,
                input_forward: torch.Tensor,
                input_backward: torch.Tensor) -> torch.Tensor:
        ctx.shape = input_backward.shape
        return input_forward

    @staticmethod
    def backward(ctx,
                 grad_in: torch.Tensor) -> Tuple[None, torch.Tensor]:
        return None, grad_in.sum_to_size(ctx.shape)


# def ste(input_forward: torch.Tensor,
#         input_backward: torch.Tensor) -> torch.Tensor:
#     """ Straight-through estimator
#     :param input_forward:
#     :param input_backward:
#     :return:
#     """
#     return StraightThroughEstimator.apply(input_forward, input_backward).clone()
