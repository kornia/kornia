from typing import Tuple, Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function


def _identity(input: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    return input


class STEFunction(Function):
    """Straight-Through Estimation (STE) function.

    STE bridges the gradients between the input tensor and output tensor as if the function
    was an identity function. By default, the output gradients will be mapped into [-1, 1]
    with ``F.hardtanh`` function.
    """
    @staticmethod
    def forward(
        ctx, input: torch.Tensor, output: torch.Tensor, transform_fn: Optional[Callable] = None,
        grad_fn: Optional[Callable] = F.hardtanh
    ) -> torch.Tensor:
        if input.shape != output.shape:
            assert transform_fn is not None, \
                f"Expect a tranformation function for different shaped input {input.shape} and output {output.shape}."
        ctx.shape = input.shape
        ctx.transform_fn = transform_fn if transform_fn is not None else _identity
        ctx.grad_fn = grad_fn if grad_fn is not None else _identity
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, None, None]:
        return (
            ctx.grad_fn(ctx.transform_fn(grad_output, ctx.shape)),
            ctx.grad_fn(grad_output),
            None,
            None
        )


class StraightThroughEstimator(nn.Module):
    """Straight-Through Estimation (STE) function.

    STE bridges the gradients between the input tensor and output tensor as if the function
    was an identity function.

    Args:
        grad_fn (callable, optional): function to restrain the gradient recieved.
            By default, the output gradients will be mapped into [-1, 1] with ``F.hardtanh`` function.

    Example:
        Let the gradients of ``torch.sign`` estimated from STE.
        >>> input = torch.randn(4, requires_grad = True)
        >>> output = torch.sign(input)
        >>> loss = output.mean()
        >>> loss.backward()
        >>> input.grad
        tensor([0., 0., 0., 0.])

        >>> with torch.no_grad():
        ...     output = torch.sign(input)
        >>> out_est = StraightThroughEstimator(None)(input, output)
        >>> loss = out_est.mean()
        >>> loss.backward()
        >>> input.grad
        tensor([0.2500, 0.2500, 0.2500, 0.2500])
    """

    def __init__(self, grad_fn: Optional[Callable] = F.hardtanh):
        super(StraightThroughEstimator, self).__init__()
        self.grad_fn = grad_fn

    def forward(
        self, input: torch.Tensor, output: torch.Tensor, transform_fn: Optional[Callable] = None
    ) -> torch.Tensor:
        output = STEFunction.apply(input, output, transform_fn, self.grad_fn)
        return output
