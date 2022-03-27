from typing import Callable, Optional, Tuple

import torch.nn as nn
from torch import Tensor
from torch.autograd import Function

__all__ = [
    "STEFunction",
    "StraightThroughEstimator"
]


class STEFunction(Function):
    """Straight-Through Estimation (STE) function.

    STE bridges the gradients between the input tensor and output tensor as if the function
    was an identity function. Meanwhile, advanced gradient functions are also supported. e.g.
    the output gradients can be mapped into [-1, 1] with ``F.hardtanh`` function.

    Args:
        grad_fn: function to restrain the gradient recieved. If None, no mapping will performed.

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
        >>> out_est = STEFunction.apply(input, output)
        >>> loss = out_est.mean()
        >>> loss.backward()
        >>> input.grad
        tensor([0.2500, 0.2500, 0.2500, 0.2500])
    """

    @staticmethod
    def forward(
        ctx, input: Tensor, output: Tensor,
        grad_fn: Optional[Callable] = None
    ) -> Tensor:
        ctx.in_shape = input.shape
        ctx.out_shape = output.shape
        ctx.grad_fn = grad_fn
        return output

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> Tuple[Tensor, Tensor, None]:
        if ctx.grad_fn is None:
            return grad_output.sum_to_size(ctx.in_shape), grad_output.sum_to_size(ctx.out_shape), None
        return (
            ctx.grad_fn(grad_output.sum_to_size(ctx.in_shape)),
            ctx.grad_fn(grad_output.sum_to_size(ctx.out_shape)),
            None
        )


class StraightThroughEstimator(nn.Module):
    """Straight-Through Estimation (STE) module.

    STE wraps the ``STEFunction`` to aid the back propagation of non-differentiable modules.
    It may also use to avoid gradient computation for differentiable operations. By default,
    STE bridges the gradients between the input tensor and output tensor as if the function
    was an identity function. Meanwhile, advanced gradient functions are also supported. e.g.
    the output gradients can be mapped into [-1, 1] with ``F.hardtanh`` function.

    Args:
        target_fn: the target function to wrap with.
        grad_fn: function to restrain the gradient recieved. If None, no mapping will performed.

    Example:
        ``RandomPosterize`` is a non-differentiable operation. Let STE estimate the gradients.
        >>> import kornia.augmentation as K
        >>> import torch.nn.functional as F
        >>> input = torch.randn(1, 1, 4, 4, requires_grad = True)
        >>> estimator = StraightThroughEstimator(K.RandomPosterize(3, p=1.), grad_fn=F.hardtanh)
        >>> out = estimator(input)
        >>> loss = out.mean()
        >>> loss.backward()
        >>> input.grad
        tensor([[[[0.0625, 0.0625, 0.0625, 0.0625],
                  [0.0625, 0.0625, 0.0625, 0.0625],
                  [0.0625, 0.0625, 0.0625, 0.0625],
                  [0.0625, 0.0625, 0.0625, 0.0625]]]])
    """

    def __init__(self, target_fn: nn.Module, grad_fn: Optional[Callable] = None):
        super().__init__()
        self.target_fn = target_fn
        self.grad_fn = grad_fn

    def __repr__(self):
        return f"{self.__class__.__name__}(target_fn={self.target_fn}, grad_fn={self.grad_fn})"

    def forward(self, input: Tensor) -> Tensor:
        out = self.target_fn(input)
        if not isinstance(out, Tensor):
            raise NotImplementedError(
                "Only Tensor is supported at the moment. Feel free to contribute to https://github.com/kornia/kornia.")
        output = STEFunction.apply(input, out, self.grad_fn)
        return output
