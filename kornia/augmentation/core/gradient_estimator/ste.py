from typing import Callable, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

__all__ = ["STEFunction", "StraightThroughEstimator"]


def _identity(input: torch.Tensor, shape: Optional[torch.Size]) -> torch.Tensor:
    return input


def _resize(input: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    align_corners: Optional[bool]
    if len(input.shape) == 5:
        mode = "trilinear"
        align_corners = True
    elif len(input.shape) == 4:
        mode = "bilinear"
        align_corners = True
    elif len(input.shape) == 3:
        mode = "linear"
        align_corners = True
    else:
        mode = "nearest"  # TODO: nearest is a non-differentiable mode, need to warn the user.
        align_corners = None
    return F.interpolate(input, shape[2:], mode=mode, align_corners=align_corners)


class STEFunction(Function):
    """Straight-Through Estimation (STE) function.

    STE bridges the gradients between the input tensor and output tensor as if the function
    was an identity function. For geometric augmentations, customized straight-through sampling
    can be defined to pass through only relevant parts.

    Args:
        input (torch.Tensor): input tensor.
        output (torch.Tensor): output tensor.
        transform_fn (callable, optional): if the output tensor is geometrically transformed from
            the input tensor, the inversed geometric transform is allowed to be applied on
            the computed output gradients in order to restore the input gradient. Any ``transform_fn``
            takes a gradient tensor and the input shape as parameters. By default, a bilinear
            interpolation (only support 3D, 4D, 5D tensors) will be applied.
        grad_fn(callable, optional): a differentiable function to control the grad output values.
            Normally, it is used to restrain the gradients to avoid gradients vanishing or exploding.
            By default, the output gradients will be mapped into [-1, 1] with ``F.hardtanh`` function.
            If None, an identity mapping will be applied.
    """

    @staticmethod
    def forward(  # type: ignore
        ctx,
        input: torch.Tensor,
        output: torch.Tensor,
        transform_fn: Optional[Callable] = None,
        grad_fn: Optional[Callable] = F.hardtanh,
    ) -> torch.Tensor:
        ctx.shape = input.shape
        if input.shape != output.shape:
            default_transform_fn = _resize
        else:
            default_transform_fn = _identity
        ctx.transform_fn = transform_fn if transform_fn is not None else default_transform_fn
        ctx.grad_fn = grad_fn if grad_fn is not None else lambda x: _identity(x, None)
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, None, None]:  # type: ignore
        return (ctx.grad_fn(ctx.transform_fn(grad_output, ctx.shape)), ctx.grad_fn(grad_output), None, None)


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
        >>> out_est = StraightThroughEstimator()(input, output)
        >>> loss = out_est.mean()
        >>> loss.backward()
        >>> input.grad
        tensor([0.2500, 0.2500, 0.2500, 0.2500])

        Perform geometric gradients estimation from STE.
        >>> input = torch.randn(1, 1, 4, requires_grad = True)
        >>> output = input[..., 2:]
        >>> out_est = StraightThroughEstimator()(input, output)
        >>> loss = out_est.mean()
        >>> loss.backward()
        >>> input.grad
        tensor([[[0.5000, 0.5000, 1.0000, 1.0000]]])
    """

    def __init__(self, grad_fn: Optional[Callable] = F.hardtanh):
        super(StraightThroughEstimator, self).__init__()
        self.grad_fn = grad_fn

    def forward(
        self, input: torch.Tensor, output: torch.Tensor, transform_fn: Optional[Callable] = None
    ) -> torch.Tensor:
        output = STEFunction.apply(input, output, transform_fn, self.grad_fn)
        return output
