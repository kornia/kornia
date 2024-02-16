from typing import Optional

import torch

from kornia.core import Tensor, eye, zeros


def eye_like(n: int, input: Tensor, shared_memory: bool = False) -> Tensor:
    r"""Return a 2-D tensor with ones on the diagonal and zeros elsewhere with the same batch size as the input.

    Args:
        n: the number of rows :math:`(N)`.
        input: image tensor that will determine the batch size of the output matrix.
          The expected shape is :math:`(B, *)`.
        shared_memory: when set, all samples in the batch will share the same memory.

    Returns:
       The identity matrix with the same batch size as the input :math:`(B, N, N)`.

    Notes:
        When the dimension to expand is of size 1, using torch.expand(...) yields the same tensor as torch.repeat(...)
        without using extra memory. Thus, when the tensor obtained by this method will be later assigned -
        use this method with shared_memory=False, otherwise, prefer using it with shared_memory=True.
    """
    if n <= 0:
        raise AssertionError(type(n), n)
    if len(input.shape) < 1:
        raise AssertionError(input.shape)

    identity = eye(n, device=input.device).type(input.dtype)

    return identity[None].expand(input.shape[0], n, n) if shared_memory else identity[None].repeat(input.shape[0], 1, 1)


def vec_like(n: int, tensor: Tensor, shared_memory: bool = False) -> Tensor:
    r"""Return a 2-D tensor with a vector containing zeros with the same batch size as the input.

    Args:
        n: the number of rows :math:`(N)`.
        tensor: image tensor that will determine the batch size of the output matrix.
          The expected shape is :math:`(B, *)`.
        shared_memory: when set, all samples in the batch will share the same memory.

    Returns:
        The vector with the same batch size as the input :math:`(B, N, 1)`.

    Notes:
        When the dimension to expand is of size 1, using torch.expand(...) yields the same tensor as torch.repeat(...)
        without using extra memory. Thus, when the tensor obtained by this method will be later assigned -
        use this method with shared_memory=False, otherwise, prefer using it with shared_memory=True.
    """
    if n <= 0:
        raise AssertionError(type(n), n)
    if len(tensor.shape) < 1:
        raise AssertionError(tensor.shape)

    vec = zeros(n, 1, device=tensor.device, dtype=tensor.dtype)
    return vec[None].expand(tensor.shape[0], n, 1) if shared_memory else vec[None].repeat(tensor.shape[0], 1, 1)


def differentiable_polynomial_rounding(input: Tensor) -> Tensor:
    """This function implements differentiable rounding.

    Args:
        input (Tensor): Input tensor of any shape to be rounded.

    Returns:
        output (Tensor): Pseudo rounded tensor of the same shape as input tensor.
    """
    # Perform differentiable rounding
    input_round = input.round()
    output: Tensor = input_round + (input - input_round) ** 3
    return output


def differentiable_polynomial_floor(input: Tensor) -> Tensor:
    """This function implements differentiable floor.

    Args:
        input (Tensor): Input tensor of any shape to be floored.

    Returns:
        output (Tensor): Pseudo rounded tensor of the same shape as input tensor.
    """
    # Perform differentiable rounding
    input_floor = input.floor()
    output: Tensor = input_floor + (input - 0.5 - input_floor) ** 3
    return output


def differentiable_clipping(
    input: Tensor,
    min_val: Optional[float] = None,
    max_val: Optional[float] = None,
    scale: float = 0.02,
) -> Tensor:
    """This function implements a differentiable and soft approximation of the clipping operation.

    Args:
        input (Tensor): Input tensor of any shape.
        min_val (Optional[float]): Minimum value.
        max_val (Optional[float]): Maximum value.
        scale (float): Scale value. Default 0.02.

    Returns:
        output (Tensor): Clipped output tensor of the same shape as the input tensor.
    """
    # Make a copy of the input tensor
    output: Tensor = input.clone()
    # Perform differentiable soft clipping
    if max_val is not None:
        output[output > max_val] = -scale * (torch.exp(-output[output > max_val] + max_val) - 1.0) + max_val
    if min_val is not None:
        output[output < min_val] = scale * (torch.exp(output[output < min_val] - min_val) - 1.0) + min_val
    return output
