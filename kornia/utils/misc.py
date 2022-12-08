from typing import Tuple, Union

import torch
from torch import Size, Tensor


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

    identity = torch.eye(n, device=input.device, dtype=input.dtype)
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

    vec = torch.zeros(n, 1, device=tensor.device, dtype=tensor.dtype)
    return vec[None].expand(tensor.shape[0], n, 1) if shared_memory else vec[None].repeat(tensor.shape[0], 1, 1)


def reduce_first_dims(x: Tensor, keep_last_dims: int, return_shape: bool = True) -> Union[Tensor, Tuple[Tensor, Size]]:
    """View of a tensor by keeping the last N dims the same, and squeezing all prior dims to a single leading dim.

    Args:
        x: Tensor.
        keep_last_dims: number of last N dims to keep unchanged.
        return_shape: whether or not to return the tensor's original shape.

    Returns:
        A view of the input tensor, with (keep_last_dims + 1) dims:
            keep_last_dims are kept the same, all prior dims are squeezed to the new first dim.
        In addition, the method returns the original shape of the input tensor.

    Example:
        >>> x = torch.rand(2, 4, 8, 16, 32)
        >>> y, shape = reduce_first_dims(x, keep_last_dims=3, return_shape=True)
        >>> z = y.view(shape)  # restoring the original shape
        >>> assert torch.equal(x, z)
        >>> y.shape
        torch.Size([8, 8, 16, 32])
    """
    shape = x.shape
    x_view = x.view(-1, *shape[-keep_last_dims:])
    if return_shape:
        return x_view, shape
    return x_view
