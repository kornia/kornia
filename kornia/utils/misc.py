import torch


def eye_like(n: int, input: torch.Tensor) -> torch.Tensor:
    r"""Return a 2-D tensor with ones on the diagonal and zeros elsewhere with the same batch size as the input.

    Args:
        n: the number of rows :math:`(N)`.
        input: image tensor that will determine the batch size of the output matrix.
          The expected shape is :math:`(B, *)`.

    Returns:
       The identity matrix with the same batch size as the input :math:`(B, N, N)`.
    """
    if n <= 0:
        raise AssertionError(type(n), n)
    if len(input.shape) < 1:
        raise AssertionError(input.shape)

    identity = torch.eye(n, device=input.device, dtype=input.dtype)
    return identity[None].repeat(input.shape[0], 1, 1)


def vec_like(n, tensor):
    r"""Return a 2-D tensor with a vector containing zeros with the same batch size as the input.

    Args:
        n: the number of rows :math:`(N)`.
        tensor: image tensor that will determine the batch size of the output matrix.
          The expected shape is :math:`(B, *)`.

    Returns:
        The vector with the same batch size as the input :math:`(B, N, 1)`.
    """
    if n <= 0:
        raise AssertionError(type(n), n)
    if len(tensor.shape) < 1:
        raise AssertionError(tensor.shape)

    vec = torch.zeros(n, 1, device=tensor.device, dtype=tensor.dtype)
    return vec[None].repeat(tensor.shape[0], 1, 1)


def zeros_like(input: torch.Tensor) ->torch.Tensor:
    r"""Returns a tensor filled with the scalar value 0, with the same size as input

    Args:
        input: input tensor size will determine the size of the output tensor.

    Returns:
        A tensor filled with value 0, with the same size as input
    """
    if len(input.shape) < 1:
        raise AssertionError(input.shape)

    return torch.zeros_like(input)
