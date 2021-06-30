"""Module containing numerical functionalities for SfM"""

import torch

# TODO: this should go to `kornia.geometry.linalg`


def cross_product_matrix(x: torch.Tensor) -> torch.Tensor:
    r"""Returns the cross_product_matrix symmetric matrix of a vector.

    Args:
        x: The input vector to construct the matrix in the shape :math:`(B, 3)`.

    Returns:
        The constructed cross_product_matrix symmetric matrix with shape :math:`(B, 3, 3)`.

    """
    assert len(x.shape) == 2 and x.shape[1] == 3, x.shape
    # get vector compononens
    x0 = x[..., 0]
    x1 = x[..., 1]
    x2 = x[..., 2]

    # construct the matrix, reshape to 3x3 and return
    zeros = torch.zeros_like(x0)
    cross_product_matrix_flat = torch.stack([zeros, -x2, x1, x2, zeros, -x0, -x1, x0, zeros], dim=-1)
    return cross_product_matrix_flat.view(-1, 3, 3)


def eye_like(n: int, input: torch.Tensor) -> torch.Tensor:
    r"""Returns a 2-D tensor with ones on the diagonal and zeros elsewhere with same size as the input.

    Args:
        n: the number of rows :math:`(N)`.
        input: image tensor that will determine the batch size of the output matrix.
          The expected shape is :math:`(B, *)`.

    Returns:
       The identity matrix with same size as input :math:`(*, N, N)`.

    """
    assert n > 0, (type(n), n)
    assert len(input.shape) >= 1, input.shape

    identity = torch.eye(n, device=input.device, dtype=input.dtype)
    return identity[None].repeat(input.shape[0], 1, 1)


def vec_like(n, tensor):
    r"""Returns a 2-D tensor with a vector containing zeros with same size as the input.

    Args:
        n: the number of rows :math:`(N)`.
        input: image tensor that will determine the batch size of the output matrix.
          The expected shape is :math:`(B, *)`.

    Returns:
        The vector with same size as input :math:`(*, N, 1)`.

    """
    assert n > 0, (type(n), n)
    assert len(tensor.shape) >= 1, tensor.shape

    vec = torch.zeros(n, 1, device=tensor.device, dtype=tensor.dtype)
    return vec[None].repeat(tensor.shape[0], 1, 1)
