"""Module containing numerical functionalities for SfM."""

import torch


def cross_product_matrix(x: torch.Tensor) -> torch.Tensor:
    r"""Return the cross_product_matrix symmetric matrix of a vector.

    Args:
        x: The input vector to construct the matrix in the shape :math:`(B, 3)`.

    Returns:
        The constructed cross_product_matrix symmetric matrix with shape :math:`(B, 3, 3)`.

    """
    if not (len(x.shape) == 2 and x.shape[1] == 3):
        raise AssertionError(x.shape)
    # get vector compononens
    x0 = x[..., 0]
    x1 = x[..., 1]
    x2 = x[..., 2]

    # construct the matrix, reshape to 3x3 and return
    zeros = torch.zeros_like(x0)
    cross_product_matrix_flat = torch.stack([zeros, -x2, x1, x2, zeros, -x0, -x1, x0, zeros], dim=-1)
    return cross_product_matrix_flat.view(-1, 3, 3)
