"""Module containing numerical functionalities for SfM."""

import torch

from kornia.core import stack, zeros_like


def cross_product_matrix(x: torch.Tensor) -> torch.Tensor:
    r"""Return the cross_product_matrix symmetric matrix of a vector.

    Args:
        x: The input vector to construct the matrix in the shape :math:`(*, 3)`.

    Returns:
        The constructed cross_product_matrix symmetric matrix with shape :math:`(*, 3, 3)`.
    """
    if not x.shape[-1] == 3:
        raise AssertionError(x.shape)
    # get vector compononens
    x0 = x[..., 0]
    x1 = x[..., 1]
    x2 = x[..., 2]

    # construct the matrix, reshape to 3x3 and return
    zeros = zeros_like(x0)
    cross_product_matrix_flat = stack([zeros, -x2, x1, x2, zeros, -x0, -x1, x0, zeros], dim=-1)
    shape_ = x.shape[:-1] + (3, 3)
    return cross_product_matrix_flat.view(*shape_)


def matrix_cofactor_tensor(matrix: torch.Tensor) -> torch.Tensor:
    """Cofactor matrix, refer to the numpy doc.

    Args:
        matrix: The input matrix in the shape :math:`(*, 3, 3)`.
    """
    det = torch.det(matrix)
    singular_mask = det != 0
    if singular_mask.sum() != 0:
        # B, 3, 3
        cofactor = torch.linalg.inv(matrix[singular_mask]).transpose(-2, -1) * det[:, None, None]
        # return cofactor matrix of the given matrix
        returned_cofactor = torch.zeros_like(matrix)
        returned_cofactor[singular_mask] = cofactor
        return returned_cofactor
    else:
        raise Exception("all singular matrices")
