"""Module containing numerical functionalities for SfM."""

import torch


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
    zeros = torch.zeros_like(x0)
    cross_product_matrix_flat = torch.stack([zeros, -x2, x1, x2, zeros, -x0, -x1, x0, zeros], dim=-1)
    shape_ = x.shape[:-1] + (3, 3)
    return cross_product_matrix_flat.view(*shape_)


# Reference
# Nistér, David. An efficient solution to the five-point relative pose problem. 2004.
# https://github.com/danini/graph-cut-ransac/blob/master/src/pygcransac/include/
# estimators/solver_essential_matrix_five_point_nister.h#L108


def o1(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    r"""Multiply two polynomials of degree one in x, y, z.

    Args:
        a, b are first order polys :math:`[x,y,z,1]`.

    Returns:
        degree 2 poly with the order :math:`[ x^2, x*y, x*z, x, y^2, y*z, y, z^2, z, 1]`.
    """

    return torch.stack(
        [
            a[:, 0] * b[:, 0],
            a[:, 0] * b[:, 1] + a[:, 1] * b[:, 0],
            a[:, 0] * b[:, 2] + a[:, 2] * b[:, 0],
            a[:, 0] * b[:, 3] + a[:, 3] * b[:, 0],
            a[:, 1] * b[:, 1],
            a[:, 1] * b[:, 2] + a[:, 2] * b[:, 1],
            a[:, 1] * b[:, 3] + a[:, 3] * b[:, 1],
            a[:, 2] * b[:, 2],
            a[:, 2] * b[:, 3] + a[:, 3] * b[:, 2],
            a[:, 3] * b[:, 3],
        ],
        dim=-1,
    )


# Reference
# Nistér, David. An efficient solution to the five-point relative pose problem. 2004.
# https://github.com/danini/graph-cut-ransac/blob/aae1f40c2e10e31fd2191bac601c53a189673f60/src/pygcransac/
# include/estimators/solver_essential_matrix_five_point_nister.h#L156


def o2(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    r"""Multiply two polynomials a and b of degrees two and one.

    Args:
        a is second degree poly with order :math:`[ x^2, x*y, x*z, x, y^2, y*z, y, z^2, z, 1]`.
        b is first degree with order :math:`[x y z 1]`.
    Returns:
        a third degree poly with order,
        :math:`[ x^3, y^3, x^2*y, x*y^2, x^2*z, x^2, y^2*z, y^2, x*y*z, x*y, x*z^2, x*z, x, y*z^2, y*z, y, z^3, z^2, z, 1]`.
    """

    return torch.stack(
        [
            a[:, 0] * b[:, 0],
            a[:, 4] * b[:, 1],
            a[:, 0] * b[:, 1] + a[:, 1] * b[:, 0],
            a[:, 1] * b[:, 1] + a[:, 4] * b[:, 0],
            a[:, 0] * b[:, 2] + a[:, 2] * b[:, 0],
            a[:, 0] * b[:, 3] + a[:, 3] * b[:, 0],
            a[:, 4] * b[:, 2] + a[:, 5] * b[:, 1],
            a[:, 4] * b[:, 3] + a[:, 6] * b[:, 1],
            a[:, 1] * b[:, 2] + a[:, 2] * b[:, 1] + a[:, 5] * b[:, 0],
            a[:, 1] * b[:, 3] + a[:, 3] * b[:, 1] + a[:, 6] * b[:, 0],
            a[:, 2] * b[:, 2] + a[:, 7] * b[:, 0],
            a[:, 2] * b[:, 3] + a[:, 3] * b[:, 2] + a[:, 8] * b[:, 0],
            a[:, 3] * b[:, 3] + a[:, 9] * b[:, 0],
            a[:, 5] * b[:, 2] + a[:, 7] * b[:, 1],
            a[:, 5] * b[:, 3] + a[:, 6] * b[:, 2] + a[:, 8] * b[:, 1],
            a[:, 6] * b[:, 3] + a[:, 9] * b[:, 1],
            a[:, 7] * b[:, 2],
            a[:, 7] * b[:, 3] + a[:, 8] * b[:, 2],
            a[:, 8] * b[:, 3] + a[:, 9] * b[:, 2],
            a[:, 9] * b[:, 3],
        ],
        dim=-1,
    )


# Compute degree 10 poly representing determinant (equation 14 in the paper)
# https://github.com/danini/graph-cut-ransac/blob/aae1f40c2e10e31fd2191bac601c53a189673f60/src/pygcransac/
# include/estimators/solver_essential_matrix_five_point_nister.h#L368C5-L368C82
def det_to_poly(A: torch.Tensor) -> torch.Tensor:
    r"""Represent the determinant by the 10th polynomial, used for 5PC solver.

    Args:
        A is in the shape of :math:`(*, 3, 13)`.
    Returns:
        a degree 10 poly, representing determinant (equation 14 in the paper).
    """

    cs = torch.zeros(A.shape[0], 11, device=A.device, dtype=A.dtype)
    cs[:, 0] = (
        A[:, 0, 12] * A[:, 1, 3] * A[:, 2, 7]
        - A[:, 0, 12] * A[:, 1, 7] * A[:, 2, 3]
        - A[:, 0, 3] * A[:, 2, 7] * A[:, 1, 12]
        + A[:, 0, 7] * A[:, 2, 3] * A[:, 1, 12]
        + A[:, 0, 3] * A[:, 1, 7] * A[:, 2, 12]
        - A[:, 0, 7] * A[:, 1, 3] * A[:, 2, 12]
    )

    cs[:, 1] = (
        A[:, 0, 11] * A[:, 1, 3] * A[:, 2, 7]
        - A[:, 0, 11] * A[:, 1, 7] * A[:, 2, 3]
        + A[:, 0, 12] * A[:, 1, 2] * A[:, 2, 7]
        + A[:, 0, 12] * A[:, 1, 3] * A[:, 2, 6]
        - A[:, 0, 12] * A[:, 1, 6] * A[:, 2, 3]
        - A[:, 0, 12] * A[:, 1, 7] * A[:, 2, 2]
        - A[:, 0, 2] * A[:, 2, 7] * A[:, 1, 12]
        - A[:, 0, 3] * A[:, 2, 6] * A[:, 1, 12]
        - A[:, 0, 3] * A[:, 2, 7] * A[:, 1, 11]
        + A[:, 0, 6] * A[:, 2, 3] * A[:, 1, 12]
        + A[:, 0, 7] * A[:, 2, 2] * A[:, 1, 12]
        + A[:, 0, 7] * A[:, 2, 3] * A[:, 1, 11]
        + A[:, 0, 2] * A[:, 1, 7] * A[:, 2, 12]
        + A[:, 0, 3] * A[:, 1, 6] * A[:, 2, 12]
        + A[:, 0, 3] * A[:, 1, 7] * A[:, 2, 11]
        - A[:, 0, 6] * A[:, 1, 3] * A[:, 2, 12]
        - A[:, 0, 7] * A[:, 1, 2] * A[:, 2, 12]
        - A[:, 0, 7] * A[:, 1, 3] * A[:, 2, 11]
    )

    cs[:, 2] = (
        A[:, 0, 10] * A[:, 1, 3] * A[:, 2, 7]
        - A[:, 0, 10] * A[:, 1, 7] * A[:, 2, 3]
        + A[:, 0, 11] * A[:, 1, 2] * A[:, 2, 7]
        + A[:, 0, 11] * A[:, 1, 3] * A[:, 2, 6]
        - A[:, 0, 11] * A[:, 1, 6] * A[:, 2, 3]
        - A[:, 0, 11] * A[:, 1, 7] * A[:, 2, 2]
        + A[:, 1, 1] * A[:, 0, 12] * A[:, 2, 7]
        + A[:, 0, 12] * A[:, 1, 2] * A[:, 2, 6]
        + A[:, 0, 12] * A[:, 1, 3] * A[:, 2, 5]
        - A[:, 0, 12] * A[:, 1, 5] * A[:, 2, 3]
        - A[:, 0, 12] * A[:, 1, 6] * A[:, 2, 2]
        - A[:, 0, 12] * A[:, 1, 7] * A[:, 2, 1]
        - A[:, 0, 1] * A[:, 2, 7] * A[:, 1, 12]
        - A[:, 0, 2] * A[:, 2, 6] * A[:, 1, 12]
        - A[:, 0, 2] * A[:, 2, 7] * A[:, 1, 11]
        - A[:, 0, 3] * A[:, 2, 5] * A[:, 1, 12]
        - A[:, 0, 3] * A[:, 2, 6] * A[:, 1, 11]
        - A[:, 0, 3] * A[:, 2, 7] * A[:, 1, 10]
        + A[:, 0, 5] * A[:, 2, 3] * A[:, 1, 12]
        + A[:, 0, 6] * A[:, 2, 2] * A[:, 1, 12]
        + A[:, 0, 6] * A[:, 2, 3] * A[:, 1, 11]
        + A[:, 0, 7] * A[:, 2, 1] * A[:, 1, 12]
        + A[:, 0, 7] * A[:, 2, 2] * A[:, 1, 11]
        + A[:, 0, 7] * A[:, 2, 3] * A[:, 1, 10]
        + A[:, 0, 1] * A[:, 1, 7] * A[:, 2, 12]
        + A[:, 0, 2] * A[:, 1, 6] * A[:, 2, 12]
        + A[:, 0, 2] * A[:, 1, 7] * A[:, 2, 11]
        + A[:, 0, 3] * A[:, 1, 5] * A[:, 2, 12]
        + A[:, 0, 3] * A[:, 1, 6] * A[:, 2, 11]
        + A[:, 0, 3] * A[:, 1, 7] * A[:, 2, 10]
        - A[:, 0, 5] * A[:, 1, 3] * A[:, 2, 12]
        - A[:, 0, 6] * A[:, 1, 2] * A[:, 2, 12]
        - A[:, 0, 6] * A[:, 1, 3] * A[:, 2, 11]
        - A[:, 0, 7] * A[:, 1, 1] * A[:, 2, 12]
        - A[:, 0, 7] * A[:, 1, 2] * A[:, 2, 11]
        - A[:, 0, 7] * A[:, 1, 3] * A[:, 2, 10]
    )

    cs[:, 3] = (
        A[:, 0, 3] * A[:, 1, 7] * A[:, 2, 9]
        - A[:, 0, 3] * A[:, 1, 9] * A[:, 2, 7]
        - A[:, 0, 7] * A[:, 1, 3] * A[:, 2, 9]
        + A[:, 0, 7] * A[:, 1, 9] * A[:, 2, 3]
        + A[:, 0, 9] * A[:, 1, 3] * A[:, 2, 7]
        - A[:, 0, 9] * A[:, 1, 7] * A[:, 2, 3]
        + A[:, 0, 10] * A[:, 1, 2] * A[:, 2, 7]
        + A[:, 0, 10] * A[:, 1, 3] * A[:, 2, 6]
        - A[:, 0, 10] * A[:, 1, 6] * A[:, 2, 3]
        - A[:, 0, 10] * A[:, 1, 7] * A[:, 2, 2]
        + A[:, 1, 0] * A[:, 0, 12] * A[:, 2, 7]
        + A[:, 0, 11] * A[:, 1, 1] * A[:, 2, 7]
        + A[:, 0, 11] * A[:, 1, 2] * A[:, 2, 6]
        + A[:, 0, 11] * A[:, 1, 3] * A[:, 2, 5]
        - A[:, 0, 11] * A[:, 1, 5] * A[:, 2, 3]
        - A[:, 0, 11] * A[:, 1, 6] * A[:, 2, 2]
        - A[:, 0, 11] * A[:, 1, 7] * A[:, 2, 1]
        + A[:, 1, 1] * A[:, 0, 12] * A[:, 2, 6]
        + A[:, 0, 12] * A[:, 1, 2] * A[:, 2, 5]
        + A[:, 0, 12] * A[:, 1, 3] * A[:, 2, 4]
        - A[:, 0, 12] * A[:, 1, 4] * A[:, 2, 3]
        - A[:, 0, 12] * A[:, 1, 5] * A[:, 2, 2]
        - A[:, 0, 12] * A[:, 1, 6] * A[:, 2, 1]
        - A[:, 0, 12] * A[:, 1, 7] * A[:, 2, 0]
        - A[:, 0, 0] * A[:, 2, 7] * A[:, 1, 12]
        - A[:, 0, 1] * A[:, 2, 6] * A[:, 1, 12]
        - A[:, 0, 1] * A[:, 2, 7] * A[:, 1, 11]
        - A[:, 0, 2] * A[:, 2, 5] * A[:, 1, 12]
        - A[:, 0, 2] * A[:, 2, 6] * A[:, 1, 11]
        - A[:, 0, 2] * A[:, 2, 7] * A[:, 1, 10]
        - A[:, 0, 3] * A[:, 2, 4] * A[:, 1, 12]
        - A[:, 0, 3] * A[:, 2, 5] * A[:, 1, 11]
        - A[:, 0, 3] * A[:, 2, 6] * A[:, 1, 10]
        + A[:, 0, 4] * A[:, 2, 3] * A[:, 1, 12]
        + A[:, 0, 5] * A[:, 2, 2] * A[:, 1, 12]
        + A[:, 0, 5] * A[:, 2, 3] * A[:, 1, 11]
        + A[:, 0, 6] * A[:, 2, 1] * A[:, 1, 12]
        + A[:, 0, 6] * A[:, 2, 2] * A[:, 1, 11]
        + A[:, 0, 6] * A[:, 2, 3] * A[:, 1, 10]
        + A[:, 0, 7] * A[:, 2, 0] * A[:, 1, 12]
        + A[:, 0, 7] * A[:, 2, 1] * A[:, 1, 11]
        + A[:, 0, 7] * A[:, 2, 2] * A[:, 1, 10]
        + A[:, 0, 0] * A[:, 1, 7] * A[:, 2, 12]
        + A[:, 0, 1] * A[:, 1, 6] * A[:, 2, 12]
        + A[:, 0, 1] * A[:, 1, 7] * A[:, 2, 11]
        + A[:, 0, 2] * A[:, 1, 5] * A[:, 2, 12]
        + A[:, 0, 2] * A[:, 1, 6] * A[:, 2, 11]
        + A[:, 0, 2] * A[:, 1, 7] * A[:, 2, 10]
        + A[:, 0, 3] * A[:, 1, 4] * A[:, 2, 12]
        + A[:, 0, 3] * A[:, 1, 5] * A[:, 2, 11]
        + A[:, 0, 3] * A[:, 1, 6] * A[:, 2, 10]
        - A[:, 0, 4] * A[:, 1, 3] * A[:, 2, 12]
        - A[:, 0, 5] * A[:, 1, 2] * A[:, 2, 12]
        - A[:, 0, 5] * A[:, 1, 3] * A[:, 2, 11]
        - A[:, 0, 6] * A[:, 1, 1] * A[:, 2, 12]
        - A[:, 0, 6] * A[:, 1, 2] * A[:, 2, 11]
        - A[:, 0, 6] * A[:, 1, 3] * A[:, 2, 10]
        - A[:, 0, 7] * A[:, 1, 0] * A[:, 2, 12]
        - A[:, 0, 7] * A[:, 1, 1] * A[:, 2, 11]
        - A[:, 0, 7] * A[:, 1, 2] * A[:, 2, 10]
    )

    cs[:, 4] = (
        A[:, 0, 2] * A[:, 1, 7] * A[:, 2, 9]
        - A[:, 0, 2] * A[:, 1, 9] * A[:, 2, 7]
        + A[:, 0, 3] * A[:, 1, 6] * A[:, 2, 9]
        + A[:, 0, 3] * A[:, 1, 7] * A[:, 2, 8]
        - A[:, 0, 3] * A[:, 1, 8] * A[:, 2, 7]
        - A[:, 0, 3] * A[:, 1, 9] * A[:, 2, 6]
        - A[:, 0, 6] * A[:, 1, 3] * A[:, 2, 9]
        + A[:, 0, 6] * A[:, 1, 9] * A[:, 2, 3]
        - A[:, 0, 7] * A[:, 1, 2] * A[:, 2, 9]
        - A[:, 0, 7] * A[:, 1, 3] * A[:, 2, 8]
        + A[:, 0, 7] * A[:, 1, 8] * A[:, 2, 3]
        + A[:, 0, 7] * A[:, 1, 9] * A[:, 2, 2]
        + A[:, 0, 8] * A[:, 1, 3] * A[:, 2, 7]
        - A[:, 0, 8] * A[:, 1, 7] * A[:, 2, 3]
        + A[:, 0, 9] * A[:, 1, 2] * A[:, 2, 7]
        + A[:, 0, 9] * A[:, 1, 3] * A[:, 2, 6]
        - A[:, 0, 9] * A[:, 1, 6] * A[:, 2, 3]
        - A[:, 0, 9] * A[:, 1, 7] * A[:, 2, 2]
        + A[:, 0, 10] * A[:, 1, 1] * A[:, 2, 7]
        + A[:, 0, 10] * A[:, 1, 2] * A[:, 2, 6]
        + A[:, 0, 10] * A[:, 1, 3] * A[:, 2, 5]
        - A[:, 0, 10] * A[:, 1, 5] * A[:, 2, 3]
        - A[:, 0, 10] * A[:, 1, 6] * A[:, 2, 2]
        - A[:, 0, 10] * A[:, 1, 7] * A[:, 2, 1]
        + A[:, 1, 0] * A[:, 0, 11] * A[:, 2, 7]
        + A[:, 1, 0] * A[:, 0, 12] * A[:, 2, 6]
        + A[:, 0, 11] * A[:, 1, 1] * A[:, 2, 6]
        + A[:, 0, 11] * A[:, 1, 2] * A[:, 2, 5]
        + A[:, 0, 11] * A[:, 1, 3] * A[:, 2, 4]
        - A[:, 0, 11] * A[:, 1, 4] * A[:, 2, 3]
        - A[:, 0, 11] * A[:, 1, 5] * A[:, 2, 2]
        - A[:, 0, 11] * A[:, 1, 6] * A[:, 2, 1]
        - A[:, 0, 11] * A[:, 1, 7] * A[:, 2, 0]
        + A[:, 1, 1] * A[:, 0, 12] * A[:, 2, 5]
        + A[:, 0, 12] * A[:, 1, 2] * A[:, 2, 4]
        - A[:, 0, 12] * A[:, 1, 4] * A[:, 2, 2]
        - A[:, 0, 12] * A[:, 1, 5] * A[:, 2, 1]
        - A[:, 0, 12] * A[:, 1, 6] * A[:, 2, 0]
        - A[:, 0, 0] * A[:, 2, 6] * A[:, 1, 12]
        - A[:, 0, 0] * A[:, 2, 7] * A[:, 1, 11]
        - A[:, 0, 1] * A[:, 2, 5] * A[:, 1, 12]
        - A[:, 0, 1] * A[:, 2, 6] * A[:, 1, 11]
        - A[:, 0, 1] * A[:, 2, 7] * A[:, 1, 10]
        - A[:, 0, 2] * A[:, 2, 4] * A[:, 1, 12]
        - A[:, 0, 2] * A[:, 2, 5] * A[:, 1, 11]
        - A[:, 0, 2] * A[:, 2, 6] * A[:, 1, 10]
        - A[:, 0, 3] * A[:, 2, 4] * A[:, 1, 11]
        - A[:, 0, 3] * A[:, 2, 5] * A[:, 1, 10]
        + A[:, 0, 4] * A[:, 2, 2] * A[:, 1, 12]
        + A[:, 0, 4] * A[:, 2, 3] * A[:, 1, 11]
        + A[:, 0, 5] * A[:, 2, 1] * A[:, 1, 12]
        + A[:, 0, 5] * A[:, 2, 2] * A[:, 1, 11]
        + A[:, 0, 5] * A[:, 2, 3] * A[:, 1, 10]
        + A[:, 0, 6] * A[:, 2, 0] * A[:, 1, 12]
        + A[:, 0, 6] * A[:, 2, 1] * A[:, 1, 11]
        + A[:, 0, 6] * A[:, 2, 2] * A[:, 1, 10]
        + A[:, 0, 7] * A[:, 2, 0] * A[:, 1, 11]
        + A[:, 0, 7] * A[:, 2, 1] * A[:, 1, 10]
        + A[:, 0, 0] * A[:, 1, 6] * A[:, 2, 12]
        + A[:, 0, 0] * A[:, 1, 7] * A[:, 2, 11]
        + A[:, 0, 1] * A[:, 1, 5] * A[:, 2, 12]
        + A[:, 0, 1] * A[:, 1, 6] * A[:, 2, 11]
        + A[:, 0, 1] * A[:, 1, 7] * A[:, 2, 10]
        + A[:, 0, 2] * A[:, 1, 4] * A[:, 2, 12]
        + A[:, 0, 2] * A[:, 1, 5] * A[:, 2, 11]
        + A[:, 0, 2] * A[:, 1, 6] * A[:, 2, 10]
        + A[:, 0, 3] * A[:, 1, 4] * A[:, 2, 11]
        + A[:, 0, 3] * A[:, 1, 5] * A[:, 2, 10]
        - A[:, 0, 4] * A[:, 1, 2] * A[:, 2, 12]
        - A[:, 0, 4] * A[:, 1, 3] * A[:, 2, 11]
        - A[:, 0, 5] * A[:, 1, 1] * A[:, 2, 12]
        - A[:, 0, 5] * A[:, 1, 2] * A[:, 2, 11]
        - A[:, 0, 5] * A[:, 1, 3] * A[:, 2, 10]
        - A[:, 0, 6] * A[:, 1, 0] * A[:, 2, 12]
        - A[:, 0, 6] * A[:, 1, 1] * A[:, 2, 11]
        - A[:, 0, 6] * A[:, 1, 2] * A[:, 2, 10]
        - A[:, 0, 7] * A[:, 1, 0] * A[:, 2, 11]
        - A[:, 0, 7] * A[:, 1, 1] * A[:, 2, 10]
    )

    cs[:, 5] = (
        A[:, 0, 1] * A[:, 1, 7] * A[:, 2, 9]
        - A[:, 0, 1] * A[:, 1, 9] * A[:, 2, 7]
        + A[:, 0, 2] * A[:, 1, 6] * A[:, 2, 9]
        + A[:, 0, 2] * A[:, 1, 7] * A[:, 2, 8]
        - A[:, 0, 2] * A[:, 1, 8] * A[:, 2, 7]
        - A[:, 0, 2] * A[:, 1, 9] * A[:, 2, 6]
        + A[:, 0, 3] * A[:, 1, 5] * A[:, 2, 9]
        + A[:, 0, 3] * A[:, 1, 6] * A[:, 2, 8]
        - A[:, 0, 3] * A[:, 1, 8] * A[:, 2, 6]
        - A[:, 0, 3] * A[:, 1, 9] * A[:, 2, 5]
        - A[:, 0, 5] * A[:, 1, 3] * A[:, 2, 9]
        + A[:, 0, 5] * A[:, 1, 9] * A[:, 2, 3]
        - A[:, 0, 6] * A[:, 1, 2] * A[:, 2, 9]
        - A[:, 0, 6] * A[:, 1, 3] * A[:, 2, 8]
        + A[:, 0, 6] * A[:, 1, 8] * A[:, 2, 3]
        + A[:, 0, 6] * A[:, 1, 9] * A[:, 2, 2]
        - A[:, 0, 7] * A[:, 1, 1] * A[:, 2, 9]
        - A[:, 0, 7] * A[:, 1, 2] * A[:, 2, 8]
        + A[:, 0, 7] * A[:, 1, 8] * A[:, 2, 2]
        + A[:, 0, 7] * A[:, 1, 9] * A[:, 2, 1]
        + A[:, 0, 8] * A[:, 1, 2] * A[:, 2, 7]
        + A[:, 0, 8] * A[:, 1, 3] * A[:, 2, 6]
        - A[:, 0, 8] * A[:, 1, 6] * A[:, 2, 3]
        - A[:, 0, 8] * A[:, 1, 7] * A[:, 2, 2]
        + A[:, 0, 9] * A[:, 1, 1] * A[:, 2, 7]
        + A[:, 0, 9] * A[:, 1, 2] * A[:, 2, 6]
        + A[:, 0, 9] * A[:, 1, 3] * A[:, 2, 5]
        - A[:, 0, 9] * A[:, 1, 5] * A[:, 2, 3]
        - A[:, 0, 9] * A[:, 1, 6] * A[:, 2, 2]
        - A[:, 0, 9] * A[:, 1, 7] * A[:, 2, 1]
        + A[:, 0, 10] * A[:, 1, 0] * A[:, 2, 7]
        + A[:, 0, 10] * A[:, 1, 1] * A[:, 2, 6]
        + A[:, 0, 10] * A[:, 1, 2] * A[:, 2, 5]
        + A[:, 0, 10] * A[:, 1, 3] * A[:, 2, 4]
        - A[:, 0, 10] * A[:, 1, 4] * A[:, 2, 3]
        - A[:, 0, 10] * A[:, 1, 5] * A[:, 2, 2]
        - A[:, 0, 10] * A[:, 1, 6] * A[:, 2, 1]
        - A[:, 0, 10] * A[:, 1, 7] * A[:, 2, 0]
        + A[:, 1, 0] * A[:, 0, 11] * A[:, 2, 6]
        + A[:, 1, 0] * A[:, 0, 12] * A[:, 2, 5]
        + A[:, 0, 11] * A[:, 1, 1] * A[:, 2, 5]
        + A[:, 0, 11] * A[:, 1, 2] * A[:, 2, 4]
        - A[:, 0, 11] * A[:, 1, 4] * A[:, 2, 2]
        - A[:, 0, 11] * A[:, 1, 5] * A[:, 2, 1]
        - A[:, 0, 11] * A[:, 1, 6] * A[:, 2, 0]
        + A[:, 1, 1] * A[:, 0, 12] * A[:, 2, 4]
        - A[:, 0, 12] * A[:, 1, 4] * A[:, 2, 1]
        - A[:, 0, 12] * A[:, 1, 5] * A[:, 2, 0]
        - A[:, 0, 0] * A[:, 2, 5] * A[:, 1, 12]
        - A[:, 0, 0] * A[:, 2, 6] * A[:, 1, 11]
        - A[:, 0, 0] * A[:, 2, 7] * A[:, 1, 10]
        - A[:, 0, 1] * A[:, 2, 4] * A[:, 1, 12]
        - A[:, 0, 1] * A[:, 2, 5] * A[:, 1, 11]
        - A[:, 0, 1] * A[:, 2, 6] * A[:, 1, 10]
        - A[:, 0, 2] * A[:, 2, 4] * A[:, 1, 11]
        - A[:, 0, 2] * A[:, 2, 5] * A[:, 1, 10]
        - A[:, 0, 3] * A[:, 2, 4] * A[:, 1, 10]
        + A[:, 0, 4] * A[:, 2, 1] * A[:, 1, 12]
        + A[:, 0, 4] * A[:, 2, 2] * A[:, 1, 11]
        + A[:, 0, 4] * A[:, 2, 3] * A[:, 1, 10]
        + A[:, 0, 5] * A[:, 2, 0] * A[:, 1, 12]
        + A[:, 0, 5] * A[:, 2, 1] * A[:, 1, 11]
        + A[:, 0, 5] * A[:, 2, 2] * A[:, 1, 10]
        + A[:, 0, 6] * A[:, 2, 0] * A[:, 1, 11]
        + A[:, 0, 6] * A[:, 2, 1] * A[:, 1, 10]
        + A[:, 0, 7] * A[:, 2, 0] * A[:, 1, 10]
        + A[:, 0, 0] * A[:, 1, 5] * A[:, 2, 12]
        + A[:, 0, 0] * A[:, 1, 6] * A[:, 2, 11]
        + A[:, 0, 0] * A[:, 1, 7] * A[:, 2, 10]
        + A[:, 0, 1] * A[:, 1, 4] * A[:, 2, 12]
        + A[:, 0, 1] * A[:, 1, 5] * A[:, 2, 11]
        + A[:, 0, 1] * A[:, 1, 6] * A[:, 2, 10]
        + A[:, 0, 2] * A[:, 1, 4] * A[:, 2, 11]
        + A[:, 0, 2] * A[:, 1, 5] * A[:, 2, 10]
        + A[:, 0, 3] * A[:, 1, 4] * A[:, 2, 10]
        - A[:, 0, 4] * A[:, 1, 1] * A[:, 2, 12]
        - A[:, 0, 4] * A[:, 1, 2] * A[:, 2, 11]
        - A[:, 0, 4] * A[:, 1, 3] * A[:, 2, 10]
        - A[:, 0, 5] * A[:, 1, 0] * A[:, 2, 12]
        - A[:, 0, 5] * A[:, 1, 1] * A[:, 2, 11]
        - A[:, 0, 5] * A[:, 1, 2] * A[:, 2, 10]
        - A[:, 0, 6] * A[:, 1, 0] * A[:, 2, 11]
        - A[:, 0, 6] * A[:, 1, 1] * A[:, 2, 10]
        - A[:, 0, 7] * A[:, 1, 0] * A[:, 2, 10]
    )

    cs[:, 6] = (
        A[:, 0, 0] * A[:, 1, 7] * A[:, 2, 9]
        - A[:, 0, 0] * A[:, 1, 9] * A[:, 2, 7]
        + A[:, 0, 1] * A[:, 1, 6] * A[:, 2, 9]
        + A[:, 0, 1] * A[:, 1, 7] * A[:, 2, 8]
        - A[:, 0, 1] * A[:, 1, 8] * A[:, 2, 7]
        - A[:, 0, 1] * A[:, 1, 9] * A[:, 2, 6]
        + A[:, 0, 2] * A[:, 1, 5] * A[:, 2, 9]
        + A[:, 0, 2] * A[:, 1, 6] * A[:, 2, 8]
        - A[:, 0, 2] * A[:, 1, 8] * A[:, 2, 6]
        - A[:, 0, 2] * A[:, 1, 9] * A[:, 2, 5]
        + A[:, 0, 3] * A[:, 1, 4] * A[:, 2, 9]
        + A[:, 0, 3] * A[:, 1, 5] * A[:, 2, 8]
        - A[:, 0, 3] * A[:, 1, 8] * A[:, 2, 5]
        - A[:, 0, 3] * A[:, 1, 9] * A[:, 2, 4]
        - A[:, 0, 4] * A[:, 1, 3] * A[:, 2, 9]
        + A[:, 0, 4] * A[:, 1, 9] * A[:, 2, 3]
        - A[:, 0, 5] * A[:, 1, 2] * A[:, 2, 9]
        - A[:, 0, 5] * A[:, 1, 3] * A[:, 2, 8]
        + A[:, 0, 5] * A[:, 1, 8] * A[:, 2, 3]
        + A[:, 0, 5] * A[:, 1, 9] * A[:, 2, 2]
        - A[:, 0, 6] * A[:, 1, 1] * A[:, 2, 9]
        - A[:, 0, 6] * A[:, 1, 2] * A[:, 2, 8]
        + A[:, 0, 6] * A[:, 1, 8] * A[:, 2, 2]
        + A[:, 0, 6] * A[:, 1, 9] * A[:, 2, 1]
        - A[:, 0, 7] * A[:, 1, 0] * A[:, 2, 9]
        - A[:, 0, 7] * A[:, 1, 1] * A[:, 2, 8]
        + A[:, 0, 7] * A[:, 1, 8] * A[:, 2, 1]
        + A[:, 0, 7] * A[:, 1, 9] * A[:, 2, 0]
        + A[:, 0, 8] * A[:, 1, 1] * A[:, 2, 7]
        + A[:, 0, 8] * A[:, 1, 2] * A[:, 2, 6]
        + A[:, 0, 8] * A[:, 1, 3] * A[:, 2, 5]
        - A[:, 0, 8] * A[:, 1, 5] * A[:, 2, 3]
        - A[:, 0, 8] * A[:, 1, 6] * A[:, 2, 2]
        - A[:, 0, 8] * A[:, 1, 7] * A[:, 2, 1]
        + A[:, 0, 9] * A[:, 1, 0] * A[:, 2, 7]
        + A[:, 0, 9] * A[:, 1, 1] * A[:, 2, 6]
        + A[:, 0, 9] * A[:, 1, 2] * A[:, 2, 5]
        + A[:, 0, 9] * A[:, 1, 3] * A[:, 2, 4]
        - A[:, 0, 9] * A[:, 1, 4] * A[:, 2, 3]
        - A[:, 0, 9] * A[:, 1, 5] * A[:, 2, 2]
        - A[:, 0, 9] * A[:, 1, 6] * A[:, 2, 1]
        - A[:, 0, 9] * A[:, 1, 7] * A[:, 2, 0]
        + A[:, 0, 10] * A[:, 1, 0] * A[:, 2, 6]
        + A[:, 0, 10] * A[:, 1, 1] * A[:, 2, 5]
        + A[:, 0, 10] * A[:, 1, 2] * A[:, 2, 4]
        - A[:, 0, 10] * A[:, 1, 4] * A[:, 2, 2]
        - A[:, 0, 10] * A[:, 1, 5] * A[:, 2, 1]
        - A[:, 0, 10] * A[:, 1, 6] * A[:, 2, 0]
        + A[:, 1, 0] * A[:, 0, 11] * A[:, 2, 5]
        + A[:, 1, 0] * A[:, 0, 12] * A[:, 2, 4]
        + A[:, 0, 11] * A[:, 1, 1] * A[:, 2, 4]
        - A[:, 0, 11] * A[:, 1, 4] * A[:, 2, 1]
        - A[:, 0, 11] * A[:, 1, 5] * A[:, 2, 0]
        - A[:, 0, 12] * A[:, 1, 4] * A[:, 2, 0]
        - A[:, 0, 0] * A[:, 2, 4] * A[:, 1, 12]
        - A[:, 0, 0] * A[:, 2, 5] * A[:, 1, 11]
        - A[:, 0, 0] * A[:, 2, 6] * A[:, 1, 10]
        - A[:, 0, 1] * A[:, 2, 4] * A[:, 1, 11]
        - A[:, 0, 1] * A[:, 2, 5] * A[:, 1, 10]
        - A[:, 0, 2] * A[:, 2, 4] * A[:, 1, 10]
        + A[:, 0, 4] * A[:, 2, 0] * A[:, 1, 12]
        + A[:, 0, 4] * A[:, 2, 1] * A[:, 1, 11]
        + A[:, 0, 4] * A[:, 2, 2] * A[:, 1, 10]
        + A[:, 0, 5] * A[:, 2, 0] * A[:, 1, 11]
        + A[:, 0, 5] * A[:, 2, 1] * A[:, 1, 10]
        + A[:, 0, 6] * A[:, 2, 0] * A[:, 1, 10]
        + A[:, 0, 0] * A[:, 1, 4] * A[:, 2, 12]
        + A[:, 0, 0] * A[:, 1, 5] * A[:, 2, 11]
        + A[:, 0, 0] * A[:, 1, 6] * A[:, 2, 10]
        + A[:, 0, 1] * A[:, 1, 4] * A[:, 2, 11]
        + A[:, 0, 1] * A[:, 1, 5] * A[:, 2, 10]
        + A[:, 0, 2] * A[:, 1, 4] * A[:, 2, 10]
        - A[:, 0, 4] * A[:, 1, 0] * A[:, 2, 12]
        - A[:, 0, 4] * A[:, 1, 1] * A[:, 2, 11]
        - A[:, 0, 4] * A[:, 1, 2] * A[:, 2, 10]
        - A[:, 0, 5] * A[:, 1, 0] * A[:, 2, 11]
        - A[:, 0, 5] * A[:, 1, 1] * A[:, 2, 10]
        - A[:, 0, 6] * A[:, 1, 0] * A[:, 2, 10]
    )

    cs[:, 7] = (
        A[:, 0, 0] * A[:, 1, 6] * A[:, 2, 9]
        + A[:, 0, 0] * A[:, 1, 7] * A[:, 2, 8]
        - A[:, 0, 0] * A[:, 1, 8] * A[:, 2, 7]
        - A[:, 0, 0] * A[:, 1, 9] * A[:, 2, 6]
        + A[:, 0, 1] * A[:, 1, 5] * A[:, 2, 9]
        + A[:, 0, 1] * A[:, 1, 6] * A[:, 2, 8]
        - A[:, 0, 1] * A[:, 1, 8] * A[:, 2, 6]
        - A[:, 0, 1] * A[:, 1, 9] * A[:, 2, 5]
        + A[:, 0, 2] * A[:, 1, 4] * A[:, 2, 9]
        + A[:, 0, 2] * A[:, 1, 5] * A[:, 2, 8]
        - A[:, 0, 2] * A[:, 1, 8] * A[:, 2, 5]
        - A[:, 0, 2] * A[:, 1, 9] * A[:, 2, 4]
        + A[:, 0, 3] * A[:, 1, 4] * A[:, 2, 8]
        - A[:, 0, 3] * A[:, 1, 8] * A[:, 2, 4]
        - A[:, 0, 4] * A[:, 1, 2] * A[:, 2, 9]
        - A[:, 0, 4] * A[:, 1, 3] * A[:, 2, 8]
        + A[:, 0, 4] * A[:, 1, 8] * A[:, 2, 3]
        + A[:, 0, 4] * A[:, 1, 9] * A[:, 2, 2]
        - A[:, 0, 5] * A[:, 1, 1] * A[:, 2, 9]
        - A[:, 0, 5] * A[:, 1, 2] * A[:, 2, 8]
        + A[:, 0, 5] * A[:, 1, 8] * A[:, 2, 2]
        + A[:, 0, 5] * A[:, 1, 9] * A[:, 2, 1]
        - A[:, 0, 6] * A[:, 1, 0] * A[:, 2, 9]
        - A[:, 0, 6] * A[:, 1, 1] * A[:, 2, 8]
        + A[:, 0, 6] * A[:, 1, 8] * A[:, 2, 1]
        + A[:, 0, 6] * A[:, 1, 9] * A[:, 2, 0]
        - A[:, 0, 7] * A[:, 1, 0] * A[:, 2, 8]
        + A[:, 0, 7] * A[:, 1, 8] * A[:, 2, 0]
        + A[:, 0, 8] * A[:, 1, 0] * A[:, 2, 7]
        + A[:, 0, 8] * A[:, 1, 1] * A[:, 2, 6]
        + A[:, 0, 8] * A[:, 1, 2] * A[:, 2, 5]
        + A[:, 0, 8] * A[:, 1, 3] * A[:, 2, 4]
        - A[:, 0, 8] * A[:, 1, 4] * A[:, 2, 3]
        - A[:, 0, 8] * A[:, 1, 5] * A[:, 2, 2]
        - A[:, 0, 8] * A[:, 1, 6] * A[:, 2, 1]
        - A[:, 0, 8] * A[:, 1, 7] * A[:, 2, 0]
        + A[:, 0, 9] * A[:, 1, 0] * A[:, 2, 6]
        + A[:, 0, 9] * A[:, 1, 1] * A[:, 2, 5]
        + A[:, 0, 9] * A[:, 1, 2] * A[:, 2, 4]
        - A[:, 0, 9] * A[:, 1, 4] * A[:, 2, 2]
        - A[:, 0, 9] * A[:, 1, 5] * A[:, 2, 1]
        - A[:, 0, 9] * A[:, 1, 6] * A[:, 2, 0]
        + A[:, 0, 10] * A[:, 1, 0] * A[:, 2, 5]
        + A[:, 0, 10] * A[:, 1, 1] * A[:, 2, 4]
        - A[:, 0, 10] * A[:, 1, 4] * A[:, 2, 1]
        - A[:, 0, 10] * A[:, 1, 5] * A[:, 2, 0]
        + A[:, 1, 0] * A[:, 0, 11] * A[:, 2, 4]
        - A[:, 0, 11] * A[:, 1, 4] * A[:, 2, 0]
        - A[:, 0, 0] * A[:, 2, 4] * A[:, 1, 11]
        - A[:, 0, 0] * A[:, 2, 5] * A[:, 1, 10]
        - A[:, 0, 1] * A[:, 2, 4] * A[:, 1, 10]
        + A[:, 0, 4] * A[:, 2, 0] * A[:, 1, 11]
        + A[:, 0, 4] * A[:, 2, 1] * A[:, 1, 10]
        + A[:, 0, 5] * A[:, 2, 0] * A[:, 1, 10]
        + A[:, 0, 0] * A[:, 1, 4] * A[:, 2, 11]
        + A[:, 0, 0] * A[:, 1, 5] * A[:, 2, 10]
        + A[:, 0, 1] * A[:, 1, 4] * A[:, 2, 10]
        - A[:, 0, 4] * A[:, 1, 0] * A[:, 2, 11]
        - A[:, 0, 4] * A[:, 1, 1] * A[:, 2, 10]
        - A[:, 0, 5] * A[:, 1, 0] * A[:, 2, 10]
    )

    cs[:, 8] = (
        A[:, 0, 0] * A[:, 1, 5] * A[:, 2, 9]
        + A[:, 0, 0] * A[:, 1, 6] * A[:, 2, 8]
        - A[:, 0, 0] * A[:, 1, 8] * A[:, 2, 6]
        - A[:, 0, 0] * A[:, 1, 9] * A[:, 2, 5]
        + A[:, 0, 1] * A[:, 1, 4] * A[:, 2, 9]
        + A[:, 0, 1] * A[:, 1, 5] * A[:, 2, 8]
        - A[:, 0, 1] * A[:, 1, 8] * A[:, 2, 5]
        - A[:, 0, 1] * A[:, 1, 9] * A[:, 2, 4]
        + A[:, 0, 2] * A[:, 1, 4] * A[:, 2, 8]
        - A[:, 0, 2] * A[:, 1, 8] * A[:, 2, 4]
        - A[:, 0, 4] * A[:, 1, 1] * A[:, 2, 9]
        - A[:, 0, 4] * A[:, 1, 2] * A[:, 2, 8]
        + A[:, 0, 4] * A[:, 1, 8] * A[:, 2, 2]
        + A[:, 0, 4] * A[:, 1, 9] * A[:, 2, 1]
        - A[:, 0, 5] * A[:, 1, 0] * A[:, 2, 9]
        - A[:, 0, 5] * A[:, 1, 1] * A[:, 2, 8]
        + A[:, 0, 5] * A[:, 1, 8] * A[:, 2, 1]
        + A[:, 0, 5] * A[:, 1, 9] * A[:, 2, 0]
        - A[:, 0, 6] * A[:, 1, 0] * A[:, 2, 8]
        + A[:, 0, 6] * A[:, 1, 8] * A[:, 2, 0]
        + A[:, 0, 8] * A[:, 1, 0] * A[:, 2, 6]
        + A[:, 0, 8] * A[:, 1, 1] * A[:, 2, 5]
        + A[:, 0, 8] * A[:, 1, 2] * A[:, 2, 4]
        - A[:, 0, 8] * A[:, 1, 4] * A[:, 2, 2]
        - A[:, 0, 8] * A[:, 1, 5] * A[:, 2, 1]
        - A[:, 0, 8] * A[:, 1, 6] * A[:, 2, 0]
        + A[:, 0, 9] * A[:, 1, 0] * A[:, 2, 5]
        + A[:, 0, 9] * A[:, 1, 1] * A[:, 2, 4]
        - A[:, 0, 9] * A[:, 1, 4] * A[:, 2, 1]
        - A[:, 0, 9] * A[:, 1, 5] * A[:, 2, 0]
        + A[:, 0, 10] * A[:, 1, 0] * A[:, 2, 4]
        - A[:, 0, 10] * A[:, 1, 4] * A[:, 2, 0]
        - A[:, 0, 0] * A[:, 2, 4] * A[:, 1, 10]
        + A[:, 0, 4] * A[:, 2, 0] * A[:, 1, 10]
        + A[:, 0, 0] * A[:, 1, 4] * A[:, 2, 10]
        - A[:, 0, 4] * A[:, 1, 0] * A[:, 2, 10]
    )

    cs[:, 9] = (
        A[:, 0, 0] * A[:, 1, 4] * A[:, 2, 9]
        + A[:, 0, 0] * A[:, 1, 5] * A[:, 2, 8]
        - A[:, 0, 0] * A[:, 1, 8] * A[:, 2, 5]
        - A[:, 0, 0] * A[:, 1, 9] * A[:, 2, 4]
        + A[:, 0, 1] * A[:, 1, 4] * A[:, 2, 8]
        - A[:, 0, 1] * A[:, 1, 8] * A[:, 2, 4]
        - A[:, 0, 4] * A[:, 1, 0] * A[:, 2, 9]
        - A[:, 0, 4] * A[:, 1, 1] * A[:, 2, 8]
        + A[:, 0, 4] * A[:, 1, 8] * A[:, 2, 1]
        + A[:, 0, 4] * A[:, 1, 9] * A[:, 2, 0]
        - A[:, 0, 5] * A[:, 1, 0] * A[:, 2, 8]
        + A[:, 0, 5] * A[:, 1, 8] * A[:, 2, 0]
        + A[:, 0, 8] * A[:, 1, 0] * A[:, 2, 5]
        + A[:, 0, 8] * A[:, 1, 1] * A[:, 2, 4]
        - A[:, 0, 8] * A[:, 1, 4] * A[:, 2, 1]
        - A[:, 0, 8] * A[:, 1, 5] * A[:, 2, 0]
        + A[:, 0, 9] * A[:, 1, 0] * A[:, 2, 4]
        - A[:, 0, 9] * A[:, 1, 4] * A[:, 2, 0]
    )

    cs[:, 10] = (
        A[:, 0, 0] * A[:, 1, 4] * A[:, 2, 8]
        - A[:, 0, 0] * A[:, 1, 8] * A[:, 2, 4]
        - A[:, 0, 4] * A[:, 1, 0] * A[:, 2, 8]
        + A[:, 0, 4] * A[:, 1, 8] * A[:, 2, 0]
        + A[:, 0, 8] * A[:, 1, 0] * A[:, 2, 4]
        - A[:, 0, 8] * A[:, 1, 4] * A[:, 2, 0]
    )

    return cs
