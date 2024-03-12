"""Module containing the functionalities for computing the real roots of polynomial equation."""

import math

import torch

from kornia.core import Tensor, cos, ones_like, stack, zeros, zeros_like
from kornia.core.check import KORNIA_CHECK_SHAPE


# Reference : https://github.com/opencv/opencv/blob/4.x/modules/calib3d/src/polynom_solver.cpp
def solve_quadratic(coeffs: Tensor) -> Tensor:
    r"""Solve given quadratic equation.

    The function takes the coefficients of quadratic equation and returns the real roots.

    .. math:: coeffs[0]x^2 + coeffs[1]x + coeffs[2] = 0

    Args:
        coeffs : The coefficients of quadratic equation :`(B, 3)`

    Returns:
        A tensor of shape `(B, 2)` containing the real roots to the quadratic equation.

    Example:
        >>> coeffs = torch.tensor([[1., 4., 4.]])
        >>> roots = solve_quadratic(coeffs)

    .. note::
       In cases where a quadratic polynomial has only one real root, the output will be in the format
       [real_root, 0]. And for the complex roots should be represented as 0. This is done to maintain
       a consistent output shape for all cases.
    """

    KORNIA_CHECK_SHAPE(coeffs, ["B", "3"])

    # Coefficients of quadratic equation
    a = coeffs[:, 0]  # coefficient of x^2
    b = coeffs[:, 1]  # coefficient of x
    c = coeffs[:, 2]  # constant term

    # Calculate discriminant
    delta = b * b - 4 * a * c

    # Create masks for negative and zero discriminant
    mask_negative = delta < 0
    mask_zero = delta == 0

    # Calculate 1/(2*a) for efficient computation
    inv_2a = 0.5 / a

    # Initialize solutions tensor
    solutions = zeros((coeffs.shape[0], 2), device=coeffs.device, dtype=coeffs.dtype)

    # Handle cases with zero discriminant
    if torch.any(mask_zero):
        solutions[mask_zero, 0] = -b[mask_zero] * inv_2a[mask_zero]
        solutions[mask_zero, 1] = solutions[mask_zero, 0]

    # Negative discriminant cases are automatically handled since solutions is initialized with zeros.

    sqrt_delta = torch.sqrt(delta)

    # Handle cases with non-negative discriminant
    mask = torch.bitwise_and(~mask_negative, ~mask_zero)
    if torch.any(mask):
        solutions[mask, 0] = (-b[mask] + sqrt_delta[mask]) * inv_2a[mask]
        solutions[mask, 1] = (-b[mask] - sqrt_delta[mask]) * inv_2a[mask]

    return solutions


def solve_cubic(coeffs: Tensor) -> Tensor:
    r"""Solve given cubic equation.

    The function takes the coefficients of cubic equation and returns
    the real roots.

    .. math:: coeffs[0]x^3 + coeffs[1]x^2 + coeffs[2]x + coeffs[3] = 0

    Args:
        coeffs : The coefficients cubic equation : `(B, 4)`

    Returns:
        A tensor of shape `(B, 3)` containing the real roots to the cubic equation.

    Example:
        >>> coeffs = torch.tensor([[32., 3., -11., -6.]])
        >>> roots = solve_cubic(coeffs)

    .. note::
       In cases where a cubic polynomial has only one or two real roots, the output for the non-real
       roots should be represented as 0. Thus, the output for a single real root should be in the
       format [real_root, 0, 0], and for two real roots, it should be [real_root_1, real_root_2, 0].
    """
    KORNIA_CHECK_SHAPE(coeffs, ["B", "4"])

    _PI = torch.tensor(math.pi, device=coeffs.device, dtype=coeffs.dtype)

    # Coefficients of cubic equation
    a = coeffs[:, 0]  # coefficient of x^3
    b = coeffs[:, 1]  # coefficient of x^2
    c = coeffs[:, 2]  # coefficient of x
    d = coeffs[:, 3]  # constant term

    solutions = zeros((len(coeffs), 3), device=a.device, dtype=a.dtype)

    mask_a_zero = a == 0
    mask_b_zero = b == 0
    mask_c_zero = c == 0

    # Zero order cases are automatically handled since solutions is initialized with zeros.
    # No need for explicit handling of mask_zero_order as solutions already contains zeros by default.

    mask_first_order = mask_a_zero & mask_b_zero & ~mask_c_zero
    mask_second_order = mask_a_zero & ~mask_b_zero & ~mask_c_zero

    if torch.any(mask_second_order):
        solutions[mask_second_order, 0:2] = solve_quadratic(coeffs[mask_second_order, 1:])

    if torch.any(mask_first_order):
        solutions[mask_first_order, 0] = torch.tensor(1.0, device=a.device, dtype=a.dtype)

    # Normalized form x^3 + a2 * x^2 + a1 * x + a0 = 0
    inv_a = 1.0 / a[~mask_a_zero]
    b_a = inv_a * b[~mask_a_zero]
    b_a2 = b_a * b_a

    c_a = inv_a * c[~mask_a_zero]
    d_a = inv_a * d[~mask_a_zero]

    # Solve the cubic equation
    Q = (3 * c_a - b_a2) / 9
    R = (9 * b_a * c_a - 27 * d_a - 2 * b_a * b_a2) / 54
    Q3 = Q * Q * Q
    D = Q3 + R * R
    b_a_3 = (1.0 / 3.0) * b_a

    a_Q_zero = ones_like(a)
    a_R_zero = ones_like(a)
    a_D_zero = ones_like(a)

    a_Q_zero[~mask_a_zero] = Q
    a_R_zero[~mask_a_zero] = R
    a_D_zero[~mask_a_zero] = D

    # Q == 0
    mask_Q_zero = (Q == 0) & (R != 0)
    mask_Q_zero_solutions = (a_Q_zero == 0) & (a_R_zero != 0)

    if torch.any(mask_Q_zero):
        x0_Q_zero = torch.pow(2 * R[mask_Q_zero], 1 / 3) - b_a_3[mask_Q_zero]
        solutions[mask_Q_zero_solutions, 0] = x0_Q_zero

    mask_QR_zero = (Q == 0) & (R == 0)
    mask_QR_zero_solutions = (a_Q_zero == 0) & (a_R_zero == 0)

    if torch.any(mask_QR_zero):
        solutions[mask_QR_zero_solutions] = stack(
            [-b_a_3[mask_QR_zero], -b_a_3[mask_QR_zero], -b_a_3[mask_QR_zero]], dim=1
        )

    # D <= 0
    mask_D_zero = (D <= 0) & (Q != 0)
    mask_D_zero_solutions = (a_D_zero <= 0) & (a_Q_zero != 0)

    if torch.any(mask_D_zero):
        theta_D_zero = torch.acos(R[mask_D_zero] / torch.sqrt(-Q3[mask_D_zero]))
        sqrt_Q_D_zero = torch.sqrt(-Q[mask_D_zero])
        x0_D_zero = 2 * sqrt_Q_D_zero * cos(theta_D_zero / 3.0) - b_a_3[mask_D_zero]
        x1_D_zero = 2 * sqrt_Q_D_zero * cos((theta_D_zero + 2 * _PI) / 3.0) - b_a_3[mask_D_zero]
        x2_D_zero = 2 * sqrt_Q_D_zero * cos((theta_D_zero + 4 * _PI) / 3.0) - b_a_3[mask_D_zero]
        solutions[mask_D_zero_solutions] = stack([x0_D_zero, x1_D_zero, x2_D_zero], dim=1)

    a_D_positive = zeros_like(a)
    a_D_positive[~mask_a_zero] = D
    # D > 0
    mask_D_positive_solution = (a_D_positive > 0) & (a_Q_zero != 0)
    mask_D_positive = (D > 0) & (Q != 0)
    if torch.any(mask_D_positive):
        AD = zeros_like(R)
        BD = zeros_like(R)
        R_abs = torch.abs(R)
        mask_R_positive = R_abs > 1e-16
        if torch.any(mask_R_positive):
            AD[mask_R_positive] = torch.pow(R_abs[mask_R_positive] + torch.sqrt(D[mask_R_positive]), 1 / 3)
            mask_R_positive_ = R < 0

            if torch.any(mask_R_positive_):
                AD[mask_R_positive_] = -AD[mask_R_positive_]

            BD[mask_R_positive] = -Q[mask_R_positive] / AD[mask_R_positive]
        x0_D_positive = AD[mask_D_positive] + BD[mask_D_positive] - b_a_3[mask_D_positive]
        solutions[mask_D_positive_solution, 0] = x0_D_positive

    return solutions


# def solve_quartic(coeffs: Tensor) -> Tensor:
#    TODO: Quartic equation solver
#     return solutions


# Reference
# https://github.com/danini/graph-cut-ransac/blob/master/src/pygcransac/include/
# estimators/solver_essential_matrix_five_point_nister.h#L108


def multiply_deg_one_poly(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    r"""Multiply two polynomials of the first order [@nister2004efficient].

    Args:
        a: a first order polynomial for variables :math:`(x,y,z,1)`.
                b: a first order polynomial for variables :math:`(x,y,z,1)`.

    Returns:
        degree 2 poly with the order :math:`(x^2, x*y, x*z, x, y^2, y*z, y, z^2, z, 1)`.
    """

    return stack(
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
# https://github.com/danini/graph-cut-ransac/blob/aae1f40c2e10e31fd2191bac601c53a189673f60/src/pygcransac/
# include/estimators/solver_essential_matrix_five_point_nister.h#L156


def multiply_deg_two_one_poly(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    r"""Multiply two polynomials a and b of degrees two and one [@nister2004efficient].

    Args:
        a: a second degree poly for variables :math:`(x^2, x*y, x*z, x, y^2, y*z, y, z^2, z, 1)`.
        b: a first degree poly for variables :math:`(x y z 1)`.

    Returns:
        a third degree poly for variables,
        :math:`(x^3, y^3, x^2*y, x*y^2, x^2*z, x^2, y^2*z, y^2,
        x*y*z, x*y, x*z^2, x*z, x, y*z^2, y*z, y, z^3, z^2, z, 1)`.
    """

    return stack(
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
def determinant_to_polynomial(A: Tensor) -> Tensor:
    r"""Represent the determinant by the 10th polynomial, used for 5PC solver [@nister2004efficient].

    Args:
        A: Tensor :math:`(*, 3, 13)`.

    Returns:
        a degree 10 poly, representing determinant (Eqn. 14 in the paper).
    """

    cs = zeros(A.shape[0], 11, device=A.device, dtype=A.dtype)
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
