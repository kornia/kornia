# LICENSE HEADER MANAGED BY add-license-header
#
# Copyright 2018 Kornia Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

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


T_deg1=torch.zeros(16,10); T_deg1[0*4+0,0]=1; T_deg1[0*4+1,1]=1; T_deg1[1*4+0,1]=1; T_deg1[0*4+2,2]=1;
T_deg1[2*4+0,2]=1; T_deg1[0*4+3,3]=1; T_deg1[3*4+0,3]=1; T_deg1[1*4+1,4]=1; T_deg1[1*4+2,5]=1;
T_deg1[2*4+1,5]=1; T_deg1[1*4+3,6]=1; T_deg1[3*4+1,6]=1; T_deg1[2*4+2,7]=1; T_deg1[2*4+3,8]=1;
T_deg1[3*4+2,8]=1; T_deg1[3*4+3,9]=1

def multiply_deg_one_poly(a: torch.Tensor, b: torch.Tensor, T_matrix: torch.Tensor = T_deg1) -> torch.Tensor:
    r"""Multiply two polynomials of the first order [@nister2004efficient].

    Args:
        a: a first order polynomial for variables :math:`(x,y,z,1)`.
        b: a first order polynomial for variables :math:`(x,y,z,1)`.

    Returns:
        degree 2 poly with the order :math:`(x^2, x*y, x*z, x, y^2, y*z, y, z^2, z, 1)`.

    """

    if T_matrix.device != a.device: T_matrix = T_matrix.to(a.device)
    return (a.unsqueeze(2) * b.unsqueeze(1)).flatten(start_dim=-2) @ T_matrix


# Reference
# https://github.com/danini/graph-cut-ransac/blob/aae1f40c2e10e31fd2191bac601c53a189673f60/src/pygcransac/
# include/estimators/solver_essential_matrix_five_point_nister.h#L156

T_deg2 = torch.zeros(40, 20)
T_deg2[0*4+0,0]=1; T_deg2[4*4+1,1]=1; T_deg2[0*4+1,2]=1; T_deg2[1*4+0,2]=1; T_deg2[1*4+1,3]=1; T_deg2[4*4+0,3]=1; T_deg2[0*4+2,4]=1;
T_deg2[2*4+0,4]=1; T_deg2[0*4+3,5]=1; T_deg2[3*4+0,5]=1; T_deg2[4*4+2,6]=1; T_deg2[5*4+1,6]=1; T_deg2[4*4+3,7]=1; T_deg2[6*4+1,7]=1;
T_deg2[1*4+2,8]=1; T_deg2[2*4+1,8]=1; T_deg2[5*4+0,8]=1; T_deg2[1*4+3,9]=1; T_deg2[3*4+1,9]=1; T_deg2[6*4+0,9]=1; T_deg2[2*4+2,10]=1;
T_deg2[7*4+0,10]=1; T_deg2[2*4+3,11]=1; T_deg2[3*4+2,11]=1; T_deg2[8*4+0,11]=1; T_deg2[3*4+3,12]=1; T_deg2[9*4+0,12]=1; T_deg2[5*4+2,13]=1;
T_deg2[7*4+1,13]=1; T_deg2[5*4+3,14]=1; T_deg2[6*4+2,14]=1; T_deg2[8*4+1,14]=1; T_deg2[6*4+3,15]=1; T_deg2[9*4+1,15]=1; T_deg2[7*4+2,16]=1;
T_deg2[7*4+3,17]=1; T_deg2[8*4+2,17]=1; T_deg2[8*4+3,18]=1; T_deg2[9*4+2,18]=1; T_deg2[9*4+3,19]=1

def multiply_deg_two_one_poly(a: torch.Tensor, b: torch.Tensor, T_matrix: torch.Tensor = T_deg2) -> torch.Tensor:
    r"""Multiply two polynomials a and b of degrees two and one [@nister2004efficient].

    Args:
        a: a second degree poly for variables :math:`(x^2, x*y, x*z, x, y^2, y*z, y, z^2, z, 1)`.
        b: a first degree poly for variables :math:`(x y z 1)`.

    Returns:
        a third degree poly for variables,
        :math:`(x^3, y^3, x^2*y, x*y^2, x^2*z, x^2, y^2*z, y^2,
        x*y*z, x*y, x*z^2, x*z, x, y*z^2, y*z, y, z^3, z^2, z, 1)`.

    """

    if T_matrix.device != a.device:
        T_matrix = T_matrix.to(a.device)
    product_basis = a.unsqueeze(2) * b.unsqueeze(1)
    product_vector = product_basis.flatten(start_dim=-2)
    return product_vector @ T_matrix

# Compute degree 10 poly representing determinant (equation 14 in the paper)
# https://github.com/danini/graph-cut-ransac/blob/aae1f40c2e10e31fd2191bac601c53a189673f60/src/pygcransac/
# include/estimators/solver_essential_matrix_five_point_nister.h#L368C5-L368C82

multiplication_indices = torch.tensor([
    [12, 16, 33],
    [12, 20, 29],
    [3, 33, 25],
    [7, 29, 25],
    [3, 20, 38],
    [7, 16, 38],
    [11, 16, 33],
    [11, 20, 29],
    [12, 15, 33],
    [12, 16, 32],
    [12, 19, 29],
    [12, 20, 28],
    [2, 33, 25],
    [3, 32, 25],
    [3, 33, 24],
    [6, 29, 25],
    [7, 28, 25],
    [7, 29, 24],
    [2, 20, 38],
    [3, 19, 38],
    [3, 20, 37],
    [6, 16, 38],
    [7, 15, 38],
    [7, 16, 37],
    [10, 16, 33],
    [10, 20, 29],
    [11, 15, 33],
    [11, 16, 32],
    [11, 19, 29],
    [11, 20, 28],
    [14, 12, 33],
    [12, 15, 32],
    [12, 16, 31],
    [12, 18, 29],
    [12, 19, 28],
    [12, 20, 27],
    [1, 33, 25],
    [2, 32, 25],
    [2, 33, 24],
    [3, 31, 25],
    [3, 32, 24],
    [3, 33, 23],
    [5, 29, 25],
    [6, 28, 25],
    [6, 29, 24],
    [7, 27, 25],
    [7, 28, 24],
    [7, 29, 23],
    [1, 20, 38],
    [2, 19, 38],
    [2, 20, 37],
    [3, 18, 38],
    [3, 19, 37],
    [3, 20, 36],
    [5, 16, 38],
    [6, 15, 38],
    [6, 16, 37],
    [7, 14, 38],
    [7, 15, 37],
    [7, 16, 36],
    [3, 20, 35],
    [3, 22, 33],
    [7, 16, 35],
    [7, 22, 29],
    [9, 16, 33],
    [9, 20, 29],
    [10, 15, 33],
    [10, 16, 32],
    [10, 19, 29],
    [10, 20, 28],
    [13, 12, 33],
    [11, 14, 33],
    [11, 15, 32],
    [11, 16, 31],
    [11, 18, 29],
    [11, 19, 28],
    [11, 20, 27],
    [14, 12, 32],
    [12, 15, 31],
    [12, 16, 30],
    [12, 17, 29],
    [12, 18, 28],
    [12, 19, 27],
    [12, 20, 26],
    [0, 33, 25],
    [1, 32, 25],
    [1, 33, 24],
    [2, 31, 25],
    [2, 32, 24],
    [2, 33, 23],
    [3, 30, 25],
    [3, 31, 24],
    [3, 32, 23],
    [4, 29, 25],
    [5, 28, 25],
    [5, 29, 24],
    [6, 27, 25],
    [6, 28, 24],
    [6, 29, 23],
    [7, 26, 25],
    [7, 27, 24],
    [7, 28, 23],
    [0, 20, 38],
    [1, 19, 38],
    [1, 20, 37],
    [2, 18, 38],
    [2, 19, 37],
    [2, 20, 36],
    [3, 17, 38],
    [3, 18, 37],
    [3, 19, 36],
    [4, 16, 38],
    [5, 15, 38],
    [5, 16, 37],
    [6, 14, 38],
    [6, 15, 37],
    [6, 16, 36],
    [7, 13, 38],
    [7, 14, 37],
    [7, 15, 36],
    [2, 20, 35],
    [2, 22, 33],
    [3, 19, 35],
    [3, 20, 34],
    [3, 21, 33],
    [3, 22, 32],
    [6, 16, 35],
    [6, 22, 29],
    [7, 15, 35],
    [7, 16, 34],
    [7, 21, 29],
    [7, 22, 28],
    [8, 16, 33],
    [8, 20, 29],
    [9, 15, 33],
    [9, 16, 32],
    [9, 19, 29],
    [9, 20, 28],
    [10, 14, 33],
    [10, 15, 32],
    [10, 16, 31],
    [10, 18, 29],
    [10, 19, 28],
    [10, 20, 27],
    [13, 11, 33],
    [13, 12, 32],
    [11, 14, 32],
    [11, 15, 31],
    [11, 16, 30],
    [11, 17, 29],
    [11, 18, 28],
    [11, 19, 27],
    [11, 20, 26],
    [14, 12, 31],
    [12, 15, 30],
    [12, 17, 28],
    [12, 18, 27],
    [12, 19, 26],
    [0, 32, 25],
    [0, 33, 24],
    [1, 31, 25],
    [1, 32, 24],
    [1, 33, 23],
    [2, 30, 25],
    [2, 31, 24],
    [2, 32, 23],
    [3, 30, 24],
    [3, 31, 23],
    [4, 28, 25],
    [4, 29, 24],
    [5, 27, 25],
    [5, 28, 24],
    [5, 29, 23],
    [6, 26, 25],
    [6, 27, 24],
    [6, 28, 23],
    [7, 26, 24],
    [7, 27, 23],
    [0, 19, 38],
    [0, 20, 37],
    [1, 18, 38],
    [1, 19, 37],
    [1, 20, 36],
    [2, 17, 38],
    [2, 18, 37],
    [2, 19, 36],
    [3, 17, 37],
    [3, 18, 36],
    [4, 15, 38],
    [4, 16, 37],
    [5, 14, 38],
    [5, 15, 37],
    [5, 16, 36],
    [6, 13, 38],
    [6, 14, 37],
    [6, 15, 36],
    [7, 13, 37],
    [7, 14, 36],
    [1, 20, 35],
    [1, 22, 33],
    [2, 19, 35],
    [2, 20, 34],
    [2, 21, 33],
    [2, 22, 32],
    [3, 18, 35],
    [3, 19, 34],
    [3, 21, 32],
    [3, 22, 31],
    [5, 16, 35],
    [5, 22, 29],
    [6, 15, 35],
    [6, 16, 34],
    [6, 21, 29],
    [6, 22, 28],
    [7, 14, 35],
    [7, 15, 34],
    [7, 21, 28],
    [7, 22, 27],
    [8, 15, 33],
    [8, 16, 32],
    [8, 19, 29],
    [8, 20, 28],
    [9, 14, 33],
    [9, 15, 32],
    [9, 16, 31],
    [9, 18, 29],
    [9, 19, 28],
    [9, 20, 27],
    [10, 13, 33],
    [10, 14, 32],
    [10, 15, 31],
    [10, 16, 30],
    [10, 17, 29],
    [10, 18, 28],
    [10, 19, 27],
    [10, 20, 26],
    [13, 11, 32],
    [13, 12, 31],
    [11, 14, 31],
    [11, 15, 30],
    [11, 17, 28],
    [11, 18, 27],
    [11, 19, 26],
    [14, 12, 30],
    [12, 17, 27],
    [12, 18, 26],
    [0, 31, 25],
    [0, 32, 24],
    [0, 33, 23],
    [1, 30, 25],
    [1, 31, 24],
    [1, 32, 23],
    [2, 30, 24],
    [2, 31, 23],
    [3, 30, 23],
    [4, 27, 25],
    [4, 28, 24],
    [4, 29, 23],
    [5, 26, 25],
    [5, 27, 24],
    [5, 28, 23],
    [6, 26, 24],
    [6, 27, 23],
    [7, 26, 23],
    [0, 18, 38],
    [0, 19, 37],
    [0, 20, 36],
    [1, 17, 38],
    [1, 18, 37],
    [1, 19, 36],
    [2, 17, 37],
    [2, 18, 36],
    [3, 17, 36],
    [4, 14, 38],
    [4, 15, 37],
    [4, 16, 36],
    [5, 13, 38],
    [5, 14, 37],
    [5, 15, 36],
    [6, 13, 37],
    [6, 14, 36],
    [7, 13, 36],
    [0, 20, 35],
    [0, 22, 33],
    [1, 19, 35],
    [1, 20, 34],
    [1, 21, 33],
    [1, 22, 32],
    [2, 18, 35],
    [2, 19, 34],
    [2, 21, 32],
    [2, 22, 31],
    [3, 17, 35],
    [3, 18, 34],
    [3, 21, 31],
    [3, 22, 30],
    [4, 16, 35],
    [4, 22, 29],
    [5, 15, 35],
    [5, 16, 34],
    [5, 21, 29],
    [5, 22, 28],
    [6, 14, 35],
    [6, 15, 34],
    [6, 21, 28],
    [6, 22, 27],
    [7, 13, 35],
    [7, 14, 34],
    [7, 21, 27],
    [7, 22, 26],
    [8, 14, 33],
    [8, 15, 32],
    [8, 16, 31],
    [8, 18, 29],
    [8, 19, 28],
    [8, 20, 27],
    [9, 13, 33],
    [9, 14, 32],
    [9, 15, 31],
    [9, 16, 30],
    [9, 17, 29],
    [9, 18, 28],
    [9, 19, 27],
    [9, 20, 26],
    [10, 13, 32],
    [10, 14, 31],
    [10, 15, 30],
    [10, 17, 28],
    [10, 18, 27],
    [10, 19, 26],
    [13, 11, 31],
    [13, 12, 30],
    [11, 14, 30],
    [11, 17, 27],
    [11, 18, 26],
    [12, 17, 26],
    [0, 30, 25],
    [0, 31, 24],
    [0, 32, 23],
    [1, 30, 24],
    [1, 31, 23],
    [2, 30, 23],
    [4, 26, 25],
    [4, 27, 24],
    [4, 28, 23],
    [5, 26, 24],
    [5, 27, 23],
    [6, 26, 23],
    [0, 17, 38],
    [0, 18, 37],
    [0, 19, 36],
    [1, 17, 37],
    [1, 18, 36],
    [2, 17, 36],
    [4, 13, 38],
    [4, 14, 37],
    [4, 15, 36],
    [5, 13, 37],
    [5, 14, 36],
    [6, 13, 36],
    [0, 19, 35],
    [0, 20, 34],
    [0, 21, 33],
    [0, 22, 32],
    [1, 18, 35],
    [1, 19, 34],
    [1, 21, 32],
    [1, 22, 31],
    [2, 17, 35],
    [2, 18, 34],
    [2, 21, 31],
    [2, 22, 30],
    [3, 17, 34],
    [3, 21, 30],
    [4, 15, 35],
    [4, 16, 34],
    [4, 21, 29],
    [4, 22, 28],
    [5, 14, 35],
    [5, 15, 34],
    [5, 21, 28],
    [5, 22, 27],
    [6, 13, 35],
    [6, 14, 34],
    [6, 21, 27],
    [6, 22, 26],
    [7, 13, 34],
    [7, 21, 26],
    [8, 13, 33],
    [8, 14, 32],
    [8, 15, 31],
    [8, 16, 30],
    [8, 17, 29],
    [8, 18, 28],
    [8, 19, 27],
    [8, 20, 26],
    [9, 13, 32],
    [9, 14, 31],
    [9, 15, 30],
    [9, 17, 28],
    [9, 18, 27],
    [9, 19, 26],
    [10, 13, 31],
    [10, 14, 30],
    [10, 17, 27],
    [10, 18, 26],
    [13, 11, 30],
    [11, 17, 26],
    [0, 30, 24],
    [0, 31, 23],
    [1, 30, 23],
    [4, 26, 24],
    [4, 27, 23],
    [5, 26, 23],
    [0, 17, 37],
    [0, 18, 36],
    [1, 17, 36],
    [4, 13, 37],
    [4, 14, 36],
    [5, 13, 36],
    [0, 18, 35],
    [0, 19, 34],
    [0, 21, 32],
    [0, 22, 31],
    [1, 17, 35],
    [1, 18, 34],
    [1, 21, 31],
    [1, 22, 30],
    [2, 17, 34],
    [2, 21, 30],
    [4, 14, 35],
    [4, 15, 34],
    [4, 21, 28],
    [4, 22, 27],
    [5, 13, 35],
    [5, 14, 34],
    [5, 21, 27],
    [5, 22, 26],
    [6, 13, 34],
    [6, 21, 26],
    [8, 13, 32],
    [8, 14, 31],
    [8, 15, 30],
    [8, 17, 28],
    [8, 18, 27],
    [8, 19, 26],
    [9, 13, 31],
    [9, 14, 30],
    [9, 17, 27],
    [9, 18, 26],
    [10, 13, 30],
    [10, 17, 26],
    [0, 30, 23],
    [4, 26, 23],
    [0, 17, 36],
    [4, 13, 36],
    [0, 17, 35],
    [0, 18, 34],
    [0, 21, 31],
    [0, 22, 30],
    [1, 17, 34],
    [1, 21, 30],
    [4, 13, 35],
    [4, 14, 34],
    [4, 21, 27],
    [4, 22, 26],
    [5, 13, 34],
    [5, 21, 26],
    [8, 13, 31],
    [8, 14, 30],
    [8, 17, 27],
    [8, 18, 26],
    [9, 13, 30],
    [9, 17, 26],
    [0, 17, 34],
    [0, 21, 30],
    [4, 13, 34],
    [4, 21, 26],
    [8, 13, 30],
    [8, 17, 26]
], dtype=torch.int64)


signs = torch.tensor([
    1.0, -1.0, -1.0, 1.0, 1.0, -1.0, 1.0, -1.0, 1.0, 1.0, -1.0, -1.0,
    -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0,
    1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0,
    -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
    1.0, 1.0, 1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0,
    1.0, -1.0, -1.0, 1.0, 1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 1.0,
    1.0, 1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0,
    -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0,
    1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
    1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0,
    1.0, -1.0, 1.0, 1.0, -1.0, -1.0, -1.0, 1.0, -1.0, -1.0, 1.0, 1.0,
    1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0,
    1.0, 1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0, 1.0, 1.0, -1.0,
    -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0,
    1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
    1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0,
    -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 1.0, -1.0, 1.0, 1.0, -1.0, -1.0,
    1.0, 1.0, -1.0, -1.0, -1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0, -1.0,
    1.0, 1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0,
    1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0,
    -1.0, -1.0, -1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0,
    -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
    1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0,
    -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 1.0, -1.0, 1.0, 1.0, -1.0, -1.0,
    1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0, -1.0, -1.0, 1.0, -1.0, -1.0,
    1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0,
    1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0,
    1.0, 1.0, 1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0,
    -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
    1.0, 1.0, 1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0,
    1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0, -1.0,
    1.0, -1.0, -1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0, -1.0,
    1.0, 1.0, -1.0, 1.0, 1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0,
    1.0, 1.0, 1.0, -1.0, -1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0, -1.0,
    -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0,
    1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0, -1.0, -1.0, -1.0,
    1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0, 1.0, 1.0, 1.0, 1.0, -1.0,
    -1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0,
    1.0, 1.0, -1.0, -1.0, 1.0, -1.0, -1.0, -1.0, 1.0, 1.0, -1.0, 1.0,
    1.0, 1.0, -1.0, -1.0, 1.0, -1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0
], dtype=torch.float32)


coefficient_map = torch.tensor([
    0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
    3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
    3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
    3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
    3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
    3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
    4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
    4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
    4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
    4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
    4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
    4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
    4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5,
    5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
    5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
    5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
    5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
    5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
    5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
    5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6,
    6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
    6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
    6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
    6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
    6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
    6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
    7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
    7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
    7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
    7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
    7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
    8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
    8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
    8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
    9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9,
    9, 9, 9, 9, 9, 9, 10, 10, 10, 10, 10, 10
], dtype=torch.int64)


def determinant_to_polynomial(A: Tensor, multiplication_indices: Tensor = multiplication_indices, signs: Tensor = signs, coefficient_map: Tensor = coefficient_map) -> Tensor:
    r"""Represent the determinant by the 10th polynomial, used for 5PC solver [@nister2004efficient].

    Args:
        A: Tensor :math:`(*, 3, 13)`.

    Returns:
        a degree 10 poly, representing determinant (Eqn. 14 in the paper).

    """
    B, device, dtype = A.shape[0], A.device, A.dtype
    
    multiplication_indices = multiplication_indices.to(device)
    signs = signs.to(device, dtype)
    coefficient_map = coefficient_map.to(device)

    A_flat = A.view(B, -1)
    gathered_values = A_flat[:, multiplication_indices]
    products = torch.prod(gathered_values, dim=-1)
    signed_products = products * signs

    cs = torch.zeros(B, 11, device=device, dtype=dtype)
    batch_coefficient_map = coefficient_map.repeat(B, 1)
    cs.scatter_add_(dim=1, index=batch_coefficient_map, src=signed_products)
    return cs
