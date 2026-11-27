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

"""Solvers the geometric estimators build on: an affine fit, a null vector, polynomial roots.

The polynomial solvers take one polynomial per row, ``(B, k + 1)`` coefficients for degree ``k``, highest degree
first, and return its ``k`` real roots ``(B, k)``, with a repeated root repeated and ``0.0`` in place of a missing
real root; a zero leading coefficient lowers the degree. The polynomial products of Nister's five-point solver use
the monomial orders their docstrings give. :ref:`Two-view geometry <two-view-conventions>` compares the root layout
with ``numpy.roots``.

Which function to use:

- A 2D affine transformation from point correspondences, by weighted least squares: :func:`find_affine`; with
  outliers, :class:`~kornia.geometry.ransac.RANSAC` with ``model_type="affine"``.
- The null vector of a rank-3 ``(*, 3, 4)`` matrix without an SVD: :func:`null_vector_3x4`.
- The real roots of polynomials: :func:`solve_quadratic`, :func:`solve_cubic`, :func:`solve_quartic`.
- Steps of Nister's five-point solver, which :func:`~kornia.geometry.epipolar.find_essential` runs whole:
  :func:`multiply_deg_one_poly`, :func:`multiply_deg_two_one_poly` and :func:`determinant_to_polynomial`.
"""

from .affine import find_affine
from .homogeneous import null_vector_3x4
from .polynomial_solver import (
    determinant_to_polynomial,
    multiply_deg_one_poly,
    multiply_deg_two_one_poly,
    solve_cubic,
    solve_quadratic,
    solve_quartic,
)

__all__ = [
    "determinant_to_polynomial",
    "find_affine",
    "multiply_deg_one_poly",
    "multiply_deg_two_one_poly",
    "null_vector_3x4",
    "solve_cubic",
    "solve_quadratic",
    "solve_quartic",
]
