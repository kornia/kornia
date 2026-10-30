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

"""Numerical helpers of the epipolar functions: the cross-product matrix and the cofactor matrix."""

import torch


def cross_product_matrix(x: torch.Tensor) -> torch.Tensor:
    r"""Build the skew-symmetric cross-product matrices of 3-vectors with any number of leading dimensions.

    ``S = cross_product_matrix(x)`` takes vectors ``(*, 3)`` and returns the matrices :math:`[x]_\times`,
    ``(*, 3, 3)``, with ``S @ w == cross(x, w)``: the matrix that :func:`essential_from_Rt` multiplies the relative
    rotation by. :func:`~kornia.geometry.conversions.vector_to_skew_symmetric_matrix` builds the same matrix for a
    ``(3,)`` or ``(B, 3)`` vector only.

    Convention:
        - Same matrix as :func:`~kornia.geometry.conversions.vector_to_skew_symmetric_matrix`,
          :math:`[x]_\times w = x \times w`, for any number of leading dims.

    Args:
        x: The input vector to construct the matrix in the shape ``(*, 3)``.

    Returns:
        The skew-symmetric matrix with shape ``(*, 3, 3)``.

    See Also:
        - :func:`~kornia.geometry.conversions.vector_to_skew_symmetric_matrix`: the same matrix, for ``(3,)`` or
          ``(B, 3)`` vectors only.
        - :meth:`~kornia.geometry.liegroup.So3.hat`: the same matrix, as the Lie-algebra element of a rotation
          vector.

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


def matrix_cofactor_tensor(matrix: torch.Tensor) -> torch.Tensor:
    """Cofactor matrix, refer to the numpy doc.

    Args:
        matrix: The input matrix in the shape ``(*, 3, 3)``.

    """
    # Closed form: the rows of the cofactor matrix are the cross products of the other two rows. Unlike the
    # ``det * inv(M)^T`` identity it is exact for singular input (essential matrices are rank 2 by construction),
    # needs no inverse, and traces under graph capture.
    r0, r1, r2 = matrix[..., 0, :], matrix[..., 1, :], matrix[..., 2, :]
    return torch.stack([torch.cross(r1, r2, dim=-1), torch.cross(r2, r0, dim=-1), torch.cross(r0, r1, dim=-1)], dim=-2)
