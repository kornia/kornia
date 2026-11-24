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


"""Estimate rigid and similarity transforms from paired 3D points.

:func:`find_similarity_transform3d` returns a batched :class:`SimilarityTransform3d` for 3D correspondences.
"""

from typing import NamedTuple

import torch

from kornia.core.check import KORNIA_CHECK_SHAPE
from kornia.core.utils import _at_least_float32, _autocast_disabled, _symmetric_eigenvector, _torch_svd_right
from kornia.geometry.conversions import quaternion_to_rotation_matrix

__all__ = ["SimilarityTransform3d", "find_similarity_transform3d"]


class SimilarityTransform3d(NamedTuple):
    """The source-to-destination rotation, translation, scale and validity of each 3D fit.

    See Also:
        - :func:`find_similarity_transform3d`: estimate these parameters from paired points.

    """

    rotation: torch.Tensor
    translation: torch.Tensor
    scale: torch.Tensor
    valid: torch.Tensor


def _center_points(
    points: torch.Tensor, weights: torch.Tensor, anchor_index: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    # Subtract a point before reducing so rounding the absolute centroid does not
    # change the shape of a small cloud with a distant world origin.
    anchor = points.gather(1, anchor_index)
    local = points - anchor
    center = (weights[..., None] * local).sum(dim=1, keepdim=True)
    return local - center, (anchor + center).squeeze(1)


def find_similarity_transform3d(
    points_src: torch.Tensor,
    points_dst: torch.Tensor,
    weights: torch.Tensor | None = None,
    estimate_scale: bool = True,
) -> SimilarityTransform3d:
    r"""Fit a least-squares similarity from paired 3D points, with unit scale for a rigid fit.

    The objective is the weighted sum of squared residuals
    :math:`\sum_i w_i \|s R x_i + t - y_i\|^2`, with a proper rotation and positive scale.
    Set ``estimate_scale=False`` to fit only rotation and translation. This is the weighted form of
    Umeyama's least-squares fit. The rotation uses Horn's quaternion eigenproblem.

    Convention:
        - Points are paired in their supplied order. The transform maps column vectors from source to destination.
          For points stored as rows, apply ``scale[:, None, None] * (points_src @ rotation.mT) + translation[:, None]``.
        - Zero weights exclude correspondences, including padding. Each fit needs a numerically unique optimum
          rotation. Noncollinear planar points can determine it. A collinear cloud or an all-zero weight vector
          cannot. Invalid rows have identity rotation, zero translation and unit scale, with zero gradients.
        - Outputs use the promoted input floating dtype. Half and bfloat16 arithmetic is evaluated in float32
          and rounded on return. An external autocast context does not lower the working precision.

    Args:
        points_src: Finite source points ``(B, N, 3)``, with at least three points.
        points_dst: Finite destination points of the same shape and device.
        weights: Optional finite, nonnegative floating weights ``(B, N)`` on the points' device.
        estimate_scale: Fit a uniform scale. If false, the scale is one.

    Returns:
        Rotation ``(B, 3, 3)``, translation ``(B, 3)``, scale ``(B,)`` and boolean validity ``(B,)``.

    See Also:
        - :class:`SimilarityTransform3d`: the result fields and their shapes.
        - :func:`~kornia.geometry.linalg.transform_points`: apply a homogeneous transformation matrix.

    Example:
        >>> src = torch.tensor([[[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]]])
        >>> out = find_similarity_transform3d(src, 2 * src + 1)
        >>> out.valid.tolist()
        [True]
        >>> out.scale.round().tolist()
        [2.0]

    """
    KORNIA_CHECK_SHAPE(points_src, ["B", "N", "3"])
    KORNIA_CHECK_SHAPE(points_dst, ["B", "N", "3"])
    if points_src.shape != points_dst.shape or points_src.shape[1] < 3:
        raise ValueError("Point clouds must have the same shape and at least three correspondences.")
    if points_src.device != points_dst.device or not (
        points_src.is_floating_point() and points_dst.is_floating_point()
    ):
        raise ValueError("Point clouds must be floating tensors on the same device.")

    dtype = torch.promote_types(points_src.dtype, points_dst.dtype)
    if weights is not None:
        KORNIA_CHECK_SHAPE(weights, ["B", "N"])
        if (
            weights.shape != points_src.shape[:2]
            or weights.device != points_src.device
            or not weights.is_floating_point()
        ):
            raise ValueError("Weights must be floating, with matching batch and correspondence dimensions and device.")
        dtype = torch.promote_types(dtype, weights.dtype)
    src = _at_least_float32(points_src.to(dtype))
    dst = points_dst.to(src)
    weights = src.new_ones(src.shape[:2]) if weights is None else weights.to(src)

    with _autocast_disabled(src.device.type):
        largest, anchor = weights.detach().max(dim=1, keepdim=True)
        scaled = weights / torch.where(largest > 0, largest, 1)
        total = scaled.sum(dim=1, keepdim=True)
        relative_weights = scaled / torch.where(total > 0, total, 1)
        anchor_index = anchor[..., None].expand(-1, 1, 3)
        x, src_center = _center_points(src, relative_weights, anchor_index)
        y, dst_center = _center_points(dst, relative_weights, anchor_index)
        covariance = (relative_weights[..., None] * y).mT @ x

        # The leading eigenvector maximizes trace(R(q)^T covariance).
        # https://people.csail.mit.edu/bkph/papers/Absolute_Orientation.pdf
        magnitude = covariance.detach().abs().amax(dim=(-2, -1), keepdim=True)
        linear = covariance / torch.where(magnitude > 0, magnitude, 1)
        trace = linear.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
        skew = linear - linear.mT
        cross = torch.stack((skew[:, 2, 1], skew[:, 0, 2], skew[:, 1, 0]), dim=-1)
        lower = linear + linear.mT - torch.diag_embed(trace[:, None].expand(-1, 3))
        horn = torch.cat(
            (torch.cat((trace[:, None], cross), dim=-1)[:, None], torch.cat((cross[:, :, None], lower), dim=-1)), dim=1
        )
        identity = torch.eye(4, device=src.device, dtype=src.dtype)
        detached = horn.detach()
        # A positive shift preserves eigenvectors and lets SVD compute the
        # leading eigenpair on backends without a symmetric eigensolver.
        shift = detached.norm(dim=(-2, -1), keepdim=True)
        eigenvalues, eigenvectors = _torch_svd_right(detached + shift * identity)
        valid = eigenvalues[:, 0] - eigenvalues[:, 1] > 8 * torch.finfo(src.dtype).eps
        variance = (relative_weights * x.square().sum(dim=-1)).sum(dim=1)
        if estimate_scale:
            valid = valid & (variance > 0)
        # Suppress an undefined derivative before constructing the eigenvector
        # projector. Lower eigenvalues may coincide in otherwise valid fits.
        safe_values = torch.where(valid[:, None], eigenvalues, src.new_tensor([1, 0, 0, 0]))
        rotation = quaternion_to_rotation_matrix(_symmetric_eigenvector(horn, safe_values, eigenvectors, 0))
        scale = (
            (rotation * covariance).sum(dim=(-2, -1)) / torch.where(valid, variance, 1)
            if estimate_scale
            else torch.ones_like(variance)
        )
        translation = dst_center - scale[:, None] * (rotation @ src_center.unsqueeze(-1)).squeeze(-1)

    return SimilarityTransform3d(
        torch.where(valid[:, None, None], rotation, identity[:3, :3]).to(dtype),
        torch.where(valid[:, None], translation, 0).to(dtype),
        torch.where(valid, scale, 1).to(dtype),
        valid,
    )
