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

"""Angular pose-error metrics and pose AUC.

The rotation error is the geodesic angle between two rotation matrices; the translation error is the
angle between two translation directions, optionally folded into ``[0, 90]`` degrees to absorb the
sign ambiguity of an essential-matrix translation. :func:`auc_from_errors` summarizes any error array
as the area under its cumulative curve.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor

from kornia.core.check import KORNIA_CHECK, KORNIA_CHECK_IS_TENSOR, KORNIA_CHECK_SHAPE


def angle_error_mat(R1: Tensor, R2: Tensor) -> Tensor:
    r"""Geodesic angle (in degrees) between two rotation matrices.

    The relative rotation :math:`R_1^\top R_2` has trace :math:`1 + 2\cos\theta`, so the geodesic
    angle is :math:`\theta = \arccos\!\big((\mathrm{tr}(R_1^\top R_2) - 1) / 2\big)`.

    Args:
        R1: a rotation matrix of shape :math:`(*, 3, 3)`.
        R2: a rotation matrix of shape :math:`(*, 3, 3)`.

    Return:
        the per-matrix angle in degrees, with shape :math:`(*,)`.

    .. note::
        The gradient is infinite/NaN exactly at :math:`0^\circ` and :math:`180^\circ` (identical or
        opposite rotations), because :math:`\frac{d}{dx}\arccos(x) \to \infty` at :math:`x = \pm 1`.
        This is inherent to every geodesic/angular metric; it only bites if you backpropagate through
        a perfect or exactly-opposite match.

    Example:
        >>> angle_error_mat(torch.eye(3), torch.eye(3))
        tensor(0.)
    """
    KORNIA_CHECK_IS_TENSOR(R1)
    KORNIA_CHECK_IS_TENSOR(R2)
    KORNIA_CHECK_SHAPE(R1, ["*", "3", "3"])
    KORNIA_CHECK_SHAPE(R2, ["*", "3", "3"])

    relative = R1.transpose(-2, -1) @ R2
    trace = relative.diagonal(dim1=-2, dim2=-1).sum(-1)
    cos_theta = ((trace - 1.0) / 2.0).clamp(-1.0, 1.0)
    return torch.rad2deg(cos_theta.acos())


def angle_error_vec(v1: Tensor, v2: Tensor) -> Tensor:
    r"""Angle (in degrees) between two vectors.

    The angle is :math:`\theta = \arccos\!\big((v_1 \cdot v_2) / (\lVert v_1 \rVert \lVert v_2 \rVert)\big)`.

    Args:
        v1: a vector of shape :math:`(*, 3)`.
        v2: a vector of shape :math:`(*, 3)`.

    Return:
        the per-vector angle in degrees, with shape :math:`(*,)`.

    .. note::
        The gradient is infinite/NaN exactly at :math:`0^\circ` and :math:`180^\circ` (identical or
        opposite vectors), because :math:`\frac{d}{dx}\arccos(x) \to \infty` at :math:`x = \pm 1`.
        This is inherent to every geodesic/angular metric; it only bites if you backpropagate through
        a perfect or exactly-opposite match.

    .. note::
        The angle is undefined if either vector is zero, and such entries come back as ``NaN`` rather
        than raising, so that one degenerate sample does not abort a batch. Callers that may hit this
        (a pure-rotation relative pose has zero translation) should mask the result before reducing.

    Example:
        >>> v = torch.tensor([1.0, 0.0, 0.0])
        >>> angle_error_vec(v, v)
        tensor(0.)
    """
    KORNIA_CHECK_IS_TENSOR(v1)
    KORNIA_CHECK_IS_TENSOR(v2)
    KORNIA_CHECK_SHAPE(v1, ["*", "3"])
    KORNIA_CHECK_SHAPE(v2, ["*", "3"])

    dot = (v1 * v2).sum(-1)
    norms = v1.norm(dim=-1) * v2.norm(dim=-1)
    cos_theta = (dot / norms).clamp(-1.0, 1.0)
    return torch.rad2deg(cos_theta.acos())


def translation_ate(t: Tensor, t_gt: Tensor) -> Tensor:
    r"""Absolute translation error (ATE) between two translations.

    Computes the raw Euclidean distance :math:`\lVert t - t_{gt} \rVert_2`. Unlike
    :func:`angle_error_vec`, this keeps the magnitude and is therefore only meaningful when both
    translations share a common metric scale (it is **not** scale-invariant, so it is not suitable
    for raw essential-matrix translations).

    Args:
        t: an estimated translation of shape :math:`(*, 3)`.
        t_gt: a ground-truth translation of the same shape as ``t``.

    Return:
        the per-sample translation error, with shape :math:`(*,)`. An unbatched :math:`(3,)` input is
        treated as a single sample and returns shape :math:`(1,)`.

    .. note::
        Unlike the :func:`angle_error_vec` / :func:`angle_error_mat` angular metrics, this has no
        ``arccos`` singularity: the gradient stays finite even at zero distance, where ``norm``
        returns the subgradient ``0``.

    Example:
        >>> t = torch.tensor([0.0, 0.0, 0.0])
        >>> t_gt = torch.tensor([3.0, 4.0, 0.0])
        >>> translation_ate(t, t_gt)
        tensor([5.])
    """
    KORNIA_CHECK_IS_TENSOR(t)
    KORNIA_CHECK_IS_TENSOR(t_gt)
    KORNIA_CHECK_SHAPE(t, ["*", "3"])
    KORNIA_CHECK_SHAPE(t_gt, ["*", "3"])
    KORNIA_CHECK(t.shape == t_gt.shape, f"t and t_gt shapes must match. Got: {t.shape} and {t_gt.shape}")

    if t.dim() == 1:
        t, t_gt = t[None], t_gt[None]
    return (t - t_gt).norm(dim=-1)


def pose_errors(P: Tensor, P_gt: Tensor, fold_translation: bool = True) -> dict[str, Tensor]:
    r"""Rotation and translation angular error (in degrees) between two relative poses.

    Args:
        P: an estimated relative pose ``[R | t]`` of shape :math:`(3, 4)`, :math:`(4, 4)`, or batched
            :math:`(B, 3, 4)` / :math:`(B, 4, 4)`.
        P_gt: a ground-truth relative pose of the same shape.
        fold_translation: if ``True`` (default), fold the translation error into :math:`[0, 90]` via
            :math:`\min(e, 180 - e)` to absorb the sign ambiguity of an essential-matrix translation.

    Return:
        a dict of per-pose errors of shape :math:`(B,)`: ``"R_err"`` (rotation), ``"t_err"``
        (translation) and ``"max_err"`` (element-wise max of the two).

    .. note::
        A pose with zero translation has an undefined translation direction, so its ``"t_err"`` and
        ``"max_err"`` come back as ``NaN``. Mask those entries out before passing ``"max_err"`` to
        :func:`auc_from_errors`, which otherwise propagates the ``NaN`` into the AUC.

    Example:
        >>> P = torch.eye(4)
        >>> P[0, 3] = 1.0
        >>> errs = pose_errors(P, P)
        >>> errs["R_err"], errs["t_err"]
        (tensor([0.]), tensor([0.]))
    """
    KORNIA_CHECK_IS_TENSOR(P)
    KORNIA_CHECK_IS_TENSOR(P_gt)
    KORNIA_CHECK(P.shape == P_gt.shape, f"P and P_gt shapes must match. Got: {P.shape} and {P_gt.shape}")
    KORNIA_CHECK(
        P.dim() in (2, 3) and P.shape[-2] in (3, 4) and P.shape[-1] == 4,
        f"P must be (3, 4)/(4, 4) or batched. Got: {P.shape}",
    )

    if P.dim() == 2:
        P, P_gt = P[None], P_gt[None]

    r_err = angle_error_mat(P[..., :3, :3], P_gt[..., :3, :3])
    t_err = angle_error_vec(P[..., :3, 3], P_gt[..., :3, 3])
    if fold_translation:
        t_err = torch.minimum(t_err, 180.0 - t_err)
    return {"R_err": r_err, "t_err": t_err, "max_err": torch.maximum(r_err, t_err)}


def auc_from_errors(errors: Tensor, thresholds: float | Sequence[float] = (1, 3, 5, 10)) -> dict[float, float]:
    r"""Area under the cumulative error curve at one or more thresholds.

    The metric is generic: any non-negative error array works. Pose-error metrics (e.g. the
    ``"max_err"`` of :func:`pose_errors`) are one common source, but the thresholds simply need to be
    in the same units as ``errors``.

    Args:
        errors: per-sample error values of shape :math:`(B,)`. Integer and half-precision inputs are
            promoted to the default floating dtype before accumulating.
        thresholds: a single threshold or a sequence of thresholds, in the same units as ``errors``.
            Must be strictly positive. Defaults to ``(1, 3, 5, 10)``.

    Return:
        a dict mapping each threshold to its AUC in :math:`[0, 100]`.

    .. note::
        A sample whose error is exactly equal to a threshold contributes no area at that threshold,
        so a set of errors all equal to ``thr`` scores ``0`` there. This matches the reference
        implementations, but it makes the curve discontinuous right at the threshold.

    Example:
        >>> auc_from_errors(torch.zeros(1), thresholds=5.0)
        {5.0: 100.0}
    """
    KORNIA_CHECK_IS_TENSOR(errors)
    if isinstance(thresholds, (int, float)):
        thresholds = [thresholds]
    thresholds = [float(thr) for thr in thresholds]
    KORNIA_CHECK(len(thresholds) > 0, "thresholds must not be empty.")
    KORNIA_CHECK(all(thr > 0 for thr in thresholds), f"thresholds must be positive. Got: {thresholds}")

    errors = errors.flatten()
    # The AUC is summarized as Python floats, so accumulate in at least single precision: an integer
    # dtype would truncate the threshold, and half precision loses integer exactness in ``arange``.
    if errors.dtype not in (torch.float32, torch.float64):
        errors = errors.to(torch.get_default_dtype())
    errors = errors.sort().values
    n = errors.numel()
    recall = torch.arange(1, n + 1, device=errors.device, dtype=errors.dtype) / n
    errors = torch.cat([errors.new_zeros(1), errors])
    recall = torch.cat([recall.new_zeros(1), recall])

    aucs: dict[float, float] = {}
    for thr in thresholds:
        # Index of the first error at or past the threshold: everything before it is kept, and the
        # curve is closed off with a horizontal segment out to ``thr``. A positive threshold always
        # lands past the prepended zero, so this slice holds at least one point.
        last = int(torch.searchsorted(errors, errors.new_tensor(thr)).item())
        recall_below = torch.cat([recall[:last], recall[last - 1 : last]])
        errors_below = torch.cat([errors[:last], errors.new_tensor([thr])])
        area = torch.trapezoid(recall_below, x=errors_below)
        aucs[thr] = (area / thr).item() * 100.0
    return aucs
