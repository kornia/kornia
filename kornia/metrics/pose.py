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

from kornia.core.check import (
    KORNIA_CHECK,
    KORNIA_CHECK_IS_TENSOR,
    KORNIA_CHECK_SAME_SHAPE,
    KORNIA_CHECK_SHAPE,
)


def _angle_deg(sin_theta: Tensor, cos_theta: Tensor) -> Tensor:
    """Angle in degrees from a non-negative sine and a cosine, the same in eager, compiled and ONNX graphs."""
    # -atan2(-sin, cos) is atan2(sin, cos). At sin = +0 and cos < 0 the ONNX export's atan2 returns -pi instead of
    # pi; with -0 it returns -pi in both, which the negation turns into pi.
    theta = torch.rad2deg(-torch.atan2(-sin_theta, cos_theta))
    # The ONNX export's atan2 also maps NaN to 0, so a NaN input (a zero vector) would read as a perfect match.
    nan = sin_theta + cos_theta
    return torch.where(torch.isnan(nan), nan, theta)


def angle_error_mat(R1: Tensor, R2: Tensor) -> Tensor:
    r"""Geodesic angle (in degrees) between two rotation matrices.

    The relative rotation :math:`R = R_1^\top R_2` has trace :math:`1 + 2\cos\theta` and skew part
    :math:`(R - R^\top) / 2 = \sin\theta\,[\mathbf{n}]_\times` for its unit axis :math:`\mathbf{n}`, so the
    geodesic angle is :math:`\theta = \operatorname{atan2}(\sin\theta, \cos\theta)`. Reading
    :math:`\sin\theta` from the skew part keeps the digits of small angles that
    :math:`\arccos\!\big((\mathrm{tr}\,R - 1) / 2\big)` loses next to :math:`1`: in float32 the result
    stays within about :math:`2 \cdot 10^{-5}` degrees of the exact angle over the whole range.

    Convention:
        See the Convention block of :func:`~kornia.metrics.pose_errors`, whose ``"R_err"`` this is: one angle in
        degrees per pair, symmetric in ``R1`` and ``R2``. An unbatched pair gives a 0-d tensor. The inputs are not
        checked to be rotations.

    Args:
        R1: a rotation matrix of shape :math:`(*, 3, 3)`.
        R2: a rotation matrix of shape :math:`(*, 3, 3)`.

    Return:
        the per-matrix angle in degrees, with shape :math:`(*,)`.

    .. note::
        The angle has a kink at exactly :math:`0^\circ` and :math:`180^\circ` (identical or opposite
        rotations). The gradient there is the subgradient :math:`0` that ``norm`` returns at a zero
        vector, so backpropagating through a perfect or exactly-opposite match stays finite.

    Example:
        >>> angle_error_mat(torch.eye(3), torch.eye(3))
        tensor(0.)
    """
    KORNIA_CHECK_IS_TENSOR(R1)
    KORNIA_CHECK_IS_TENSOR(R2)
    KORNIA_CHECK_SHAPE(R1, ["*", "3", "3"])
    KORNIA_CHECK_SHAPE(R2, ["*", "3", "3"])
    KORNIA_CHECK_SAME_SHAPE(R1, R2)

    relative = R1.transpose(-2, -1) @ R2
    trace = relative.diagonal(dim1=-2, dim2=-1).sum(-1)
    cos_theta = (trace - 1.0) / 2.0
    # (R - R^T) / 2 = sin(theta) [n]_x, so its three independent entries have norm sin(theta).
    skew = relative - relative.transpose(-2, -1)
    sin_theta = 0.5 * torch.stack((skew[..., 2, 1], skew[..., 0, 2], skew[..., 1, 0]), dim=-1).norm(dim=-1)
    return _angle_deg(sin_theta, cos_theta)


def angle_error_vec(v1: Tensor, v2: Tensor) -> Tensor:
    r"""Angle (in degrees) between two vectors.

    With the unit vectors :math:`\hat v_1` and :math:`\hat v_2`, the angle is
    :math:`\theta = \operatorname{atan2}(\lVert \hat v_1 \times \hat v_2 \rVert, \hat v_1 \cdot \hat v_2)`.
    Reading :math:`\sin\theta` from the cross product keeps the digits of small angles that
    :math:`\arccos(\hat v_1 \cdot \hat v_2)` loses next to :math:`1`: in float32 the result stays within
    about :math:`2 \cdot 10^{-5}` degrees of the exact angle over the whole range.

    Convention:
        See the Convention block of :func:`~kornia.metrics.pose_errors`, whose ``"t_err"`` is this angle before
        folding: one angle in degrees per pair, symmetric in ``v1`` and ``v2`` up to roundoff. An unbatched pair gives
        a 0-d tensor. Only the directions are compared, so the vectors need not have unit length. The angle is not
        folded: opposite vectors give 180.

    Args:
        v1: a vector of shape :math:`(*, 3)`.
        v2: a vector of shape :math:`(*, 3)`.

    Return:
        the per-vector angle in degrees, with shape :math:`(*,)`.

    .. note::
        The angle has a kink at exactly :math:`0^\circ` and :math:`180^\circ` (identical or opposite
        vectors). The gradient there is the subgradient :math:`0` that ``norm`` returns at a zero
        vector, so backpropagating through a perfect or exactly-opposite match stays finite.

    .. note::
        A zero-length vector gives ``NaN`` rather than raising, since the angle is undefined there.
        Mask those entries before reducing.

    Example:
        >>> v = torch.tensor([1.0, 0.0, 0.0])
        >>> angle_error_vec(v, v)
        tensor(0.)
    """
    KORNIA_CHECK_IS_TENSOR(v1)
    KORNIA_CHECK_IS_TENSOR(v2)
    KORNIA_CHECK_SHAPE(v1, ["*", "3"])
    KORNIA_CHECK_SHAPE(v2, ["*", "3"])
    KORNIA_CHECK_SAME_SHAPE(v1, v2)

    # atan2 does not depend on the length of either vector, so scaling each one by its largest entry is enough.
    # Unlike a norm, that cannot overflow or underflow, and a zero vector still gives 0 / 0 = NaN.
    v1 = v1 / v1.abs().amax(dim=-1, keepdim=True)
    v2 = v2 / v2.abs().amax(dim=-1, keepdim=True)
    sin_theta = torch.linalg.cross(v1, v2, dim=-1).norm(dim=-1)
    cos_theta = (v1 * v2).sum(-1)
    return _angle_deg(sin_theta, cos_theta)


def translation_ate(t: Tensor, t_gt: Tensor) -> Tensor:
    r"""Absolute translation error (ATE) between two translations.

    Computes the raw Euclidean distance :math:`\lVert t - t_{gt} \rVert_2`. Unlike
    :func:`angle_error_vec`, this keeps the magnitude and is therefore only meaningful when both
    translations share a common metric scale (it is **not** scale-invariant, so it is not suitable
    for raw essential-matrix translations).

    Convention:
        See the Convention block of :func:`~kornia.metrics.pose_errors`, whose ``"t_err"`` compares translation
        directions only; this function keeps the magnitude. The result is one distance per sample, in the units of
        the translations, with no alignment: a trajectory shifted by a constant keeps that offset at every pose.

    Args:
        t: an estimated translation of shape :math:`(*, 3)`.
        t_gt: a ground-truth translation of the same shape as ``t``.

    Return:
        the per-sample translation error, with shape :math:`(*,)`. An unbatched :math:`(3,)` input is
        treated as a single sample and returns shape :math:`(1,)`.

    .. note::
        The gradient stays finite even at zero distance, where ``norm`` returns the subgradient ``0``.

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

    Convention:
        - Every error is an angle in **degrees**, one per pose and never averaged over the batch; an unbatched pose
          gives shape :math:`(1,)`, where :func:`~kornia.metrics.angle_error_mat` gives a 0-d tensor.
        - ``"R_err"`` is :func:`~kornia.metrics.angle_error_mat` of the two rotation blocks and is never folded.
          ``"t_err"`` is :func:`~kornia.metrics.angle_error_vec` of the two translations, so it compares their
          directions only, folded as ``fold_translation`` says. ``"max_err"`` is the larger of the two per pose.
        - Only the top three rows are read: the bottom row of a :math:`(4, 4)` pose is ignored.

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
        A pose with zero translation gives ``NaN`` for ``"t_err"`` and ``"max_err"``, and
        :func:`auc_from_errors` propagates that into the AUC. Mask those entries first.

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

    Convention:
        - See the Convention block of :func:`~kornia.metrics.pose_errors`, whose ``"max_err"`` is the usual input.
        - All errors are pooled into one curve, whatever their shape. The recall curve rises by :math:`1/n` at each
          of the :math:`n` sorted errors, joined to :math:`(0, 0)` and to each other by straight lines (the trapezoid
          rule), and is held flat from the last error below the threshold out to the threshold. The AUC is its area
          divided by the threshold, times 100: a **percentage**, keyed by the threshold as a Python float. An ``inf``
          error never enters the area, so it counts as a failure. :ref:`Losses and metrics
          <losses-metrics-conventions>` maps the AUC and :func:`~kornia.metrics.pose_errors` onto glue-factory and
          SuperGlue.

    Args:
        errors: per-sample error values of shape :math:`(B,)`. Must be non-negative. Integer and
            half-precision inputs are promoted to the default floating dtype before accumulating.
        thresholds: a single threshold or a sequence of thresholds, in the same units as ``errors``.
            Must be strictly positive. Defaults to ``(1, 3, 5, 10)``.

    Return:
        a dict mapping each threshold to its AUC in :math:`[0, 100]`, or ``NaN`` at every threshold
        if any error is ``NaN``.

    .. note::
        An error exactly equal to a threshold contributes no area there, so errors all equal to
        ``thr`` score ``0`` at ``thr``. This follows the reference implementations.

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
    # An integer dtype would truncate the threshold, half precision loses exactness in arange.
    if errors.dtype not in (torch.float32, torch.float64):
        errors = errors.to(torch.get_default_dtype())
    # A negative error would sort ahead of the zero prepended below, leaving the array unsorted and
    # searchsorted free to return nonsense. NaN compares false here and is handled next.
    KORNIA_CHECK(not bool((errors < 0).any()), "errors must be non-negative.")
    if bool(torch.isnan(errors).any()):
        return dict.fromkeys(thresholds, float("nan"))
    errors = errors.sort().values
    n = errors.numel()
    recall = torch.arange(1, n + 1, device=errors.device, dtype=errors.dtype) / n
    errors = torch.cat([errors.new_zeros(1), errors])
    recall = torch.cat([recall.new_zeros(1), recall])

    aucs: dict[float, float] = {}
    for thr in thresholds:
        # First error at or past the threshold. A positive thr always lands past the prepended
        # zero, so last >= 1 and the curve can be closed off with a flat segment out to thr.
        last = int(torch.searchsorted(errors, errors.new_tensor(thr)).item())
        recall_below = torch.cat([recall[:last], recall[last - 1 : last]])
        errors_below = torch.cat([errors[:last], errors.new_tensor([thr])])
        area = torch.trapezoid(recall_below, x=errors_below)
        aucs[thr] = (area / thr).item() * 100.0
    return aucs
