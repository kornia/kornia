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

from typing import Optional

import torch

from kornia.core.check import KORNIA_CHECK, KORNIA_CHECK_IS_TENSOR, KORNIA_CHECK_SAME_SHAPE


def _is_broadcastable_to(shape: torch.Size, target_shape: torch.Size) -> bool:
    """Check whether ``shape`` can be broadcast to ``target_shape``."""
    if len(shape) > len(target_shape):
        return False
    return all(s in (1, t) for s, t in zip(reversed(shape), reversed(target_shape), strict=False))


def _check_disparity_inputs(
    input: torch.Tensor, target: torch.Tensor, valid_mask: Optional[torch.Tensor]
) -> Optional[torch.Tensor]:
    """Validate disparity metric inputs and return the valid mask broadcast to the input shape."""
    KORNIA_CHECK_IS_TENSOR(input)
    KORNIA_CHECK_IS_TENSOR(target)
    KORNIA_CHECK_SAME_SHAPE(input, target)

    if valid_mask is None:
        return None

    KORNIA_CHECK_IS_TENSOR(valid_mask)
    KORNIA_CHECK(
        _is_broadcastable_to(valid_mask.shape, input.shape),
        f"valid_mask shape must be broadcastable to the input shape. Got: {valid_mask.shape} and {input.shape}",
    )

    return valid_mask.to(torch.bool).broadcast_to(input.shape)


def _reduce_disparity_error(error: torch.Tensor, valid_mask: Optional[torch.Tensor], reduction: str) -> torch.Tensor:
    """Reduce a per-pixel error map over the valid pixels according to ``reduction``.

    The sums are accumulated in ``float32`` for half-precision inputs. A ``float16`` sum saturates
    at 65504, so an image-sized map would otherwise reduce to ``inf``, and a ``bfloat16`` sum loses
    several significant digits over the same number of terms. The result is cast back to the dtype
    of ``error`` so that the metrics keep returning the dtype of their input.
    """
    out_dtype = error.dtype
    acc_dtype = torch.promote_types(out_dtype, torch.float32)

    if valid_mask is not None:
        # Zeroing the invalid pixels keeps the shape static. Indexing with the mask would make the
        # output shape depend on the mask values, which torch.compile cannot trace in a full graph.
        error = torch.where(valid_mask, error, 0)

    if reduction == "mean":
        count = error.numel() if valid_mask is None else valid_mask.sum()
        error = (error.sum(dtype=acc_dtype) / count).to(out_dtype)
    elif reduction == "sum":
        error = error.sum(dtype=acc_dtype).to(out_dtype)
    elif reduction == "none":
        pass
    else:
        raise NotImplementedError("Invalid reduction option.")

    return error


def mean_absolute_disparity_error(
    input: torch.Tensor,
    target: torch.Tensor,
    valid_mask: Optional[torch.Tensor] = None,
    reduction: str = "mean",
) -> torch.Tensor:
    r"""Compute the mean absolute error (MAE) between two disparity maps.

    Given predicted and ground truth disparity maps :math:`D` and :math:`D^{gt}` with
    valid pixels :math:`\mathcal{V}`, the metric is:

    .. math::

        \text{MAE}(D, D^{gt}) = \frac{1}{|\mathcal{V}|}\sum_{p \in \mathcal{V}} |D_{p} - D^{gt}_{p}|

    Args:
        input: the predicted disparity map with arbitrary shape :math:`(*)`.
        target: the ground truth disparity map with the same shape as ``input``.
        valid_mask: optional mask broadcastable to the shape of ``input``, where nonzero
            (``True``) values mark the pixels to evaluate. Non-boolean masks are converted
            to boolean. If ``None``, all pixels are evaluated.
        reduction: specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'mean'``: the error is averaged over the
            valid pixels, ``'sum'``: the error is summed over the valid pixels, ``'none'``: no
            reduction will be applied and the per-pixel error map is returned, with masked-out
            positions set to zero.

    Return:
        the computed metric as a scalar, or the per-pixel error map if ``reduction='none'``.

    Note:
        If ``valid_mask`` selects no pixels, ``'mean'`` reduction returns ``nan``.

    Note:
        Sums are accumulated in ``float32`` for ``float16`` and ``bfloat16`` inputs and the result
        is cast back, so ``'mean'`` stays accurate on image-sized maps. ``'sum'`` returns the input
        dtype as well, so a ``float16`` total above 65504 still saturates to ``inf`` — reduce with
        ``'none'`` and sum in a wider dtype if you need totals that large.

    Examples:
        >>> input = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
        >>> target = torch.tensor([[0.0, 1.0], [2.0, 4.0]])
        >>> mean_absolute_disparity_error(input, target)
        tensor(0.2500)
        >>> valid_mask = torch.tensor([[True, True], [True, False]])
        >>> mean_absolute_disparity_error(input, target, valid_mask)
        tensor(0.)

    Reference:
        D. Scharstein and R. Szeliski. A taxonomy and evaluation of dense two-frame stereo
        correspondence algorithms. IJCV 2002. https://vision.middlebury.edu/stereo/taxonomy-IJCV.pdf

    """
    mask = _check_disparity_inputs(input, target, valid_mask)
    error = (input - target).abs()
    return _reduce_disparity_error(error, mask, reduction)


def root_mean_squared_disparity_error(
    input: torch.Tensor,
    target: torch.Tensor,
    valid_mask: Optional[torch.Tensor] = None,
    reduction: str = "mean",
) -> torch.Tensor:
    r"""Compute the root mean squared error (RMSE) between two disparity maps.

    Given predicted and ground truth disparity maps :math:`D` and :math:`D^{gt}` with
    valid pixels :math:`\mathcal{V}`, the metric is:

    .. math::

        \text{RMSE}(D, D^{gt}) =
        \sqrt{\frac{1}{|\mathcal{V}|}\sum_{p \in \mathcal{V}} (D_{p} - D^{gt}_{p})^{2}}

    Args:
        input: the predicted disparity map with arbitrary shape :math:`(*)`.
        target: the ground truth disparity map with the same shape as ``input``.
        valid_mask: optional mask broadcastable to the shape of ``input``, where nonzero
            (``True``) values mark the pixels to evaluate. Non-boolean masks are converted
            to boolean. If ``None``, all pixels are evaluated.
        reduction: specifies the reduction to apply to the squared error before the square
            root: ``'none'`` | ``'mean'`` | ``'sum'``. ``'mean'``: the squared error is
            averaged over the valid pixels, ``'sum'``: the squared error is summed over the
            valid pixels, ``'none'``: no reduction will be applied and the per-pixel absolute
            error map is returned, with masked-out positions set to zero.

    Return:
        the computed metric as a scalar, or the per-pixel error map if ``reduction='none'``.

    Note:
        If ``valid_mask`` selects no pixels, ``'mean'`` reduction returns ``nan``.

    Note:
        For ``float16`` and ``bfloat16`` inputs the squared error and its reduction are computed in
        ``float32`` and only the final value is cast back. Without that, a disparity error above
        256 px would already saturate the ``float16`` squared error to ``inf``.

    Examples:
        >>> input = torch.zeros(2, 2)
        >>> target = torch.tensor([[0.0, 0.0], [0.0, 1.0]])
        >>> root_mean_squared_disparity_error(input, target)
        tensor(0.5000)

    Reference:
        D. Scharstein and R. Szeliski. A taxonomy and evaluation of dense two-frame stereo
        correspondence algorithms. IJCV 2002. https://vision.middlebury.edu/stereo/taxonomy-IJCV.pdf

    """
    mask = _check_disparity_inputs(input, target, valid_mask)
    # Square in float32 for half-precision inputs: a 300 px disparity error already squares to
    # 90000, past the 65504 float16 ceiling, so the per-pixel map would saturate to inf before any
    # reduction happens. The square root brings the value back into range before the cast back.
    acc_dtype = torch.promote_types(input.dtype, torch.float32)
    error = (input.to(acc_dtype) - target.to(acc_dtype)) ** 2
    return _reduce_disparity_error(error, mask, reduction).sqrt().to(input.dtype)


def mean_bad_pixel_error(
    input: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 3.0,
    valid_mask: Optional[torch.Tensor] = None,
    reduction: str = "mean",
) -> torch.Tensor:
    r"""Compute the bad pixel ratio between two disparity maps.

    A pixel is considered bad when its absolute disparity error is strictly greater than
    ``threshold``. Given predicted and ground truth disparity maps :math:`D` and :math:`D^{gt}`
    with valid pixels :math:`\mathcal{V}`, the metric is:

    .. math::

        \text{Bad}_{\tau}(D, D^{gt}) =
        \frac{1}{|\mathcal{V}|}\sum_{p \in \mathcal{V}} [|D_{p} - D^{gt}_{p}| > \tau]

    This corresponds to the bad-pixel percentage reported by the Middlebury and KITTI stereo
    benchmarks, expressed as a fraction in :math:`[0, 1]` instead of a percentage.

    Args:
        input: the predicted disparity map with arbitrary shape :math:`(*)`.
        target: the ground truth disparity map with the same shape as ``input``.
        threshold: the disparity error above which a pixel is considered bad.
        valid_mask: optional mask broadcastable to the shape of ``input``, where nonzero
            (``True``) values mark the pixels to evaluate. Non-boolean masks are converted
            to boolean. If ``None``, all pixels are evaluated.
        reduction: specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'mean'``: the fraction of bad pixels among
            the valid pixels, ``'sum'``: the number of bad pixels among the valid pixels,
            ``'none'``: no reduction will be applied and the per-pixel bad-pixel map is
            returned, with masked-out positions set to zero.

    Return:
        the computed metric as a scalar, or the per-pixel bad-pixel map if ``reduction='none'``.

    Note:
        If ``valid_mask`` selects no pixels, ``'mean'`` reduction returns ``nan``.

    Note:
        Sums are accumulated in ``float32`` for ``float16`` and ``bfloat16`` inputs and the result
        is cast back, so ``'mean'`` stays accurate on image-sized maps. ``'sum'`` returns the input
        dtype as well, so a ``float16`` count above 65504 still saturates to ``inf`` — reduce with
        ``'none'`` and sum in a wider dtype if you need counts that large.

    Examples:
        >>> input = torch.zeros(2, 2)
        >>> target = torch.tensor([[0.0, 1.0], [2.0, 4.0]])
        >>> mean_bad_pixel_error(input, target, threshold=1.5)
        tensor(0.5000)

    Reference:
        D. Scharstein and R. Szeliski. A taxonomy and evaluation of dense two-frame stereo
        correspondence algorithms. IJCV 2002. https://vision.middlebury.edu/stereo/taxonomy-IJCV.pdf

    """
    mask = _check_disparity_inputs(input, target, valid_mask)
    bad = ((input - target).abs() > threshold).to(input.dtype)
    return _reduce_disparity_error(bad, mask, reduction)


def kitti_d1_error(
    input: torch.Tensor,
    target: torch.Tensor,
    abs_threshold: float = 3.0,
    rel_threshold: float = 0.05,
    valid_mask: Optional[torch.Tensor] = None,
    reduction: str = "mean",
) -> torch.Tensor:
    r"""Compute the KITTI D1 error between two disparity maps.

    A pixel is considered an outlier when its absolute disparity error exceeds ``abs_threshold``
    **and** its relative disparity error exceeds ``rel_threshold``. Given predicted and ground truth
    disparity maps :math:`D` and :math:`D^{gt}` with valid pixels :math:`\mathcal{V}`, the metric is:

    .. math::

        \text{D1}(D, D^{gt}) = \frac{1}{|\mathcal{V}|}\sum_{p \in \mathcal{V}}
        \left[|D_{p} - D^{gt}_{p}| > \tau_{abs} \;\wedge\;
        \frac{|D_{p} - D^{gt}_{p}|}{|D^{gt}_{p}|} > \tau_{rel}\right]

    This is the outlier ratio reported by the KITTI 2015 stereo benchmark, expressed as a fraction
    in :math:`[0, 1]` instead of a percentage. Unlike :func:`mean_bad_pixel_error`, the additional
    relative criterion keeps large disparities from being penalised for errors that are small
    compared to their magnitude.

    Args:
        input: the predicted disparity map with arbitrary shape :math:`(*)`.
        target: the ground truth disparity map with the same shape as ``input``.
        abs_threshold: the absolute disparity error above which a pixel may be an outlier.
        rel_threshold: the relative disparity error above which a pixel may be an outlier.
        valid_mask: optional mask broadcastable to the shape of ``input``, where nonzero
            (``True``) values mark the pixels to evaluate. Non-boolean masks are converted
            to boolean. If ``None``, all pixels are evaluated.
        reduction: specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'mean'``: the fraction of outliers among
            the valid pixels, ``'sum'``: the number of outliers among the valid pixels,
            ``'none'``: no reduction will be applied and the per-pixel outlier map is
            returned, with masked-out positions set to zero.

    Return:
        the computed metric as a scalar, or the per-pixel outlier map if ``reduction='none'``.

    Note:
        If ``valid_mask`` selects no pixels, ``'mean'`` reduction returns ``nan``.

    Note:
        Pixels with a zero ground truth disparity yield a non-finite relative error. Because both
        criteria must hold, such pixels are classified by ``abs_threshold`` alone and the output
        stays finite. KITTI marks these pixels as invalid, so prefer passing ``valid_mask``.

    Note:
        Sums are accumulated in ``float32`` for ``float16`` and ``bfloat16`` inputs and the result
        is cast back, so ``'mean'`` stays accurate on image-sized maps. ``'sum'`` returns the input
        dtype as well, so a ``float16`` count above 65504 still saturates to ``inf`` — reduce with
        ``'none'`` and sum in a wider dtype if you need counts that large.

    Examples:
        >>> input = torch.tensor([1.0, 5.0, 104.0, 20.0])
        >>> target = torch.tensor([1.0, 1.0, 100.0, 10.0])
        >>> kitti_d1_error(input, target)
        tensor(0.5000)

    Reference:
        M. Menze and A. Geiger. Object Scene Flow for Autonomous Vehicles. CVPR 2015.
        https://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=stereo

    """
    mask = _check_disparity_inputs(input, target, valid_mask)
    error = (input - target).abs()
    outlier = ((error > abs_threshold) & (error / target.abs() > rel_threshold)).to(input.dtype)
    return _reduce_disparity_error(outlier, mask, reduction)
