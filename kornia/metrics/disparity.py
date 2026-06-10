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
    return all(s == 1 or s == t for s, t in zip(reversed(shape), reversed(target_shape), strict=False))


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
    """Reduce a per-pixel error map over the valid pixels according to ``reduction``."""
    if reduction == "mean":
        error = error.mean() if valid_mask is None else error[valid_mask].mean()
    elif reduction == "sum":
        error = error.sum() if valid_mask is None else error[valid_mask].sum()
    elif reduction == "none":
        if valid_mask is not None:
            error = torch.where(valid_mask, error, torch.zeros_like(error))
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
    error = (input - target) ** 2
    return _reduce_disparity_error(error, mask, reduction).sqrt()


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
