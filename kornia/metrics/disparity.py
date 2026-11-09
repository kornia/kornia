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

from kornia.core.check import (
    KORNIA_CHECK,
    KORNIA_CHECK_IS_TENSOR,
    KORNIA_CHECK_SAME_DEVICE,
    KORNIA_CHECK_SAME_SHAPE,
)


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
    # torch combines a 0-d CPU tensor with a tensor on any device, as a scalar, so a 0-d pair with one map on the CPU
    # is not checked. Maps with dimensions, and 0-d maps on two other devices, must share a device.
    if input.ndim > 0 or (input.device.type != "cpu" and target.device.type != "cpu"):
        KORNIA_CHECK_SAME_DEVICE(input, target)
    # The reductions below accumulate in float32 and cast back to the map dtype. For an integer
    # map that cast truncates, which would turn a ratio into 0 instead of raising, so reject
    # integers here. KITTI ships disparity as a uint16 PNG, so this is a realistic input.
    KORNIA_CHECK(
        input.is_floating_point() and target.is_floating_point(),
        "Disparity maps must have a floating point dtype. "
        f"Got: {input.dtype} and {target.dtype}. Convert them first, for example with `.float()`.",
    )

    if valid_mask is None:
        return None

    KORNIA_CHECK_IS_TENSOR(valid_mask)
    KORNIA_CHECK(
        _is_broadcastable_to(valid_mask.shape, input.shape),
        f"valid_mask shape must be broadcastable to the input shape. Got: {valid_mask.shape} and {input.shape}",
    )

    # A mask on another device, such as a CPU mask with CUDA maps, is moved to the maps.
    return valid_mask.to(device=input.device, dtype=torch.bool).broadcast_to(input.shape)


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
    r"""Compute the mean absolute error (MAE) between two disparity maps, over the valid pixels.

    ``mae = mean_absolute_disparity_error(input, target, valid_mask)`` takes two floating disparity maps in pixels
    with shape ``(*)`` and returns the mean of their absolute difference over the valid pixels :math:`\mathcal{V}`
    as a 0-d tensor:

    .. math::

        \text{MAE}(D, D^{gt}) = \frac{1}{|\mathcal{V}|}\sum_{p \in \mathcal{V}} |D_{p} - D^{gt}_{p}|

    Convention:
        - ``'mean'`` and ``'sum'`` pool every valid pixel of every image into one 0-d value, so for images with
          different numbers of valid pixels the result is not the mean of the per-image errors. The other disparity
          metrics pool the same way. If ``valid_mask`` selects no pixels, ``'mean'`` returns ``nan`` and ``'sum'``
          returns 0.
        - ``valid_mask`` is broadcast to the maps from the right: an ``(H, W)`` mask applies to every image of a
          ``(B, H, W)`` batch and a ``(B, 1, 1)`` mask selects whole images, while a ``(B,)`` mask is
          matched against the last axis, not the batch. Nothing is masked automatically: a pixel with an invalid
          ground truth, such as a zero disparity, counts unless ``valid_mask`` excludes it.
        - A ``nan`` at a valid pixel of either map propagates into this error and the RMSE, while
          :func:`~kornia.metrics.mean_bad_pixel_error` and :func:`~kornia.metrics.kitti_d1_error` count it as a
          good pixel.
        - Half-precision sums are accumulated in ``float32`` and the result is cast back to the input dtype, so
          ``'mean'`` stays accurate on image-sized maps, while a ``float16`` ``'sum'`` above 65504 saturates to
          ``inf``: reduce with ``'none'`` and sum in a wider dtype for totals that large.

    Args:
        input: predicted disparity map in pixels, with shape ``(*)``.
        target: ground-truth disparity map with the shape of ``input``.
        valid_mask: optional mask broadcastable to ``input``; nonzero (``True``) values mark the pixels to evaluate.
            ``None`` evaluates all pixels.
        reduction: ``'mean'`` (default) averages over the valid pixels, ``'sum'`` adds them and ``'none'`` returns the
            per-pixel error map with masked-out positions set to zero.

    Returns:
        the error as a 0-d tensor, or the per-pixel error map for ``reduction='none'``.

    See Also:
        - :func:`~kornia.metrics.root_mean_squared_disparity_error`: the root of the mean squared error, which weights
          large errors more.
        - :func:`~kornia.metrics.mean_bad_pixel_error`: the fraction of pixels above an error threshold.
        - :func:`~kornia.metrics.kitti_d1_error`: the KITTI outlier fraction, with a relative criterion.

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
    r"""Compute the root mean squared error (RMSE) between two disparity maps, over the valid pixels.

    ``rmse = root_mean_squared_disparity_error(input, target, valid_mask)`` takes two floating disparity maps in
    pixels with shape ``(*)`` and returns the root of the mean squared difference over the valid pixels
    :math:`\mathcal{V}` as a 0-d tensor:

    .. math::

        \text{RMSE}(D, D^{gt}) =
        \sqrt{\frac{1}{|\mathcal{V}|}\sum_{p \in \mathcal{V}} (D_{p} - D^{gt}_{p})^{2}}

    Convention:
        - See the Convention block of :func:`~kornia.metrics.mean_absolute_disparity_error` for the pooling, the
          mask, the empty-mask result and half-precision sums.
        - ``'sum'`` gives :math:`\sqrt{\sum e^2}`, not a sum of errors. The squared error is computed in ``float32``
          for half-precision inputs and only the final value is cast back, so an error above 256 px does not
          saturate ``float16``.

    Args:
        input: predicted disparity map in pixels, with shape ``(*)``.
        target: ground-truth disparity map with the shape of ``input``.
        valid_mask: optional mask broadcastable to ``input``; nonzero (``True``) values mark the pixels to evaluate.
            ``None`` evaluates all pixels.
        reduction: ``'mean'`` (default) averages the squared error over the valid pixels before the root, ``'sum'``
            adds it, and ``'none'`` returns the per-pixel absolute error with masked-out positions set to zero.

    Returns:
        the error as a 0-d tensor, or the per-pixel absolute error map for ``reduction='none'``.

    See Also:
        - :func:`~kornia.metrics.mean_absolute_disparity_error`: the mean of the absolute errors, which weights large
          errors less.

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
    out_dtype = torch.promote_types(input.dtype, target.dtype)
    acc_dtype = torch.promote_types(out_dtype, torch.float32)
    diff = input.to(acc_dtype) - target.to(acc_dtype)
    if mask is not None:
        # Zero a masked-out difference before squaring: the reduction drops its value, but the backward of the square
        # would still multiply the zero gradient by 2 * diff, which is NaN for a NaN or inf ground truth there.
        diff = torch.where(mask, diff, 0)
    squared = _reduce_disparity_error(diff**2, mask, reduction)
    # d sqrt(x)/dx is infinite at 0 and autograd multiplies it by the zero derivative of the square, so an exact match
    # would get a NaN gradient (and a mask cannot hide it: NaN * 0 is NaN). Differentiate the sqrt of a safe positive
    # argument and select 0 at zero error, the subgradient aepe and torch.linalg.vector_norm use. The value is
    # unchanged, and an empty mask still gives NaN under 'mean'.
    is_zero = squared == 0
    safe = torch.where(is_zero, torch.ones_like(squared), squared)
    return torch.where(is_zero, torch.zeros_like(squared), safe.sqrt()).to(out_dtype)


def mean_bad_pixel_error(
    input: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 3.0,
    valid_mask: Optional[torch.Tensor] = None,
    reduction: str = "mean",
) -> torch.Tensor:
    r"""Compute the fraction of bad pixels between two disparity maps, over the valid pixels.

    ``bad = mean_bad_pixel_error(input, target, threshold, valid_mask)`` takes two floating disparity maps in pixels
    with shape ``(*)`` and returns the fraction of valid pixels :math:`\mathcal{V}` whose absolute error is strictly
    greater than ``threshold`` :math:`\tau`, a 0-d tensor in :math:`[0, 1]`:

    .. math::

        \text{Bad}_{\tau}(D, D^{gt}) =
        \frac{1}{|\mathcal{V}|}\sum_{p \in \mathcal{V}} [|D_{p} - D^{gt}_{p}| > \tau]

    This is the bad-pixel percentage of the Middlebury stereo benchmark as a fraction. It is **not** the number
    KITTI reports: use :func:`~kornia.metrics.kitti_d1_error`.

    Convention:
        - See the Convention block of :func:`~kornia.metrics.mean_absolute_disparity_error` for the pooling, the
          mask, the empty-mask result and half-precision sums. ``'sum'`` gives the number of bad pixels.
        - The output is built from comparisons, so it does not require grad and cannot be used as a loss.

    Args:
        input: predicted disparity map in pixels, with shape ``(*)``.
        target: ground-truth disparity map with the shape of ``input``.
        threshold: the absolute error above which a pixel is bad.
        valid_mask: optional mask broadcastable to ``input``; nonzero (``True``) values mark the pixels to evaluate.
            ``None`` evaluates all pixels.
        reduction: ``'mean'`` (default) gives the fraction of bad pixels among the valid pixels, ``'sum'`` their
            number, and ``'none'`` the per-pixel bad-pixel map with masked-out positions set to zero.

    Returns:
        the fraction as a 0-d tensor, or the per-pixel bad-pixel map for ``reduction='none'``.

    See Also:
        - :func:`~kornia.metrics.kitti_d1_error`: adds a relative criterion.
        - :func:`~kornia.metrics.mean_absolute_disparity_error`: the mean error in pixels, with no threshold.

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
    bad = ((input - target).abs() > threshold).to(torch.promote_types(input.dtype, target.dtype))
    return _reduce_disparity_error(bad, mask, reduction)


def kitti_d1_error(
    input: torch.Tensor,
    target: torch.Tensor,
    abs_threshold: float = 3.0,
    rel_threshold: float = 0.05,
    valid_mask: Optional[torch.Tensor] = None,
    reduction: str = "mean",
) -> torch.Tensor:
    r"""Compute the KITTI D1 outlier fraction between two disparity maps, over the valid pixels.

    ``d1 = kitti_d1_error(input, target, abs_threshold, rel_threshold, valid_mask)`` takes two floating disparity
    maps in pixels with shape ``(*)`` and returns the fraction of valid pixels :math:`\mathcal{V}` that are
    outliers, a 0-d tensor in :math:`[0, 1]`. A pixel is an outlier when its absolute error exceeds ``abs_threshold``
    **and** its error relative to the ground truth exceeds ``rel_threshold``:

    .. math::

        \text{D1}(D, D^{gt}) = \frac{1}{|\mathcal{V}|}\sum_{p \in \mathcal{V}}
        \left[|D_{p} - D^{gt}_{p}| > \tau_{abs} \;\wedge\;
        \frac{|D_{p} - D^{gt}_{p}|}{|D^{gt}_{p}|} > \tau_{rel}\right]

    Convention:
        - See the Convention block of :func:`~kornia.metrics.mean_absolute_disparity_error` for the pooling, the
          mask, the empty-mask result and half-precision sums. ``'sum'`` gives the number of outliers.
          :ref:`Losses and metrics <losses-metrics-conventions>` relates it to the KITTI devkit.
        - The relative error divides by ``target``, so the two maps are not interchangeable: swapping them can
          change the result.
        - A zero ground-truth disparity gives a non-finite relative error, so that pixel is classified by
          ``abs_threshold`` alone and the output stays finite. KITTI marks these pixels invalid: pass ``valid_mask``.
        - A ``nan`` in either map counts as an **inlier**, because every comparison against ``nan`` is false and both
          criteria must hold. Mask non-finite pixels out with ``valid_mask``.
        - The output is built from comparisons, so it does not require grad and cannot be used as a loss.

    Args:
        input: predicted disparity map in pixels, with shape ``(*)``.
        target: ground-truth disparity map with the shape of ``input``.
        abs_threshold: the absolute error above which a pixel may be an outlier.
        rel_threshold: the relative error above which a pixel may be an outlier.
        valid_mask: optional mask broadcastable to ``input``; nonzero (``True``) values mark the pixels to evaluate.
            ``None`` evaluates all pixels.
        reduction: ``'mean'`` (default) gives the fraction of outliers among the valid pixels, ``'sum'`` their number,
            and ``'none'`` the per-pixel outlier map with masked-out positions set to zero.

    Returns:
        the fraction as a 0-d tensor, or the per-pixel outlier map for ``reduction='none'``.

    See Also:
        - :func:`~kornia.metrics.mean_bad_pixel_error`: the absolute criterion alone, as Middlebury reports.

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
    outlier = ((error > abs_threshold) & (error / target.abs() > rel_threshold)).to(
        torch.promote_types(input.dtype, target.dtype)
    )
    return _reduce_disparity_error(outlier, mask, reduction)
