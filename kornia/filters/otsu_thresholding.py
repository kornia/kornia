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

from __future__ import annotations

from typing import Optional, Tuple

import torch

from kornia.core.check import KORNIA_CHECK
from kornia.core.utils import _torch_histc_cast


class OtsuThreshold(torch.nn.Module):
    """Otsu thresholding module for PyTorch tensors."""

    # Bandwidth of the kernel density estimate behind ``slow_and_differentiable=True``, in bin widths.
    _KDE_BANDWIDTH: float = 0.1
    # Temperature of the soft-argmax that gives the slow path's threshold its gradient, relative to the largest
    # between-class variance of the plane.
    _SOFT_ARGMAX_TEMPERATURE: float = 0.01

    def __init__(self) -> None:
        """Initialize the OtsuThreshold module."""
        super().__init__()

    @staticmethod
    def __histogram(xs: torch.Tensor, bins: int, diff: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute a histogram for each row of xs, on the range of that row.

        Args:
            xs (torch.Tensor): 2D tensor (n, N) with values to histogram.
            bins (int): Number of bins.
            diff: if True, estimate the histogram with a Gaussian kernel density estimate whose bin masses are
                differentiable with respect to ``xs``. Default: False

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Normalized histograms (n, bins), and the minimum and
            maximum of each row (n,), in float32, or float64 for a float64 input.
        """
        # Histogram and threshold arithmetic run in float32 (float64 stays float64): integer inputs need a float
        # dtype for torch.histc, and half-precision counts and sums lose integers past 2048 (float16) or 256
        # (bfloat16).
        xs = xs.to(torch.float64 if xs.dtype == torch.float64 else torch.float32)
        if not diff:
            xs = xs.detach()

        min_val = xs.amin(dim=1)
        max_val = xs.amax(dim=1)

        histograms = []
        if diff:
            # Bin coordinates: the row's minimum maps to 0 and its maximum to `bins`, bin k is [k, k + 1). A constant
            # row maps to 0; its threshold does not come from the histogram.
            span = max_val - min_val
            bin_width = torch.where(span > 0, span, torch.ones_like(span)) / bins
            coords = (xs - min_val[:, None]) / bin_width[:, None]
            inner_edges = torch.arange(1, bins, device=xs.device, dtype=xs.dtype)
            for i in range(xs.shape[0]):
                # Each pixel is a Gaussian of standard deviation `_KDE_BANDWIDTH` bins: the fraction of the row's
                # mass above each inner edge, with the mass beyond the range kept in the first and last bins.
                z = (coords[i, :, None] - inner_edges) / OtsuThreshold._KDE_BANDWIDTH
                above = torch.special.ndtr(z).mean(dim=0)
                above = torch.cat([above.new_ones(1), above, above.new_zeros(1)])
                histograms.append(above[:-1] - above[1:])
        else:
            for i in range(xs.shape[0]):
                # Note: torch.histogram is in PyTorch 1.10+, and should replace histc in future versions when
                #       no longer supporting older pytorch versions.
                hist = _torch_histc_cast(xs[i], bins=bins, min=min_val[i].item(), max=max_val[i].item())
                histograms.append(hist / hist.sum())

        return torch.stack(histograms), min_val, max_val

    def transform_input(
        self, x: torch.Tensor, original_shape: Optional[torch.Size] = None
    ) -> Tuple[torch.Tensor, torch.Size]:
        """Flatten the input to make it compatible with threshold computation.

        Args:
            x (torch.Tensor): Image or batch of images.
            original_shape (Optional[torch.Size]): Shape to preserve.

        Returns:
            Tuple[torch.Tensor, torch.Size]: Flattened tensor, original shape.
        """
        if original_shape is None:
            original_shape = x.shape
        dimensionality: int = x.dim()

        if dimensionality <= 2:
            return x.flatten().unsqueeze(0), original_shape
        if dimensionality == 3:
            return x.flatten(start_dim=1), original_shape
        if dimensionality == 4:
            b, c, h, w = x.shape
            return self.transform_input(x.reshape(b * c, h, w), original_shape=original_shape)
        if dimensionality == 5:
            f, b, c, h, w = x.shape
            return self.transform_input(x.reshape(f * b * c, h, w), original_shape=original_shape)
        raise ValueError(f"Unsupported tensor dimensionality: {dimensionality}")

    def forward(
        self, x: torch.Tensor, nbins: int = 256, slow_and_differentiable: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply Otsu thresholding to the input x.

        Args:
            x (torch.Tensor): Image or batch of images to threshold.
            nbins (int, optional): Number of bins for histogram computation. Default is 256.
            slow_and_differentiable (bool, optional): If True, estimate the histogram with a Gaussian kernel density
                estimate and make the threshold differentiable with respect to ``x``; see
                :func:`~kornia.filters.otsu_threshold`. Default is False.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Thresholded tensor, threshold values.
        """
        # Flatten input and store original shape
        x_flattened, orig_shape = self.transform_input(x)

        # Check tensor type compatibility
        KORNIA_CHECK(
            x.dtype
            in [
                torch.uint8,
                torch.int8,
                torch.int16,
                torch.int32,
                torch.int64,
                torch.float32,
                torch.float64,
                torch.float16,
                torch.bfloat16,
            ],
            "Tensor dtype not supported for Otsu thresholding.",
        )

        # Compute the histogram of each plane on its own range
        histograms, min_val, max_val = self.__histogram(x_flattened, bins=nbins, diff=slow_and_differentiable)
        span = max_val - min_val

        # Vectorized computation of optimal thresholds
        bin_values = torch.arange(nbins, device=histograms.device, dtype=histograms.dtype)
        total_weight = torch.sum(histograms, dim=1)  # Shape: (nchannel,)
        total_sum = torch.sum(histograms * bin_values, dim=1)  # Shape: (nchannel,)
        cumsum_weight = torch.cumsum(histograms, dim=1)  # Shape: (nchannel, nbins)
        cumsum_sum = torch.cumsum(histograms * bin_values, dim=1)  # Shape: (nchannel, nbins)

        # Compute weights and sums for background and foreground
        weight_bg = cumsum_weight[:, :-1]  # Shape: (nchannel, nbins-1)
        sum_bg = cumsum_sum[:, :-1]  # Shape: (nchannel, nbins-1)
        weight_fg = total_weight[:, None] - weight_bg  # Shape: (nchannel, nbins-1)
        sum_fg = total_sum[:, None] - sum_bg  # Shape: (nchannel, nbins-1)

        # Compute means; an empty class divides by 1 so that no inf or nan reaches the gradient
        nonempty_bg = weight_bg > 0
        nonempty_fg = weight_fg > 0
        mean_bg = sum_bg / torch.where(nonempty_bg, weight_bg, torch.ones_like(weight_bg))
        mean_fg = sum_fg / torch.where(nonempty_fg, weight_fg, torch.ones_like(weight_fg))

        # Compute inter-class variance, setting invalid cases to -1
        valid = nonempty_bg & nonempty_fg
        inter_class_var = torch.where(
            valid, weight_bg * weight_fg * (mean_bg - mean_fg) ** 2, torch.full_like(weight_bg, -1.0)
        )

        # The best split t keeps bins 0..t as background: the threshold is the upper edge of bin t in the histogram's
        # own bins, min + (t + 1) * (max - min) / nbins. A split after an empty bin repeats the partition of the split
        # before it, so only splits after a non-empty bin compete: the lowest of equivalent splits wins on every
        # device, whatever the rounding of the cumulative sums.
        candidates = torch.where(histograms[:, :-1] > 0, inter_class_var, torch.full_like(inter_class_var, -1.0))
        t_max = torch.argmax(candidates, dim=1)  # Shape: (nchannel,)
        best_thresholds = min_val + (t_max + 1).to(span.dtype) * span / nbins
        if slow_and_differentiable:
            # Straight-through: the value is the hard split above, the gradient that of a soft-argmax over the
            # between-class variance curve
            soft_thresholds = self._soft_argmax_threshold(inter_class_var, valid, min_val, span, nbins)
            best_thresholds = best_thresholds.detach() + (soft_thresholds - soft_thresholds.detach())

        # A constant plane has no split and span 0, so its threshold above is min_val, its own value
        best_thresholds = best_thresholds.to(x.dtype)

        # Apply thresholding: keep values strictly greater than the threshold
        thresholded = (x_flattened > best_thresholds[:, None]).to(x.dtype) * x_flattened
        thresholded = thresholded.reshape(orig_shape)

        return thresholded, best_thresholds

    @staticmethod
    def _soft_argmax_threshold(
        inter_class_var: torch.Tensor, valid: torch.Tensor, min_val: torch.Tensor, span: torch.Tensor, nbins: int
    ) -> torch.Tensor:
        """Threshold at the soft-argmax of the between-class variance curve, differentiable with respect to the input.

        Split ``k`` is weighted by ``softmax(var_k / (T * max(var)))`` with ``T = _SOFT_ARGMAX_TEMPERATURE``: the
        temperature is relative to the curve, so the weights do not depend on the intensity scale of the plane.
        """
        largest = inter_class_var.amax(dim=1, keepdim=True)
        largest = torch.where(largest > 0, largest, torch.ones_like(largest))
        logits = inter_class_var / (OtsuThreshold._SOFT_ARGMAX_TEMPERATURE * largest)
        # A finite floor, not -inf, for invalid splits: a constant plane has no valid split, and its weights stay finite
        logits = torch.where(valid, logits, torch.full_like(logits, torch.finfo(logits.dtype).min))
        weights = torch.softmax(logits, dim=1)
        splits = torch.arange(nbins - 1, device=weights.device, dtype=weights.dtype)
        soft_split = (weights * splits).sum(dim=1)
        return min_val + (soft_split + 1) * span / nbins


def otsu_threshold(
    x: torch.Tensor,
    nbins: int = 256,
    slow_and_differentiable: bool = False,
    return_mask: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Apply automatic image thresholding using Otsu algorithm to the input tensor.

    Args:
        x (Tensor): Input tensor (image or batch of images).
        nbins (int): Number of bins for histogram computation, default is 256.
        slow_and_differentiable (bool): If True, use a differentiable histogram computation. Default is False.
        return_mask (bool): If True, return a binary mask indicating the thresholded pixels. If False,
            return the thresholded image.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Thresholded tensor and the computed threshold values.

    Raises:
        ValueError: If the input tensor has unsupported dimensionality or dtype.

    .. note::
        - The input tensor can be of various types, but float types are preferred for accuracy
          in histogram computation, especially on CPU. Integer types will be cast to float.
        - If `use_thresh` is True, the threshold must have been computed previously and set in the module.
        - If `threshold` is provided, it overrides the computed threshold.

    .. note::
        You may found more information about the Otsu algorithm here: https://en.wikipedia.org/wiki/Otsu's_method

    .. note::
        Each plane, over the last two axes of ``x`` (all of ``x`` when it has at most two dimensions), is thresholded
        on its own range: its histogram has ``nbins`` bins of width :math:`w = (\max - \min) / n_{bins}` from the
        plane's minimum to its maximum, and the best split, after bin :math:`t`, gives the threshold
        :math:`\min + (t + 1) w`, the upper edge of bin :math:`t`. A constant plane has no split, and its threshold is
        its own value, so none of its pixels is kept.

        With ``slow_and_differentiable=True`` the histogram holds the mass that a Gaussian kernel density estimate of
        bandwidth :math:`0.1 w` puts in each bin, so every pixel contributes to it, and the threshold has a gradient
        with respect to ``x``. Its value is the best split of that histogram, as above, usually within one bin of the
        default threshold. Its gradient is that of a soft-argmax over the between-class variance curve
        :math:`\sigma_B^2(k)` (a straight-through estimator): split :math:`k` is weighted by
        :math:`\operatorname{softmax}_k\left(\sigma_B^2(k) / (0.01 \max_j \sigma_B^2(j))\right)`, a temperature
        relative to the curve and so independent of the intensity scale of ``x``. The value jumps when the best split
        changes, so finite differences of the threshold do not match this gradient. With the default
        ``slow_and_differentiable=False`` the threshold has no gradient. In both modes the thresholded image is
        ``x * (x > threshold)``, whose gradient with respect to ``x`` is that mask: none flows through the threshold.

    Example:
        >>> import torch
        >>> from kornia.filters.otsu_thresholding import otsu_threshold
        >>> x = torch.tensor([[10, 20, 30], [40, 50, 60], [70, 80, 90]])
        >>> x
        tensor([[10, 20, 30],
                [40, 50, 60],
                [70, 80, 90]])
        >>> otsu_threshold(x)
        (tensor([[ 0,  0,  0],
                [ 0, 50, 60],
                [70, 80, 90]]), tensor([40]))
    """
    module = OtsuThreshold()

    result, threshold = module(x, nbins=nbins, slow_and_differentiable=slow_and_differentiable)

    if return_mask:
        return result > 0, threshold

    return result, threshold
