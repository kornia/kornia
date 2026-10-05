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

    # Standard deviation, in bin widths, of the Gaussian kernel density estimate whose histogram selects the
    # threshold of ``slow_and_differentiable=True``.
    _KDE_BANDWIDTH: float = 0.1
    # Standard deviation, in bin widths, of the wider estimate behind that threshold's gradient. At 0.1 bin a pixel more
    # than about 0.3 bin from every bin edge would get under 1e-3 of the largest pixel gradient; at 0.5 bin a pixel's
    # gradient varies smoothly with its value instead of concentrating on pixels near a bin edge.
    _SURROGATE_BANDWIDTH: float = 0.5
    # Temperature of the soft-argmax behind that gradient, relative to the largest between-class variance of the plane.
    _SOFT_ARGMAX_TEMPERATURE: float = 0.01
    # A bin of the kernel density estimate counts as non-empty above this fraction of one pixel's mass: the kernel
    # tails give every bin a positive mass, down to about 1e-23 of a pixel.
    _NONEMPTY_BIN_MASS: float = 0.01

    def __init__(self) -> None:
        """Initialize the OtsuThreshold module."""
        super().__init__()

    @staticmethod
    def __histogram(xs: torch.Tensor, bins: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute a histogram for each row of xs, CUDA compatible.

        Args:
            xs (torch.Tensor): 2D tensor (n, N) with values to histogram.
            bins (int): Number of bins.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Normalized histograms and bin edges.
        """
        # Ensure input is float for histogram computation if it's integer type
        # For torch.histc, input should be floating point or quantized.
        if xs.dtype in [torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64]:
            xs = xs.to(torch.float32)

        min_values = xs.amin(dim=1)
        max_values = xs.amax(dim=1)
        histograms = []
        bin_edges = []
        edge_dtype = torch.float64 if xs.dtype == torch.float64 else torch.float32

        for i in range(xs.shape[0]):
            min_val = min_values[i].item()
            max_val = max_values[i].item()
            # histc divides [min, max] into `bins` intervals
            edges = torch.linspace(min_val, max_val, bins + 1, device=xs.device, dtype=edge_dtype)
            if min_val == max_val:
                # No split exists; avoid histc's automatic range expansion for large constant values.
                hist = torch.zeros(bins, device=xs.device, dtype=edge_dtype)
                hist[0] = 1
            else:
                hist = _torch_histc_cast(xs[i], bins=bins, min=min_val, max=max_val)

            histograms.append(hist / hist.sum())
            bin_edges.append(edges)

        return torch.stack(histograms), torch.stack(bin_edges)

    @staticmethod
    def _kde_histogram(coords: torch.Tensor, bins: int, bandwidth: float) -> torch.Tensor:
        """Compute the bin masses of a Gaussian kernel density estimate for each row of coords.

        Args:
            coords (torch.Tensor): 2D tensor (n, N) of values in bin coordinates: bin k is [k, k + 1).
            bins (int): Number of bins.
            bandwidth (float): Standard deviation of the Gaussian kernel, in bin widths.

        Returns:
            torch.Tensor: Histograms (n, bins), each summing to 1. A pixel's mass below 0 or above ``bins`` stays in
            the first or the last bin, so every pixel carries its full mass.
        """
        inner_edges = torch.arange(1, bins, device=coords.device, dtype=coords.dtype)
        histograms = []
        for i in range(coords.shape[0]):
            # fraction of the row's mass above each inner edge
            above = torch.special.ndtr((coords[i, :, None] - inner_edges) / bandwidth).mean(dim=0)
            above = torch.cat([above.new_ones(1), above, above.new_zeros(1)])
            histograms.append(above[:-1] - above[1:])
        return torch.stack(histograms)

    @staticmethod
    def _between_class_variance(histograms: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute Otsu's between-class variance of each split of each histogram, in squared bin widths.

        Args:
            histograms (torch.Tensor): Normalized histograms (n, bins).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The between-class variance (n, bins - 1) of the split after each bin
            but the last, -1 where a class is empty, and the mask of the splits whose two classes are non-empty.
        """
        bin_values = torch.arange(histograms.shape[1], device=histograms.device, dtype=histograms.dtype)
        total_weight = torch.sum(histograms, dim=1)  # Shape: (nchannel,)
        total_sum = torch.sum(histograms * bin_values, dim=1)  # Shape: (nchannel,)
        cumsum_weight = torch.cumsum(histograms, dim=1)  # Shape: (nchannel, nbins)
        cumsum_sum = torch.cumsum(histograms * bin_values, dim=1)  # Shape: (nchannel, nbins)

        # Compute weights and sums for background and foreground
        weight_bg = cumsum_weight[:, :-1]  # Shape: (nchannel, nbins-1)
        sum_bg = cumsum_sum[:, :-1]  # Shape: (nchannel, nbins-1)
        weight_fg = total_weight[:, None] - weight_bg  # Shape: (nchannel, nbins-1)
        sum_fg = total_sum[:, None] - sum_bg  # Shape: (nchannel, nbins-1)

        # Compute means; an empty class divides by 1, so that no inf or nan reaches a gradient
        valid_bg = weight_bg > 0
        valid_fg = weight_fg > 0
        mean_bg = sum_bg / torch.where(valid_bg, weight_bg, torch.ones_like(weight_bg))
        mean_fg = sum_fg / torch.where(valid_fg, weight_fg, torch.ones_like(weight_fg))

        # Compute inter-class variance, setting invalid cases to -1
        valid = valid_bg & valid_fg
        inter_class_var = torch.where(
            valid, weight_bg * weight_fg * (mean_bg - mean_fg) ** 2, torch.full_like(weight_bg, -1.0)
        )
        return inter_class_var, valid

    @staticmethod
    def _upper_edge(min_val: torch.Tensor, max_val: torch.Tensor, index: torch.Tensor, bins: int) -> torch.Tensor:
        """Compute edge ``index`` of ``bins`` equal bins from ``min_val`` to ``max_val``, without leaving the graph.

        On ordinary ranges this uses the scalar formula of ``torch.linspace(min_val, max_val, bins + 1)[index]``,
        which the default path reads its edges from: from the minimum for the lower half of the edges, from the
        maximum for the upper half. Extreme ranges are reconstructed in bounded units. The vectorized linspace
        kernels can differ from the scalar formula in the last bit.
        """
        span = max_val - min_val
        step = span / bins
        k = index.to(step.dtype)
        edge = torch.where(index < (bins + 1) // 2, min_val + step * k, max_val - step * (bins - k))
        # Preserve the scalar linspace arithmetic on ordinary ranges. Opposite-sign finite extrema can overflow
        # their difference, while subnormal ranges lose their step on division by bins; reconstruct those edges
        # in bounded units instead. This helper receives detached extrema, so its unused arm has no backward path.
        scale = torch.maximum(min_val.abs(), max_val.abs()).clamp_min(torch.finfo(min_val.dtype).tiny)
        lo, hi = min_val / scale, max_val / scale
        bounded_step = (hi - lo) / bins
        bounded_edge = torch.where(index < (bins + 1) // 2, lo + bounded_step * k, hi - bounded_step * (bins - k))
        ordinary = span.isfinite() & (span >= torch.finfo(span.dtype).tiny * bins)
        return torch.where(ordinary, edge, bounded_edge * scale)

    @staticmethod
    def _soft_threshold(coords: torch.Tensor, min_val: torch.Tensor, span: torch.Tensor, bins: int) -> torch.Tensor:
        """Compute the bounded-unit soft-argmax behind the first-order gradient of ``slow_and_differentiable=True``.

        The between-class variance curve of a kernel density estimate of ``_SURROGATE_BANDWIDTH`` bins weights the
        upper edge of split ``k`` by ``softmax(var_k / (T * max(var)))`` with ``T = _SOFT_ARGMAX_TEMPERATURE``. The
        curve is in bin units and the temperature is relative to it, so the weights do not depend on the intensity
        scale of the plane.
        """
        histograms = OtsuThreshold._kde_histogram(coords, bins, OtsuThreshold._SURROGATE_BANDWIDTH)
        inter_class_var, valid = OtsuThreshold._between_class_variance(histograms)
        largest = inter_class_var.amax(dim=1, keepdim=True)
        largest = torch.where(largest > 0, largest, torch.ones_like(largest))
        logits = inter_class_var / (OtsuThreshold._SOFT_ARGMAX_TEMPERATURE * largest)
        # A finite floor, not -inf, for invalid splits: a constant plane has no valid split, and its weights stay finite
        logits = torch.where(valid, logits, torch.full_like(logits, torch.finfo(logits.dtype).min))
        weights = torch.softmax(logits, dim=1)
        # the split after bin k has its upper edge at k + 1 bins from the minimum
        upper_edges = torch.arange(1, bins, device=weights.device, dtype=weights.dtype)
        return min_val + ((weights * upper_edges).sum(dim=1) / bins) * span

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
        """Apply Otsu thresholding independently to each image/channel plane of x.

        A constant plane uses its constant value as the threshold, so all its pixels are background.

        Args:
            x (torch.Tensor): Image or batch of images to threshold.
            nbins (int, optional): Number of bins for histogram computation. Default is 256.
            slow_and_differentiable (bool, optional): If True, build the histogram from a Gaussian kernel density
                estimate and give the threshold a gradient with respect to ``x``; see
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

        if slow_and_differentiable:
            # Histogram arithmetic in float32, or float64 for a float64 input, as on the default path. No .item():
            # the minimum, maximum and bin coordinates of each plane stay tensors.
            xs = x_flattened.to(torch.float64 if x.dtype == torch.float64 else torch.float32)
            min_val = xs.amin(dim=1).detach()
            max_val = xs.amax(dim=1).detach()
            # Evaluate the surrogate in bounded units, with an identity gradient to xs. For a detached positive
            # scale s, the original-unit surrogate is s * f(xs / s), whose derivative is exactly f'(xs / s).
            # Cancelling s here avoids both overflowing ranges and underflowing intermediate backward signals.
            scale = xs.detach().abs().amax(dim=1, keepdim=True).clamp_min(torch.finfo(xs.dtype).tiny)
            bounded = xs.detach() / scale + (xs - xs.detach())
            bounded_min = bounded.amin(dim=1)
            bounded_span = bounded.amax(dim=1) - bounded_min
            # The minimum maps to 0, the maximum to nbins, bin k is [k, k + 1). A constant plane maps to 0.
            safe_span = torch.where(bounded_span > 0, bounded_span, torch.ones_like(bounded_span))
            coords = ((bounded - bounded_min[:, None]) / safe_span[:, None]) * nbins
            histograms = self._kde_histogram(coords.detach(), nbins, self._KDE_BANDWIDTH)
            inter_class_var, _ = self._between_class_variance(histograms)
            # The kernel tails give every bin some mass, so a split after an empty bin, which repeats the partition of
            # the split before it, would compete on rounding noise: only splits after a non-empty bin compete, and
            # the lowest of equivalent splits wins on every device.
            nonempty = histograms[:, :-1] > self._NONEMPTY_BIN_MASS / xs.shape[1]
            inter_class_var = torch.where(nonempty, inter_class_var, torch.full_like(inter_class_var, -1.0))
            lower_edges = min_val.detach()
        else:
            # Compute histogram and bin edges
            histograms, bin_edges = self.__histogram(x_flattened, bins=nbins)
            inter_class_var, _ = self._between_class_variance(histograms)
            lower_edges = bin_edges[:, 0]

        # Find the maximum inter-class variance and corresponding threshold
        t_max = torch.argmax(inter_class_var, dim=1)  # Shape: (nchannel,)
        max_var = inter_class_var.gather(1, t_max[:, None]).squeeze(1)  # Shape: (nchannel,)
        if slow_and_differentiable:
            upper_edges = self._upper_edge(min_val.detach(), max_val.detach(), t_max + 1, nbins)
        else:
            upper_edges = bin_edges.gather(1, (t_max + 1)[:, None]).squeeze(1)
        if not x.is_floating_point():
            # An integer pixel on or above the upper edge is counted in the foreground, so the integer threshold is
            # the largest integer below that edge. Truncating toward zero would round a negative edge up, and keep
            # an integer edge, and drop the level just above the split from the foreground.
            upper_edges = upper_edges.ceil() - 1
        best_thresholds = torch.where(max_var > 0, upper_edges, lower_edges).to(x.dtype)

        # Preserve a constant plane's exact input value, including integers outside floating-point precision.
        plane_min = x_flattened.amin(dim=1).detach()
        plane_max = x_flattened.amax(dim=1).detach()
        best_thresholds = torch.where(plane_min == plane_max, plane_min, best_thresholds)

        if slow_and_differentiable and x.is_floating_point() and torch.is_grad_enabled() and x.requires_grad:
            # Straight-through: the value stays the hard split above, the gradient is that of the soft-argmax
            # threshold of the wider estimate. On a constant plane that threshold is the plane's minimum.
            soft_thresholds = self._soft_threshold(coords, bounded_min, bounded_span, nbins)
            best_thresholds = best_thresholds + (soft_thresholds - soft_thresholds.detach()).to(x.dtype)

        # Apply thresholding: keep values strictly greater than the threshold
        thresholded = (x_flattened > best_thresholds[:, None]).to(x.dtype) * x_flattened
        thresholded = thresholded.reshape(orig_shape)

        return thresholded, best_thresholds


def otsu_threshold(
    x: torch.Tensor,
    nbins: int = 256,
    slow_and_differentiable: bool = False,
    return_mask: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Apply automatic image thresholding using Otsu algorithm to the input tensor.

    Each image/channel plane uses its own histogram range. The threshold is the upper edge of the selected histogram
    bin. For an integer input it is the largest integer below that value, so ``x > threshold`` keeps every pixel on
    or above it. A constant plane uses its constant value as the threshold.

    Args:
        x (Tensor): Input tensor (image or batch of images).
        nbins (int): Number of bins for histogram computation, default is 256.
        slow_and_differentiable (bool): If True, build the histogram from a Gaussian kernel density estimate and give
            the threshold a gradient with respect to ``x``; see the note below. Default is False.
        return_mask (bool): If True, return the boolean mask ``x > threshold`` in place of the thresholded image,
            with each pixel compared against the threshold of its own image and channel. If False, return the
            thresholded image.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Thresholded tensor, or the boolean mask ``x > threshold`` when
        ``return_mask`` is True, and the computed threshold values. The thresholded tensor cannot tell a kept pixel
        of value 0 from a dropped one; use the mask for that.

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
        With ``slow_and_differentiable=True``, each pixel is spread into a Gaussian with a standard deviation of 0.1
        bin, and the histogram holds the mass of these Gaussians in each bin, a pixel's mass beyond the plane's range
        staying in the first or the last bin, so every pixel contributes its full mass. The threshold is the upper edge
        of the bin selected by the best split of that histogram, as on the default path, and is usually within one bin
        of the default threshold. A bin counts as non-empty only above 1 % of one pixel's mass, so of the splits that
        give the same partition the lowest is taken on every device.

        The threshold uses a hard split, and its first-order gradient is a straight-through surrogate: the
        gradient of a soft-argmax over the between-class variance curve :math:`\sigma_B^2(k)` of a wider estimate,
        with a standard deviation of 0.5 bin, so that a pixel's gradient varies smoothly with its value instead of
        concentrating on pixels near a bin edge. Split :math:`k` is weighted by
        :math:`\operatorname{softmax}_k\left(\sigma_B^2(k) / (0.01 \max_j \sigma_B^2(j))\right)`. The curve's means
        are measured in bins and the temperature is relative to the curve, so the weights do not depend on the
        intensity scale or offset of ``x``. The surrogate is evaluated in bounded units with the scale factors
        cancelled from its first-order gradient, so finite extreme ranges stay safe. Higher-order derivatives through
        this straight-through normalization do not represent derivatives of the original-unit soft threshold.
        Finite differences of the hard threshold do not match the surrogate gradient. The wider estimate is computed
        only when ``x`` requires a gradient. With the default ``slow_and_differentiable=False``
        the threshold has no gradient. On both paths the thresholded image is ``x * (x > threshold)``, whose gradient
        with respect to ``x`` is that mask.

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
        # `result > 0` is not the mask: it is False for a kept pixel whose value is 0 or below. Recompute the comparison
        # `forward` applies, `x > threshold`, against the threshold of each image and channel of the flattened input.
        x_flattened, _ = module.transform_input(x)
        return (x_flattened > threshold[:, None]).reshape(x.shape), threshold

    return result, threshold
