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
from kornia.enhance.histogram import histogram as diff_histogram


class OtsuThreshold(torch.nn.Module):
    """Otsu thresholding module for PyTorch tensors.

    Convention:
        See the Convention block on :func:`~kornia.filters.otsu_threshold`. ``forward`` has no ``return_mask`` and
        always returns the thresholded tensor with the thresholds.
    """

    def __init__(self) -> None:
        """Initialize the OtsuThreshold module."""
        super().__init__()

    @staticmethod
    def __histogram(xs: torch.Tensor, bins: int, diff: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute a histogram for each row of xs, CUDA compatible.

        Args:
            xs (torch.Tensor): 2D tensor (n, N) with values to histogram.
            bins (int): Number of bins.
            diff: denote if the differentiable histagram will be used. Default: False

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
        # histc divides [min, max] into `bins` intervals. The KDE path instead samples `bins` locations.
        edge_count = bins if diff else bins + 1
        edge_dtype = torch.float64 if xs.dtype == torch.float64 else torch.float32

        for i in range(xs.shape[0]):
            min_val = min_values[i].item()
            max_val = max_values[i].item()
            edges = torch.linspace(min_val, max_val, edge_count, device=xs.device, dtype=edge_dtype)
            if min_val == max_val:
                # No split exists; avoid histc's automatic range expansion for large constant values.
                hist = torch.zeros(bins, device=xs.device, dtype=edge_dtype)
                hist[0] = 1
            elif diff:
                hist = diff_histogram(xs[i].view(1, -1), edges, torch.tensor(0.001, device=xs.device)).squeeze()
            else:
                hist = _torch_histc_cast(xs[i], bins=bins, min=min_val, max=max_val)

            histograms.append(hist / hist.sum())
            bin_edges.append(edges)

        return torch.stack(histograms), torch.stack(bin_edges)

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
            slow_and_differentiable (bool, optional): If True, estimate the histogram with
                :func:`~kornia.enhance.histogram` instead of :func:`torch.histc`. Default is False.

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

        # Compute histogram and bin edges
        histograms, bin_edges = self.__histogram(x_flattened, bins=nbins, diff=slow_and_differentiable)

        # Vectorized computation of optimal thresholds
        bin_values = torch.arange(nbins, device=histograms.device, dtype=torch.float32)
        total_weight = torch.sum(histograms, dim=1)  # Shape: (nchannel,)
        total_sum = torch.sum(histograms * bin_values, dim=1)  # Shape: (nchannel,)
        cumsum_weight = torch.cumsum(histograms, dim=1)  # Shape: (nchannel, nbins)
        cumsum_sum = torch.cumsum(histograms * bin_values, dim=1)  # Shape: (nchannel, nbins)

        # Compute weights and sums for background and foreground
        weight_bg = cumsum_weight[:, :-1]  # Shape: (nchannel, nbins-1)
        sum_bg = cumsum_sum[:, :-1]  # Shape: (nchannel, nbins-1)
        weight_fg = total_weight[:, None] - weight_bg  # Shape: (nchannel, nbins-1)
        sum_fg = total_sum[:, None] - sum_bg  # Shape: (nchannel, nbins-1)

        # Compute means, avoiding division by zero
        mean_bg = torch.where(weight_bg > 0, sum_bg / weight_bg, torch.tensor(0.0, device=histograms.device))
        mean_fg = torch.where(weight_fg > 0, sum_fg / weight_fg, torch.tensor(0.0, device=histograms.device))

        # Compute inter-class variance, setting invalid cases to -1
        valid = (weight_bg > 0) & (weight_fg > 0)
        inter_class_var = torch.where(
            valid, weight_bg * weight_fg * (mean_bg - mean_fg) ** 2, torch.tensor(-1.0, device=histograms.device)
        )

        # Find the maximum inter-class variance and corresponding threshold
        t_max = torch.argmax(inter_class_var, dim=1)  # Shape: (nchannel,)
        max_var = inter_class_var.gather(1, t_max[:, None]).squeeze(1)  # Shape: (nchannel,)
        upper_edges = bin_edges.gather(1, (t_max + 1)[:, None]).squeeze(1)
        if not x.is_floating_point():
            # An integer pixel on or above the upper edge is counted in the foreground, so the integer threshold is
            # the largest integer below that edge. Truncating toward zero would round a negative edge up, and keep
            # an integer edge, and drop the level just above the split from the foreground.
            upper_edges = upper_edges.ceil() - 1
        best_thresholds = torch.where(max_var > 0, upper_edges, bin_edges[:, 0]).to(x.dtype)

        # Preserve a constant plane's exact input value, including integers outside floating-point precision.
        plane_min = x_flattened.amin(dim=1).detach()
        plane_max = x_flattened.amax(dim=1).detach()
        best_thresholds = torch.where(plane_min == plane_max, plane_min, best_thresholds)

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

    Convention:
        - One threshold is returned per plane over the last two axes: a :math:`(B, C, H, W)` input gives ``B * C``
          thresholds, flattened, and a tensor of at most two dimensions is one plane.
        - Each plane is histogrammed on its own range, ``nbins`` bins between its minimum and its maximum, not over a
          fixed :math:`[0, 1]` or :math:`[0, 255]`. On the default path the threshold is the upper edge of the
          selected histogram bin, in input units and in the input's dtype. For an integer input it is the largest
          integer below that value, so ``x > threshold`` keeps every pixel on or above it. Among splits of equal
          between-class variance the lowest wins. A constant plane uses its constant value as the threshold.
        - The foreground is ``x > threshold``, strictly. The first output is ``x * (x > threshold)``, not a
          binary image, and its gradient is that mask.
        - Known defects:

          - the minimum and maximum are taken over the whole input, so every plane is histogrammed on the joint
            range and its threshold depends on the other images and channels in the call
            (`#5172 <https://github.com/kornia/kornia/issues/5172>`_).
          - the threshold is read from ``linspace(min, max, nbins)``, not the histogram's bin edges, so it sits up to
            one bin above the split; at ``nbins=2`` nothing is kept
            (`#5172 <https://github.com/kornia/kornia/issues/5172>`_).
          - a constant plane, which has no split, gets the threshold 0 whatever its value
            (`#5172 <https://github.com/kornia/kornia/issues/5172>`_).
          - ``return_mask=True`` returns ``result > 0``, so foreground pixels of value 0 or below are reported as
            background (`#5173 <https://github.com/kornia/kornia/issues/5173>`_).
          - ``slow_and_differentiable=True`` only swaps in a kernel density estimate of fixed bandwidth ``1e-3``: the
            threshold still has no gradient, and pixels far from the ``nbins`` sample points are missed
            (`#5174 <https://github.com/kornia/kornia/issues/5174>`_).

    Args:
        x (Tensor): Input tensor (image or batch of images) of at most five dimensions.
        nbins (int): Number of bins for histogram computation, default is 256.
        slow_and_differentiable (bool): If True, estimate the histogram with
            :func:`~kornia.enhance.histogram` instead of :func:`torch.histc`. Default is False.
        return_mask (bool): If True, return the boolean mask ``x > threshold`` in place of the thresholded image,
            with each pixel compared against the threshold of its own image and channel. If False, return the
            thresholded image.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Thresholded tensor, or the boolean mask ``x > threshold`` when
        ``return_mask`` is True, either with the shape of ``x``, and the computed threshold values. The thresholded
        tensor cannot tell a kept pixel of value 0 from a dropped one; use the mask for that.

    Raises:
        ValueError: If the input tensor has more than five dimensions.
        ~kornia.core.exceptions.BaseError: If the input dtype is not supported.

    .. note::
        You can find more information about the Otsu algorithm here: https://en.wikipedia.org/wiki/Otsu's_method

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
