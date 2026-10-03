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
from kornia.enhance.histogram import histogram as diff_histogram


class OtsuThreshold(torch.nn.Module):
    """Otsu thresholding module for PyTorch tensors."""

    def __init__(self) -> None:
        """Initialize the OtsuThreshold module."""
        super().__init__()

    @staticmethod
    def __histogram(
        xs: torch.Tensor, bins: int, diff: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Compute a histogram for each row of xs, CUDA compatible.

        Args:
            xs (torch.Tensor): 2D tensor (n, N) with values to histogram.
            bins (int): Number of bins.
            diff: Whether to use a differentiable histogram. Default: False.

        Returns:
            Normalized histograms, bin edges, and each pixel's bin index (None for the KDE path).
        """
        edge_dtype = torch.float64 if xs.dtype == torch.float64 else torch.float32
        if not diff:
            # Keep ranges on the device: scalar extraction and a data-dependent constant-plane branch prevent
            # torch.export and full-graph compilation. As with histc, histogram construction does not propagate grads.
            values = xs.detach().to(edge_dtype)
            min_values = values.amin(dim=1, keepdim=True)
            max_values = values.amax(dim=1, keepdim=True)
            widths = max_values - min_values
            safe_widths = torch.where(widths > 0, widths, torch.ones_like(widths))
            indices = ((values - min_values) * bins / safe_widths).to(torch.int64).clamp(0, bins - 1)
            histograms = values.new_zeros((values.shape[0], bins)).scatter_add(1, indices, torch.ones_like(values))

            # Match linspace's symmetric interpolation, including its exact end points and fused multiply-add.
            positions = torch.arange(bins + 1, device=xs.device, dtype=edge_dtype)
            steps = widths / bins
            bin_edges = torch.where(
                positions < (bins + 1) // 2,
                torch.addcmul(min_values, positions, steps),
                torch.addcmul(max_values, positions - bins, steps),
            )
            return histograms / histograms.sum(dim=1, keepdim=True), bin_edges, indices

        # The KDE path samples `bins` locations rather than histc's `bins + 1` interval edges.
        if not xs.is_floating_point():
            xs = xs.to(torch.float32)
        min_values = xs.amin(dim=1)
        max_values = xs.amax(dim=1)
        histograms_list = []
        bin_edges_list = []
        for i in range(xs.shape[0]):
            min_val = min_values[i].item()
            max_val = max_values[i].item()
            edges = torch.linspace(min_val, max_val, bins, device=xs.device, dtype=edge_dtype)
            if min_val == max_val:
                # No split exists; avoid the KDE's zero-range case for constant values.
                hist = torch.zeros(bins, device=xs.device, dtype=edge_dtype)
                hist[0] = 1
            else:
                hist = diff_histogram(xs[i].view(1, -1), edges, torch.tensor(0.001, device=xs.device)).squeeze()
            histograms_list.append(hist / hist.sum())
            bin_edges_list.append(edges)

        return torch.stack(histograms_list), torch.stack(bin_edges_list), None

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
            slow_and_differentiable (bool, optional): If True, use a differentiable histogram computation.
                Default is False.

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
        histograms, bin_edges, indices = self.__histogram(x_flattened, bins=nbins, diff=slow_and_differentiable)

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
        if not slow_and_differentiable:
            # Moving through an empty bin cannot change either class. Exclude repeated splits so floating-point
            # reduction roundoff cannot move the selected edge across an empty gap.
            valid = valid & (histograms[:, :-1] > 0)
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
        if x.is_floating_point():
            if indices is None:
                # Preserve the KDE edge convention, lowering only a collision with a foreground pixel.
                boundary = best_thresholds
            else:
                # Use actual histogram membership to guard the strict comparison. Interpolated edges can round
                # onto a pixel, including when a compiler changes fused multiply-add, or when casting to half.
                # Keep the edge whenever it already separates the classes, otherwise fit it between their pixels.
                foreground = indices > t_max[:, None]
                boundary = torch.where(foreground, x_flattened, float("inf")).amin(dim=1).detach()
                background_max = torch.where(foreground, -float("inf"), x_flattened).amax(dim=1).detach()
                best_thresholds = torch.where(
                    max_var > 0, torch.maximum(best_thresholds, background_max), best_thresholds
                )
            if x.dtype in (torch.float16, torch.bfloat16):
                # MPS on older supported torch versions has no half/bfloat16 nextafter kernel. Both formats have
                # monotonically ordered magnitude bits; advance negative values and decrement positive values.
                # Widen the integer arithmetic explicitly so older Inductor versions preserve the final bitcast.
                bits = boundary.view(torch.int16).to(torch.int32)
                direction = torch.where(boundary > 0, 1, -1).to(torch.int32)
                previous = torch.where(boundary == 0, -32767, bits - direction).to(torch.int16).view(x.dtype)
            elif x.device.type == "mps":
                # Metal Inductor does not lower nextafter, so step the float32 representation directly as well.
                bits = boundary.view(torch.int32)
                direction = torch.where(boundary > 0, 1, -1).to(torch.int32)
                previous = torch.where(boundary == 0, -2147483647, bits - direction).view(x.dtype)
            else:
                previous = torch.nextafter(boundary, torch.full_like(boundary, -float("inf")))
            if x.device.type == "mps" and x.dtype in (torch.float32, torch.bfloat16):
                # MPS flushes these dtypes' subnormals in comparisons. Use the smallest normal negative value so
                # the strict comparison can still include a foreground pixel whose value is zero.
                previous = torch.where(boundary == 0, -torch.finfo(x.dtype).tiny, previous)
            if indices is None:
                on_threshold = (x_flattened == best_thresholds[:, None]).any(dim=1)
                lower_threshold = on_threshold & (best_thresholds.to(upper_edges.dtype) >= upper_edges) & (max_var > 0)
                best_thresholds = torch.where(lower_threshold, previous, best_thresholds)
            else:
                # Clamp against the predecessor itself: compilers may defer the half cast until after the
                # comparison, so comparing with the foreground value can miss an edge that will round up to it.
                bounded_thresholds = torch.minimum(
                    best_thresholds.to(bin_edges.dtype), previous.to(bin_edges.dtype)
                ).to(x.dtype)
                best_thresholds = torch.where(max_var > 0, bounded_thresholds, best_thresholds)

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

    Each image/channel plane uses its own histogram range. On the default path, the threshold is the upper edge
    of the selected histogram bin. For an integer input it is the largest integer below that value. For floating
    input, rounding is corrected when necessary to keep the threshold at or above the largest background pixel
    and below the smallest foreground pixel, so ``x > threshold`` matches the histogram split. On MPS, a zero
    foreground boundary in float32 or bfloat16 uses the smallest normal negative value because comparisons flush
    subnormal values to zero. Empty bins do not introduce candidate splits. A constant plane uses its constant value
    as the threshold. The default path supports
    ``torch.export`` and full-graph ``torch.compile``.

    Args:
        x (Tensor): Input tensor (image or batch of images).
        nbins (int): Number of bins for histogram computation, default is 256.
        slow_and_differentiable (bool): If True, use a differentiable histogram computation. Default is False.
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
