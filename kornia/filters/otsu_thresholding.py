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
from kornia.utils.helpers import _torch_histc_cast


class OtsuThreshold(torch.nn.Module):
    """Otsu thresholding module for PyTorch tensors."""

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

        min_val = xs.min()
        max_val = xs.max()

        histograms = []
        bin_edges = torch.linspace(min_val.item(), max_val.item(), bins, device=xs.device)

        for i in range(xs.shape[0]):
            if diff:
                hist = diff_histogram(xs[i].view(1, -1), bin_edges, torch.tensor(0.001)).squeeze()
            else:
                # Use torch.histc for non-differentiable histogram
                # Note: torch.histogram is in PyTorch 1.10+, and should replace histc in future versions when
                #       no longer supporting older pytorch versions.
                hist = _torch_histc_cast(xs[i], bins=bins, min=min_val.item(), max=max_val.item())

            # Normalize and append the histogram
            histograms.append(hist / hist.sum())

        return torch.stack(histograms), bin_edges

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
        elif dimensionality == 3:
            return x.flatten(start_dim=1), original_shape
        elif dimensionality == 4:
            b, c, h, w = x.shape
            return self.transform_input(x.reshape(b * c, h, w), original_shape=original_shape)
        elif dimensionality == 5:
            f, b, c, h, w = x.shape
            return self.transform_input(x.reshape(f * b * c, h, w), original_shape=original_shape)
        else:
            raise ValueError(f"Unsupported tensor dimensionality: {dimensionality}")

    def forward(
        self, x: torch.Tensor, nbins: int = 256, slow_and_differentiable: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply Otsu thresholding to the input x.

        Args:
            x (torch.Tensor): Image or batch of images to threshold.
            nbins (int, optional): Number of bins for histogram computation. Default is 256.
            slow_and_differentiable (bool, optional): If True, use a differentiable histogram computation. Default is False.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Thresholded tensor, threshold values.
        """
        x_flattened, orig_shape = self.transform_input(x)
        nchannel = x_flattened.shape[0]

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

        histograms, bin_edges = OtsuThreshold.__histogram(x_flattened, bins=nbins, diff=slow_and_differentiable)
        best_thresholds = torch.zeros(nchannel, device=x_flattened.device, dtype=x.dtype)

        # Iterate over each image/flattened channel in the batch
        for i in range(nchannel):
            # Get histogram and bin edges for the current image/channel
            hist = histograms[i]
            current_bin_edges = bin_edges

            max_inter_class_var = -1.0
            optimal_thresh_val = 0.0

            sum_bg = 0.0
            weight_bg = 0.0

            # Sum of pixels for foreground (initially total sum)
            sum_fg = torch.sum(hist * torch.arange(nbins, device=x.device).to(hist.dtype))

            # Iterate over each possible threshold (each bin)
            for t in range(nbins):
                bin_value = (t * hist[t]).item()

                # Update background
                sum_bg += bin_value
                weight_bg += hist[t].item()

                # Update foreground
                sum_fg -= bin_value
                weight_fg = 1.0 - weight_bg

                # Avoid division by zero
                if weight_bg == 0 or weight_fg == 0:
                    continue

                mean_bg = sum_bg / weight_bg
                mean_fg = sum_fg / weight_fg

                # Calculate inter-class variance
                inter_class_var = weight_bg * weight_fg * ((mean_bg - mean_fg) ** 2)

                if inter_class_var > max_inter_class_var:
                    max_inter_class_var = inter_class_var
                    optimal_thresh_val = current_bin_edges[t + 1].item()

            best_thresholds[i] = optimal_thresh_val

        x_out, _ = self.transform_input(x.clone())
        # Apply threshold to each flattened image/channel
        for i in range(nchannel):
            x_out[i][x_out[i] <= best_thresholds[i]] = 0

        return x_out.reshape(orig_shape), best_thresholds


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
