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

from typing import Optional, Tuple, Union

import torch

from kornia.core.check import KORNIA_CHECK
from kornia.utils.image import perform_keep_shape_image


class ThreshOtsu(torch.nn.Module):
    """Otsu thresholding module for PyTorch tensors."""

    def __init__(self, nbins: int = 256) -> None:
        """Initialize the ThreshOtsu module.

        Args:
            nbins (int, optional): Number of bins for histogram computation. Default is 256.

        Attributes:
            nbins (int): Number of bins for histogram computation.
            _threshold (Union[float, torch.Tensor]): Otsu-computed threshold, initialized to -1.
        """
        super().__init__()
        self.nbins: int = nbins
        self._threshold: Union[float, torch.Tensor] = -1

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

        min_val = xs.min()
        max_val = xs.max()

        histograms = []
        for i in range(xs.shape[0]):
            hist, bin_edges = torch.histogram(
                input=xs[i],
                bins=bins,
                range=(min_val.item(), max_val.item())
            )
            histograms.append(hist / hist.sum())

        return torch.stack(histograms), bin_edges

    @property
    def threshold(self) -> Union[float, torch.Tensor]:
        """Return the Otsu-computed threshold."""
        return self._threshold

    @threshold.setter
    def threshold(self, value: Union[float, torch.Tensor]) -> None:
        """Manually set the threshold."""
        self._threshold = value

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
            raise ValueError(
                f"Unsupported tensor dimensionality: {dimensionality}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply Otsu thresholding to the input x.

        Args:
            x (torch.Tensor): Image or batch of images to threshold.
            use_thresh (bool): If True, use the already computed threshold.

        Returns:
            torch.Tensor: Thresholded image.
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

        # Normalize data to range [0, self.nbins-1] for histogram, or use min/max for histc directly.
        # It's generally better to work with original values and iterate through potential thresholds.

        histograms, bin_edges = ThreshOtsu.__histogram(
            x_flattened, bins=self.nbins)
        best_thresholds = torch.zeros(
            nchannel, device=x_flattened.device, dtype=x.dtype)

        # Iterate over each image/flattened channel in the batch
        for i in range(nchannel):
            # Get histogram and bin edges for the current image/channel
            hist = histograms[i]
            current_bin_edges = bin_edges  # Bin edges are global in this updated __histogram

            max_inter_class_var = -1.0
            optimal_thresh_val = 0.0

            sum_bg = 0.0
            weight_bg = 0.0

            # Sum of pixels for foreground (initially total sum)
            sum_fg = torch.sum(
                hist * torch.arange(self.nbins, device=x.device).to(hist.dtype))

            # Iterate over each possible threshold (each bin)
            for t in range(self.nbins):
                bin_value = (t * hist[t]).item()

                # Update background
                sum_bg += bin_value
                weight_bg += hist[t].item()

                # Update foreground
                sum_fg -= bin_value
                weight_fg = 1.0 - weight_bg

                if weight_bg == 0 or weight_fg == 0:  # Avoid division by zero
                    continue

                mean_bg = sum_bg / weight_bg
                mean_fg = sum_fg / weight_fg

                # Calculate inter-class variance
                inter_class_var = weight_bg * \
                    weight_fg * ((mean_bg - mean_fg) ** 2)

                if inter_class_var > max_inter_class_var:
                    max_inter_class_var = inter_class_var
                    optimal_thresh_val = current_bin_edges[t + 1]

            best_thresholds[i] = optimal_thresh_val

        self._threshold = best_thresholds

        x_out, _ = self.transform_input(x.clone())
        # Apply threshold to each flattened image/channel
        for i in range(nchannel):
            x_out[i][x_out[i] <= best_thresholds[i]] = 0

        return x_out.reshape(orig_shape)


@perform_keep_shape_image
def otsu_threshold(
    x: torch.Tensor,
    nbins: int = 256,
    return_mask: bool = False,
) -> torch.Tensor:
    r"""Apply automatic image thresholding using Otsu algorithm to the input tensor.

    Args:
        x (Tensor): Input tensor (image or batch of images).
        nbins (int): Number of bins for histogram computation, default is 256.
        return_mask (bool): If True, return a binary mask indicating the thresholded pixels. If False,
            return the thresholded image.

    Returns:
        Tensor: Thresholded image or binary mask.

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
        >>> image = torch.tensor([[[0.4963, 0.7682, 0.0885, 0.1320, 0.3074, 0.6341],
        ...                       [0.4901, 0.8964, 0.4556, 0.6323, 0.3489, 0.4017],
        ...                       [0.0223, 0.1689, 0.2939, 0.5185, 0.6977, 0.8000],
        ...                       [0.1610, 0.2823, 0.6816, 0.9152, 0.3971, 0.8742]]])
        >>> image
        tensor([[[0.4963, 0.7682, 0.0885, 0.1320, 0.3074, 0.6341],
                 [0.4901, 0.8964, 0.4556, 0.6323, 0.3489, 0.4017],
                 [0.0223, 0.1689, 0.2939, 0.5185, 0.6977, 0.8000],
                 [0.1610, 0.2823, 0.6816, 0.9152, 0.3971, 0.8742]]])
        >>> otsu_threshold(image)
        tensor([[[0.0000, 0.7682, 0.0000, 0.0000, 0.0000, 0.6341],
                 [0.0000, 0.8964, 0.0000, 0.6323, 0.0000, 0.0000],
                 [0.0000, 0.0000, 0.0000, 0.0000, 0.6977, 0.8000],
                 [0.0000, 0.0000, 0.6816, 0.9152, 0.0000, 0.8742]]])
    """
    module = ThreshOtsu(nbins=nbins)

    result = module(x)
    if return_mask:
        return result > 0
    return result
