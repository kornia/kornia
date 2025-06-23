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
            _early_stop (float): Fraction of max iterations without improvement before early stopping.
        """
        super().__init__()
        self.nbins: int = nbins

        self._threshold: Union[float, torch.Tensor] = -1
        self._early_stop: float = 0.1

    @staticmethod
    def __rfill(x: torch.Tensor, length: int, dim: int = 0) -> torch.Tensor:
        """Right-fill a tensor with zeros to reach a given length along a dimension.

        Args:
            x (torch.Tensor): Tensor to pad.
            length (int): Target length.
            dim (int): Dimension to pad.

        Returns:
            torch.Tensor: Padded tensor.
        """
        return torch.cat([x, torch.zeros(max(length - x.shape[dim], 0), device=x.device, dtype=x.dtype)], dim=dim)

    @staticmethod
    def __histogram(xs: torch.Tensor, bins: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute a histogram for each row of xs, CUDA compatible.

        Args:
            xs (torch.Tensor): 2D tensor (n, N) with values to histogram.
            bins (int): Number of bins.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Histograms and bin edges.
        """
        min_val, max_val = xs.min(), xs.max()
        counts = []
        boundaries = []
        for lx in xs:
            counts.append(
                ThreshOtsu.__rfill(
                    torch.histc(input=lx, bins=bins, min=min_val.item(), max=max_val.item()),
                    length=int(max_val.item() - min_val.item()),
                )
            )
            boundaries.append(
                ThreshOtsu.__rfill(
                    torch.linspace(min_val.item(), max_val.item(), bins + 1),
                    length=int(max_val.item() - min_val.item()) + 1,
                )
            )
        return torch.stack(counts).to(xs.device), torch.stack(boundaries).to(xs.device)

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
            raise ValueError(f"Unsupported tensor dimensionality: {dimensionality}")

    def forward(self, x: torch.Tensor, use_thresh: bool = False) -> torch.Tensor:
        """Apply Otsu thresholding to the input x.

        Args:
            x (torch.Tensor): Image or batch of images to threshold.
            use_thresh (bool): If True, use the already computed threshold.

        Returns:
            torch.Tensor: Thresholded image.
        """
        x, orig_shape = self.transform_input(x)

        if use_thresh and self._threshold > 0:
            features = x.to(torch.float64)
            features[features < self._threshold] = 0
            return features.reshape(orig_shape)

        # Check tensor type compatibility
        if x.device.type == "cpu":
            # Error of torch.histc on CPU with int and uint types
            KORNIA_CHECK(
                x.dtype in [torch.float32, torch.float64, torch.float16, torch.bfloat16],
                "Tensor dtype not supported for Otsu thresholding: only float types are supported on CPU.",
            )
        else:
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

        min_value = x.min(dim=-1)[0]
        x = (x.T - min_value).T  # Shift values to start at 0

        histogram, bin_edges = ThreshOtsu.__histogram(x, bins=self.nbins)
        nb_px = x[0, :].numel()

        # Initialize Otsu variables
        px_bellow = torch.zeros((x.shape[0]), device=x.device, requires_grad=False)
        best_thresh = torch.zeros((x.shape[0]), device=x.device, requires_grad=False)
        max_intra_class_var = torch.zeros((x.shape[0]), device=x.device, requires_grad=False)

        # Initialize mean variables
        mu0 = torch.zeros((x.shape[0]), device=x.device, requires_grad=False)
        mu1 = torch.zeros((x.shape[0]), device=x.device, requires_grad=False)

        x = x.to(torch.float64)

        max_try = int(self.nbins * self._early_stop)
        no_update = 0

        # Iterate over each bin to find the best threshold
        for bin_num in range(self.nbins):
            px_bellow += histogram[:, bin_num]

            w0 = px_bellow / nb_px
            w1 = 1 - w0

            thresh = bin_edges[:, bin_num + 1]
            condition = x <= thresh.repeat(1, nb_px).reshape(x.shape)

            for idx, (x_, cond) in enumerate(zip(x, condition)):
                mu0[idx] = torch.var(x_[cond])
                mu1[idx] = torch.var(x_[~cond])

            intra_class_var = w0 * w1 * torch.pow(mu0 - mu1, 2)

            update_condition = intra_class_var > max_intra_class_var
            max_intra_class_var[update_condition] = intra_class_var[update_condition]
            best_thresh[update_condition] = thresh[update_condition]

            if torch.all(~update_condition):
                no_update += 1
                if no_update >= max_try:
                    break
            else:
                no_update = 0

        # Apply the found threshold
        x = (x.T + min_value).T
        x[x < (best_thresh + min_value).repeat(1, nb_px).reshape(x.shape)] = 0
        self._threshold = best_thresh + min_value

        return x.reshape(orig_shape)


@perform_keep_shape_image
def otsu_threshold(
    x: torch.Tensor,
    nbins: int = 256,
    return_mask: bool = False,
    use_thresh: bool = False,
    threshold: Optional[Union[float, torch.Tensor]] = None,
) -> torch.Tensor:
    r"""Apply automatic image thresholding using Otsu algorithm to the input tensor.

    Args:
        x (Tensor): Input tensor (image or batch of images).
        nbins (int): Number of bins for histogram computation, default is 256.
        return_mask (bool): If True, return a binary mask indicating the thresholded pixels. If False, \\
            return the thresholded image.
        use_thresh (bool): If True, use the already computed threshold.
        threshold (Optional[Union[float, Tensor]]): Manually set threshold value.

    Returns:
        Tensor: Thresholded image.

    Raises:
        ValueError: If the input tensor has unsupported dimensionality or dtype.

    .. note::
        - The input tensor should be of type `torch.uint8`, `torch.int8`, `torch.int16`, `torch.int32`, or \\
            `torch.int64`.
        - If `use_thresh` is True, the threshold must have been computed previously and set in the module.
        - If `threshold` is provided, it overrides the computed threshold.

    .. note::
        You may found more information about the Otsu algorithm here: https://en.wikipedia.org/wiki/Otsu's_method

    Example:
        >>> import torch
        >>> from kornia.filters.otsu_thresholding import otsu_threshold
        >>> image = torch.tensor([[[0.4963, 0.7682, 0.0885, 0.1320, 0.3074, 0.6341],
                 [0.4901, 0.8964, 0.4556, 0.6323, 0.3489, 0.4017],
                 [0.0223, 0.1689, 0.2939, 0.5185, 0.6977, 0.8000],
                 [0.1610, 0.2823, 0.6816, 0.9152, 0.3971, 0.8742]]])
        >>> image
        tensor([[[0.4963, 0.7682, 0.0885, 0.1320, 0.3074, 0.6341],
                 [0.4901, 0.8964, 0.4556, 0.6323, 0.3489, 0.4017],
                 [0.0223, 0.1689, 0.2939, 0.5185, 0.6977, 0.8000],
                 [0.1610, 0.2823, 0.6816, 0.9152, 0.3971, 0.8742]]])
        >>> otsu_threshold(image, nbins=3)
        tensor([[[0.4963, 0.7682, 0.0000, 0.0000, 0.0000, 0.6341],
                 [0.4901, 0.8964, 0.4556, 0.6323, 0.3489, 0.4017],
                 [0.0000, 0.0000, 0.0000, 0.5185, 0.6977, 0.8000],
                 [0.0000, 0.0000, 0.6816, 0.9152, 0.3971, 0.8742]]],
               dtype=torch.float64)
    """
    module = ThreshOtsu(nbins=nbins)
    if threshold is not None:
        module.threshold = threshold

    result = module(x, use_thresh=use_thresh)
    if return_mask:
        return result > 0
    return result
