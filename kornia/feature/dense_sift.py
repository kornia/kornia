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
# limitations under the License.

"""Sparse SIFT features assembled from shared DenseSIFT image-pyramid maps."""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F
from torch import nn

from kornia.constants import pi
from kornia.core.check import KORNIA_CHECK_LAF, KORNIA_CHECK_SHAPE
from kornia.core.utils import _l2_normalize
from kornia.filters import get_gaussian_kernel2d, spatial_gradient
from kornia.geometry.transform import pyrdown

from .laf import (
    _grid_sample_patches,
    _promoted_grid_dtype,
    laf_is_valid,
    rotate_laf,
)
from .siftdesc import _dense_sift_histograms_from_gradients, _rootsift, get_sift_pooling_kernel


class DenseSIFTFeature(nn.Module):
    """Orient and describe LAFs using one shared DenseSIFT Gaussian pyramid.

    The module combines the shared histogram maps of `VLFeat DSIFT
    <https://www.vlfeat.org/api/dsift.html>`_ with orientation-relative spatial and
    angular binning as in OpenCV's `SIFT implementation
    <https://github.com/opencv/opencv/blob/4.x/modules/features2d/src/sift.simd.hpp>`_:
    gradients are binned once at each pyramid octave, then local orientation and
    descriptor histograms are sampled from those maps. Unlike patch-wise SIFT,
    no image patch is resampled before either stage.

    For a general affine LAF the global angular bins are remapped through the
    LAF's ``A.T`` gradient pullback, including its direction-dependent magnitude.
    This makes affine frames useful without extracting patches, but is an
    approximation: a histogram bin represents every gradient in its angular
    interval by its centre direction, and spatial pooling is performed in the
    image pyramid before the affine footprint is sampled.

    Args:
        num_ang_bins: Number of descriptor angular bins.
        num_spatial_bins: Number of descriptor spatial bins per axis.
        spatial_bin_size: Side of the DenseSIFT spatial pooling window, in pixels.
            The default 8 approximately matches the 19-pixel octave support to the patch SIFT
            descriptor's 16-pixel pooling bins.
        rootsift: If ``True``, apply RootSIFT after clipping and normalization.
        clipval: Descriptor clipping threshold.
        orientation_bins: Number of bins for dominant-orientation assignment.
        upright: Keep the supplied LAF orientation instead of assigning one.

    Shape:
        - image: :math:`(B, 1, H, W)`
        - lafs: :math:`(B, N, 2, 3)` in image pixel coordinates
        - output LAFs: :math:`(B, N, 2, 3)`
        - descriptors: :math:`(B, N, num_ang_bins * num_spatial_bins^2)`
    """

    def __init__(
        self,
        num_ang_bins: int = 8,
        num_spatial_bins: int = 4,
        spatial_bin_size: int = 8,
        rootsift: bool = True,
        clipval: float = 0.2,
        orientation_bins: int = 36,
        upright: bool = False,
    ) -> None:
        super().__init__()
        if num_ang_bins < 2 or orientation_bins < 2 or num_spatial_bins < 1 or spatial_bin_size < 1:
            raise ValueError(
                "SIFT bin counts and spatial_bin_size must be positive (angular bin counts need at least 2)"
            )
        self.num_ang_bins = num_ang_bins
        self.num_spatial_bins = num_spatial_bins
        self.spatial_bin_size = spatial_bin_size
        self.rootsift = rootsift
        self.clipval = clipval
        self.orientation_bins = orientation_bins
        self.upright = upright
        self.eps = 1e-10
        pooling = get_sift_pooling_kernel(spatial_bin_size).reshape(1, 1, spatial_bin_size, spatial_bin_size)
        self.register_buffer("descriptor_pooling_kernel", pooling)
        # A 9x9 grid approximates integration over the roughly 19-pixel octave
        # support while keeping sparse orientation assignment inexpensive.
        weighting = get_gaussian_kernel2d((9, 9), (9.0 / 6.0, 9.0 / 6.0), True)
        self.register_buffer("orientation_weighting", weighting.reshape(1, 1, 9, 9), persistent=False)

    def _pyramid(self, image: torch.Tensor) -> list[torch.Tensor]:
        levels = [image]
        while min(levels[-1].shape[-2:]) >= 38:
            levels.append(pyrdown(levels[-1]))
        return levels

    @staticmethod
    def _sample(maps: torch.Tensor, level_image: torch.Tensor, lafs: torch.Tensor, patch_size: int) -> torch.Tensor:
        b, n = lafs.shape[:2]
        if n == 0:
            return maps.new_zeros(b, 0, maps.shape[1], patch_size * patch_size)
        coordinates = (2.0 * torch.arange(patch_size, device=lafs.device, dtype=lafs.dtype) + 1.0) / patch_size - 1.0
        yy, xx = torch.meshgrid(coordinates, coordinates, indexing="ij")
        local = torch.stack([xx, yy], dim=-1).reshape(1, 1, patch_size, patch_size, 2)
        points = torch.einsum("bnij,xyj->bnxyi", lafs[..., :2, :2], local[0, 0]) + lafs[..., :2, 2].view(b, n, 1, 1, 2)
        h, w = level_image.shape[-2:]
        grid = 2.0 * (points + 0.5) / points.new_tensor([w, h]) - 1.0
        grid = grid.reshape(b, n * patch_size, patch_size, 2)
        grid_dtype = _promoted_grid_dtype(maps.dtype, grid.dtype)
        sample_maps = maps.to(grid_dtype) if maps.dtype != grid_dtype else maps
        grid = grid.to(grid_dtype) if grid.dtype != grid_dtype else grid
        sampled = _grid_sample_patches(sample_maps, grid, level_image.shape[-2], level_image.shape[-1]).to(maps.dtype)
        return sampled.reshape(b, maps.shape[1], n, patch_size * patch_size).permute(0, 2, 1, 3)

    @staticmethod
    def _remap_angles(histograms: torch.Tensor, affine: torch.Tensor, output_bins: int) -> torch.Tensor:
        """Move globally binned gradients into each LAF's local coordinates."""
        b, n, source_bins, samples = histograms.shape
        output_dtype = histograms.dtype
        working_dtype = _promoted_grid_dtype(histograms.dtype, affine.dtype)
        histograms = histograms.to(working_dtype)
        affine = affine.to(working_dtype)
        angles = torch.arange(source_bins, dtype=histograms.dtype, device=histograms.device)
        angles = 2.0 * pi * angles / float(source_bins)
        directions = torch.stack([torch.cos(angles), torch.sin(angles)], dim=-1)
        transformed = torch.einsum("bnji,kj->bnki", affine, directions)
        magnitude = torch.linalg.vector_norm(transformed, dim=-1)
        local_angle = torch.atan2(transformed[..., 1], transformed[..., 0])
        position = (local_angle % (2.0 * pi)) * float(output_bins) / (2.0 * pi)
        lower = torch.floor(position).long() % output_bins
        upper = (lower + 1) % output_bins
        upper_weight = position - torch.floor(position)
        values = histograms * magnitude.unsqueeze(-1)
        output = histograms.new_zeros(b, n, output_bins, samples)
        output.scatter_add_(
            2, lower.unsqueeze(-1).expand(-1, -1, -1, samples), values * (1.0 - upper_weight).unsqueeze(-1)
        )
        output.scatter_add_(2, upper.unsqueeze(-1).expand(-1, -1, -1, samples), values * upper_weight.unsqueeze(-1))
        return output.to(output_dtype)

    def _select_levels(self, lafs: torch.Tensor, num_levels: int) -> torch.Tensor:
        affine = lafs[..., :2, :2]
        det = (affine[..., 0, 0] * affine[..., 1, 1] - affine[..., 0, 1] * affine[..., 1, 0]).abs().sqrt()
        # At an octave the 19-pixel orientation window covers approximately one LAF diameter.
        return (det / 9.5).clamp_min(self.eps).log2().floor().clamp(0, num_levels - 1).long()

    @staticmethod
    def _laf_at_level(lafs: torch.Tensor, pyramid: list[torch.Tensor], level: int) -> torch.Tensor:
        """Map pixel-coordinate LAFs through ``pyrdown``'s align-corners-false grid."""
        output = lafs
        for index in range(level):
            source_h, source_w = pyramid[index].shape[-2:]
            target_h, target_w = pyramid[index + 1].shape[-2:]
            scale = output.new_tensor([float(target_w) / source_w, float(target_h) / source_h])
            output = output.clone()
            output[..., :2, :2] *= scale.view(1, 1, 2, 1)
            output[..., :, 2] = (output[..., :, 2] + 0.5) * scale - 0.5
        return output

    def _orientation(
        self, pyramid: list[torch.Tensor], histograms: list[torch.Tensor], lafs: torch.Tensor, levels: torch.Tensor
    ) -> torch.Tensor:
        b, n = lafs.shape[:2]
        result = lafs.new_zeros(b, n, self.orientation_bins)
        for level_index, (image, hist) in enumerate(zip(pyramid, histograms)):
            weighting = self.orientation_weighting.to(dtype=hist.dtype, device=hist.device).reshape(1, 1, 1, -1)
            for batch_index in range(b):
                selected = (levels[batch_index] == level_index).nonzero().flatten()
                if selected.numel() == 0:
                    continue
                current_lafs = lafs[batch_index : batch_index + 1, selected]
                level_lafs = self._laf_at_level(current_lafs, pyramid, level_index)
                # Histogram remapping is linear, so weight the sampled support before
                # the per-LAF angular pullback and avoid a 9x9 remapped temporary.
                sampled = (
                    self._sample(
                        hist[batch_index : batch_index + 1], image[batch_index : batch_index + 1], level_lafs, 9
                    )
                    * weighting
                ).sum(-1)
                mapped = self._remap_angles(
                    sampled.unsqueeze(-1), level_lafs[..., :2, :2], self.orientation_bins
                ).squeeze(-1)
                result[batch_index, selected] = mapped[0].to(result.dtype)
        smoothed = F.conv1d(
            F.pad(result.reshape(b * n, 1, -1), (2, 2), mode="circular"),
            result.new_tensor([1, 4, 6, 4, 1]).view(1, 1, -1) / 16.0,
        )
        smoothed = smoothed.squeeze(1)
        index = smoothed.argmax(dim=1)
        left = smoothed.gather(1, ((index - 1) % self.orientation_bins).unsqueeze(-1)).squeeze(-1)
        center = smoothed.gather(1, index.unsqueeze(-1)).squeeze(-1)
        right = smoothed.gather(1, ((index + 1) % self.orientation_bins).unsqueeze(-1)).squeeze(-1)
        denominator = left + right - 2.0 * center
        safe_denominator = torch.where(denominator != 0, denominator, torch.ones_like(denominator))
        offset = torch.where(denominator != 0, 0.5 * (left - right) / safe_denominator, torch.zeros_like(denominator))
        return 2.0 * pi * (index.to(lafs.dtype).reshape(b, n) + offset.reshape(b, n)) / float(self.orientation_bins)

    def _descriptors(
        self, pyramid: list[torch.Tensor], histograms: list[torch.Tensor], lafs: torch.Tensor, levels: torch.Tensor
    ) -> torch.Tensor:
        b, n = lafs.shape[:2]
        result = lafs.new_zeros(b, n, self.num_ang_bins, self.num_spatial_bins * self.num_spatial_bins)
        for level_index, (image, hist) in enumerate(zip(pyramid, histograms)):
            kernel = self.descriptor_pooling_kernel.to(dtype=hist.dtype, device=hist.device)
            pooled = F.conv2d(
                hist.reshape(-1, 1, hist.shape[-2], hist.shape[-1]), kernel, padding=self.spatial_bin_size // 2
            ).reshape(
                hist.shape[0],
                hist.shape[1],
                hist.shape[-2] + (self.spatial_bin_size % 2 == 0),
                hist.shape[-1] + (self.spatial_bin_size % 2 == 0),
            )
            if self.spatial_bin_size % 2 == 0:
                # An even pooling kernel is centred between pixel centres.  Average
                # its four neighbours so map index (y, x) stays registered to image
                # pixel (y, x), which is the LAF/grid-sampling convention.
                hist = 0.25 * (
                    pooled[:, :, :-1, :-1] + pooled[:, :, 1:, :-1] + pooled[:, :, :-1, 1:] + pooled[:, :, 1:, 1:]
                )
            else:
                hist = pooled
            for batch_index in range(b):
                selected = (levels[batch_index] == level_index).nonzero().flatten()
                if selected.numel() == 0:
                    continue
                current_lafs = lafs[batch_index : batch_index + 1, selected]
                level_lafs = self._laf_at_level(current_lafs, pyramid, level_index)
                sampled = self._sample(
                    hist[batch_index : batch_index + 1],
                    image[batch_index : batch_index + 1],
                    level_lafs,
                    self.num_spatial_bins,
                )
                mapped = self._remap_angles(sampled, level_lafs[..., :2, :2], self.num_ang_bins)
                result[batch_index, selected] = mapped[0].to(result.dtype)
        desc = result.flatten(2)
        desc = _l2_normalize(desc, dim=-1).clamp(0.0, self.clipval)
        desc = _l2_normalize(desc, dim=-1)
        if self.rootsift:
            return _rootsift(desc.reshape(-1, desc.shape[-1]), self.eps).reshape_as(desc)
        return desc

    def forward(self, image: torch.Tensor, lafs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Assign orientations and compute descriptors from a shared image pyramid.

        Args:
            image: Grayscale floating-point image of shape ``(B, 1, H, W)``.
            lafs: Pixel-coordinate frames of shape ``(B, N, 2, 3)``.

        Returns:
            Oriented frames in the input LAF dtype and descriptors in the image
            dtype, both on the image device. Invalid frames are preserved and
            receive zero descriptors. Empty frame sets are supported.
        """
        KORNIA_CHECK_SHAPE(image, ["B", "1", "H", "W"])
        KORNIA_CHECK_LAF(lafs)
        if image.shape[0] != lafs.shape[0]:
            raise ValueError(f"image and lafs must have the same batch size. Got {image.shape[0]} and {lafs.shape[0]}")
        image_dtype, laf_dtype = image.dtype, lafs.dtype
        lafs = lafs.to(device=image.device)
        if image.shape[0] == 0 or lafs.shape[1] == 0:
            return lafs, image.new_zeros(image.shape[0], lafs.shape[1], self.num_ang_bins * self.num_spatial_bins**2)
        # Promote before constructing coordinates: upcasting an already rounded
        # half-precision grid cannot recover subpixel geometry on large images.
        working_dtype = _promoted_grid_dtype(image.dtype, lafs.dtype)
        lafs = lafs.to(working_dtype)
        if image.dtype in (torch.float16, torch.bfloat16):
            image = image.float()
        valid = laf_is_valid(lafs)
        identity = torch.eye(2, 3, device=lafs.device, dtype=lafs.dtype).view(1, 1, 2, 3)
        # Padded detector slots and invalid training frames must never reach
        # grid_sample/atan2.  Their public output remains the zero descriptor.
        safe_lafs = torch.where(valid.view(*valid.shape, 1, 1), lafs, identity)
        pyramid = self._pyramid(image)
        histograms = []
        for level_image in pyramid:
            gradients = spatial_gradient(level_image, "diff")
            histograms.append(
                _dense_sift_histograms_from_gradients(
                    gradients[:, :, 0], gradients[:, :, 1], self.orientation_bins, self.eps
                )
            )
        levels = self._select_levels(safe_lafs, len(pyramid))
        if self.upright:
            oriented_lafs = safe_lafs
        else:
            angles = self._orientation(pyramid, histograms, safe_lafs, levels)
            # Kornia's LAF rotation is clockwise in image coordinates.  Rotate the
            # frame against its dominant local gradient so it becomes horizontal.
            oriented_lafs = rotate_laf(safe_lafs, -torch.rad2deg(angles).unsqueeze(-1))
        descriptors = self._descriptors(pyramid, histograms, oriented_lafs, levels)
        oriented_lafs = torch.where(valid.view(*valid.shape, 1, 1), oriented_lafs, lafs)
        descriptors = descriptors.masked_fill(~valid.unsqueeze(-1), 0.0)
        return oriented_lafs.to(laf_dtype), descriptors.to(image_dtype)
