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

from abc import abstractmethod
from typing import Any, Optional

import torch
from torch import nn

from kornia.core.utils import _torch_grid_sample
from kornia.geometry.grid import create_meshgrid

from .imgwarp import _normalize_warped_pixel_grid, _warp_pixel_grid, homography_warp, warp_grid

__all__ = ["BaseWarper", "HomographyWarper"]


class BaseWarper(nn.Module):
    """Provide a base class for homography-based image warping.

    Convention:
        - subclasses receive ``src_homo_dst`` as the destination→source homography

    """

    def __init__(self, height: int, width: int, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.height = height
        self.width = width

    @abstractmethod
    def forward(self, patch_src: torch.Tensor, src_homo_dst: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Sample a source patch on this warper's destination grid.

        Args:
            patch_src: Source image or patch tensor with shape
                :math:`(B, C, H, W)`, where :math:`B` is batch size,
                :math:`C` is channels, :math:`H` is source height, and
                :math:`W` is source width.
            src_homo_dst: Optional homography with shape :math:`(B, 3, 3)`
                mapping destination coordinates into the source image. Concrete
                implementations may also use a grid precomputed from this
                matrix.

        Returns:
            Tensor sampled in the destination frame managed by this warper.
        """
        ...

    @abstractmethod
    def precompute_warp_grid(self, src_homo_dst: torch.Tensor) -> None:
        """Precompute a sampling grid for repeated homography warps.

        Args:
            src_homo_dst: Homography tensor with shape :math:`(B, 3, 3)` used
                to transform destination grid points into source coordinates.
        """
        ...


class HomographyWarper(BaseWarper):
    r"""Warp tensors by homographies.

    .. math::

        X_{src} = H_{src}^{\{dst\}} * X_{dst}

    Convention:
        See :doc:`Conventions & Pitfalls </get-started/conventions>` for homography direction, normalized
        coordinates and sampling conventions.

        - See the convention block of :func:`homography_warp`.
        - with ``normalized_coordinates=False`` a grid from :meth:`precompute_warp_grid` holds source pixel
          coordinates computed as in :func:`homography_warp` (in at least float32, unaffected by autocast), normalized
          at sampling time with the size of the patch passed to ``forward`` and then cast to the homography's dtype

    Args:
        height: The height of the destination torch.Tensor.
        width: The width of the destination torch.Tensor.
        mode: interpolation mode to calculate output values ``'bilinear'`` | ``'nearest'``.
        padding_mode: padding mode for outside grid values
          ``'zeros'`` | ``'border'`` | ``'reflection'``.
        normalized_coordinates: whether the homographies are in normalized :math:`[-1, 1]` coordinates
          (``True``) or in pixel coordinates (``False``), as in :func:`homography_warp`.
        align_corners: interpolation flag.

    """

    _warped_grid: Optional[torch.Tensor]
    _warped_grid_dtype: Optional[torch.dtype]

    def __init__(
        self,
        height: int,
        width: int,
        mode: str = "bilinear",
        padding_mode: str = "zeros",
        normalized_coordinates: bool = True,
        align_corners: bool = False,
    ) -> None:
        super().__init__(height, width)
        self.mode = mode
        self.padding_mode = padding_mode
        self.normalized_coordinates = normalized_coordinates
        self.align_corners = align_corners
        # create base grid to compute the flow, under the same normalization convention that
        # forward() will hand to grid_sample, so the precomputed path matches the direct one
        self.grid = create_meshgrid(
            height, width, normalized_coordinates=normalized_coordinates, align_corners=align_corners
        )

        # initialice the warped destination grid, and in pixel mode the dtype forward() returns it to
        self._warped_grid = None
        self._warped_grid_dtype = None

    def precompute_warp_grid(self, src_homo_dst: torch.Tensor) -> None:
        r"""Compute and store internally the transformations of the points.

        Useful when the same homography/homographies are reused.

        Args:
            src_homo_dst: Homography or homographies (stacked) to
              transform all points in the grid. Shape of the homography
              has to be :math:`(1, 3, 3)`, :math:`(N, 3, 3)` or :math:`(N, 1, 3, 3)`.
              The homography assumes normalized coordinates [-1, 1] if
              normalized_coordinates is True, and pixel coordinates otherwise.

        """
        if self.normalized_coordinates:
            self._warped_grid = warp_grid(self.grid, src_homo_dst)
            return
        # Source pixels, from a pixel grid built and warped as in homography_warp (in at least float32, unaffected by
        # autocast; not from self.grid, which a half default dtype has already rounded), kept until forward()
        # normalizes them with the patch size; the normalized grid then returns to the homography's dtype.
        self._warped_grid = _warp_pixel_grid(src_homo_dst, self.height, self.width)
        self._warped_grid_dtype = src_homo_dst.dtype

    def forward(self, patch_src: torch.Tensor, src_homo_dst: Optional[torch.Tensor] = None) -> torch.Tensor:
        r"""Warp a torch.Tensor from source into reference frame.

        Args:
            patch_src: The torch.Tensor to warp.
            src_homo_dst: The homography or torch.stack of
              homographies from destination to source. The homography assumes
              normalized coordinates [-1, 1] if normalized_coordinates is True,
              and pixel coordinates otherwise.

        Return:
            Patch sampled at locations from source to destination.

        Shape:
            - Input: :math:`(N, C, H, W)` and :math:`(N, 3, 3)`
            - Output: :math:`(N, C, H, W)`

        Example:
            >>> input = torch.rand(1, 3, 32, 32)
            >>> homography = torch.eye(3).view(1, 3, 3)
            >>> warper = HomographyWarper(32, 32)
            >>> # without precomputing the warp
            >>> output = warper(input, homography)  # NxCxHxW
            >>> # precomputing the warp
            >>> warper.precompute_warp_grid(homography)
            >>> output = warper(input)  # NxCxHxW

        """
        _warped_grid = self._warped_grid
        if src_homo_dst is not None:
            warped_patch = homography_warp(
                patch_src,
                src_homo_dst,
                (self.height, self.width),
                mode=self.mode,
                padding_mode=self.padding_mode,
                align_corners=self.align_corners,
                normalized_coordinates=self.normalized_coordinates,
            )
        elif _warped_grid is not None:
            if not _warped_grid.device == patch_src.device:
                raise TypeError(
                    "Patch and warped grid must be on the same device. Got"
                    f" patch.device: {patch_src.device} warped_grid.device: {_warped_grid.device}. Whether recall"
                    " precompute_warp_grid() with the correct device for the homograhy"
                    " or change the patch device."
                )
            if not self.normalized_coordinates:
                # the cached grid holds source pixels; normalize them with the size of the patch being sampled
                _warped_grid = _normalize_warped_pixel_grid(
                    _warped_grid, patch_src.shape[-2], patch_src.shape[-1], self.align_corners
                )
                # getattr: a module pickled before this attribute existed has no _warped_grid_dtype
                warped_grid_dtype = getattr(self, "_warped_grid_dtype", None)
                if warped_grid_dtype is not None:
                    _warped_grid = _warped_grid.to(warped_grid_dtype)
            warped_patch = _torch_grid_sample(
                patch_src,
                _warped_grid,
                mode=self.mode,
                padding_mode=self.padding_mode,
                align_corners=self.align_corners,
            )
        else:
            raise RuntimeError(
                "Unknown warping. If homographies are not provided they must be preset"
                " using the method: precompute_warp_grid()."
            )

        return warped_patch
