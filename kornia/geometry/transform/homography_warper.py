from __future__ import annotations

from abc import abstractmethod
from typing import Any, Optional

import torch.nn.functional as F

from kornia.core import Module, Tensor
from kornia.utils import create_meshgrid

from .imgwarp import homography_warp, warp_grid

__all__ = ["HomographyWarper", "BaseWarper"]


class BaseWarper(Module):
    def __init__(self, height: int, width: int, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.height = height
        self.width = width

    @abstractmethod
    def forward(self, patch_src: Tensor, src_homo_dst: Optional[Tensor] = None) -> Tensor: ...

    @abstractmethod
    def precompute_warp_grid(self, src_homo_dst: Tensor) -> None: ...


class HomographyWarper(BaseWarper):
    r"""Warp tensors by homographies.

    .. math::

        X_{dst} = H_{src}^{\{dst\}} * X_{src}

    Args:
        height: The height of the destination tensor.
        width: The width of the destination tensor.
        mode: interpolation mode to calculate output values ``'bilinear'`` | ``'nearest'``.
        padding_mode: padding mode for outside grid values
          ``'zeros'`` | ``'border'`` | ``'reflection'``.
        normalized_coordinates: whether to use a grid with normalized coordinates.
        align_corners: interpolation flag.
    """

    _warped_grid: Optional[Tensor]

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
        # create base grid to compute the flow
        self.grid = create_meshgrid(height, width, normalized_coordinates=normalized_coordinates)

        # initialice the warped destination grid
        self._warped_grid = None

    def precompute_warp_grid(self, src_homo_dst: Tensor) -> None:
        r"""Compute and store internally the transformations of the points.

        Useful when the same homography/homographies are reused.

        Args:
            src_homo_dst: Homography or homographies (stacked) to
              transform all points in the grid. Shape of the homography
              has to be :math:`(1, 3, 3)` or :math:`(N, 1, 3, 3)`.
              The homography assumes normalized coordinates [-1, 1] if
              normalized_coordinates is True.
        """
        self._warped_grid = warp_grid(self.grid, src_homo_dst)

    def forward(self, patch_src: Tensor, src_homo_dst: Optional[Tensor] = None) -> Tensor:
        r"""Warp a tensor from source into reference frame.

        Args:
            patch_src: The tensor to warp.
            src_homo_dst: The homography or stack of
              homographies from destination to source. The homography assumes
              normalized coordinates [-1, 1] if normalized_coordinates is True.

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
            warped_patch = F.grid_sample(
                patch_src,
                _warped_grid,
                mode=self.mode,
                padding_mode=self.padding_mode,
                align_corners=self.align_corners,
            )
        else:
            raise RuntimeError(
                "Unknown warping. If homographies are not provided                                 they must be preset"
                " using the method:                                 precompute_warp_grid()."
            )

        return warped_patch
