from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from kornia.utils import create_meshgrid
from kornia.geometry.linalg import transform_points


__all__ = [
    "HomographyWarper",
    "homography_warp",
]


# layer api

class HomographyWarper(nn.Module):
    r"""Warp tensors by homographies.

    .. math::

        X_{dst} = H_{src}^{\{dst\}} * X_{src}

    Args:
        height (int): The height of the destination tensor.
        width (int): The width of the destination tensor.
        mode (str): interpolation mode to calculate output values
          'bilinear' | 'nearest'. Default: 'bilinear'.
        padding_mode (str): padding mode for outside grid values
          'zeros' | 'border' | 'reflection'. Default: 'zeros'.
        normalized_coordinates (bool): wether to use a grid with
          normalized coordinates.
    """

    def __init__(
            self,
            height: int,
            width: int,
            mode: str = 'bilinear',
            padding_mode: str = 'zeros',
            normalized_coordinates: bool = True) -> None:
        super(HomographyWarper, self).__init__()
        self.width: int = width
        self.height: int = height
        self.mode: str = mode
        self.padding_mode: str = padding_mode
        self.normalized_coordinates: bool = normalized_coordinates

        # create base grid to compute the flow
        self.grid: torch.Tensor = create_meshgrid(
            height, width, normalized_coordinates=normalized_coordinates)

        # initialice the warped destination grid
        self._warped_grid: Optional[torch.Tensor] = None

    def precompute_warp_grid(self, src_homo_dst: torch.Tensor) -> None:
        r"""Compute and store internaly the transformations of the points.

        Useful when the same homography/homographies are reused.

        Args:
            src_homo_dst (torch.Tensor): Homography or homographies (stacked) to
              transform all points in the grid. Shape of the homography
              has to be :math:`(1, 3, 3)` or :math:`(N, 1, 3, 3)`.
         """
        self._warped_grid = self.warp_grid(src_homo_dst)

    def warp_grid(self, src_homo_dst: torch.Tensor) -> torch.Tensor:
        r"""Compute the grid to warp the coordinates grid by the homography/ies.

        Args:
            src_homo_dst (torch.Tensor): Homography or homographies (stacked) to
              transform all points in the grid. Shape of the homography
              has to be :math:`(1, 3, 3)` or :math:`(N, 1, 3, 3)`.

        Returns:
            torch.Tensor: the transformed grid of shape :math:`(N, H, W, 2)`.
        """

        batch_size: int = src_homo_dst.shape[0]
        device: torch.device = src_homo_dst.device
        dtype: torch.dtype = src_homo_dst.dtype
        # expand grid to match the input batch size
        grid: torch.Tensor = self.grid.expand(batch_size, -1, -1, -1)  # NxHxWx2
        if len(src_homo_dst.shape) == 3:  # local homography case
            src_homo_dst = src_homo_dst.view(batch_size, 1, 3, 3)  # Nx1x3x3
        # perform the actual grid transformation,
        # the grid is copied to input device and casted to the same type
        flow: torch.Tensor = transform_points(
            src_homo_dst, grid.to(device).to(dtype))  # NxHxWx2
        return flow.view(batch_size, self.height, self.width, 2)  # NxHxWx2

    def forward(  # type: ignore
            self,
            patch_src: torch.Tensor,
            src_homo_dst: Optional[torch.Tensor] = None) -> torch.Tensor:
        r"""Warp a tensor from source into reference frame.

        Args:
            patch_src (torch.Tensor): The tensor to warp.
            src_homo_dst (torch.Tensor, optional): The homography or stack of
              homographies from destination to source. The homography assumes
              normalized coordinates [-1, 1]. Default: None.

        Return:
            torch.Tensor: Patch sampled at locations from source to destination.

        Shape:
            - Input: :math:`(N, C, H, W)` and :math:`(N, 3, 3)`
            - Output: :math:`(N, C, H, W)`

        Example:
            >>> input = torch.rand(1, 3, 32, 32)
            >>> homography = torch.eye(3).view(1, 3, 3)
            >>> warper = kornia.HomographyWarper(32, 32)
            >>> # without precomputing the warp
            >>> output = warper(input, homography)  # NxCxHxW
            >>> # precomputing the warp
            >>> warper.precompute_warp_grid(homography)
            >>> output = warper(input)  # NxCxHxW
        """
        warped_grid: torch.Tensor
        if src_homo_dst is not None:
            if not src_homo_dst.device == patch_src.device:
                raise TypeError("Patch and homography must be on the same device. \
                                Got patch.device: {} src_H_dst.device: {}."
                                .format(patch_src.device, src_homo_dst.device))
            warped_grid = self.warp_grid(src_homo_dst)
        elif self._warped_grid is not None:
            if not self._warped_grid.device == patch_src.device:
                raise TypeError("Patch and warped grid must be on the same device. \
                                Got patch.device: {} warped_grid.device: {}. Wheter \
                                recall precompute_warp_grid() with the correct device \
                                for the homograhy or change the patch device."
                                .format(patch_src.device, self._warped_grid.device))
            warped_grid = self._warped_grid
        else:
            raise RuntimeError("Unknown warping. If homographies are not provided \
                                they must be presetted using the method: \
                                precompute_warp_grid().")

        return F.grid_sample(patch_src, warped_grid,  # type: ignore
                             mode=self.mode, padding_mode=self.padding_mode,
                             align_corners=True)


# functional api


def homography_warp(patch_src: torch.Tensor,
                    src_homo_dst: torch.Tensor,
                    dsize: Tuple[int, int],
                    mode: str = 'bilinear',
                    padding_mode: str = 'zeros') -> torch.Tensor:
    r"""Function that warps image patchs or tensors by homographies.

    See :class:`~kornia.geometry.warp.HomographyWarper` for details.

    Args:
        patch_src (torch.Tensor): The image or tensor to warp. Should be from
                                  source of shape :math:`(N, C, H, W)`.
        src_homo_dst (torch.Tensor): The homography or stack of homographies
                                     from destination to source of shape
                                     :math:`(N, 3, 3)`.
        dsize (Tuple[int, int]): The height and width of the image to warp.
        mode (str): interpolation mode to calculate output values
          'bilinear' | 'nearest'. Default: 'bilinear'.
        padding_mode (str): padding mode for outside grid values
          'zeros' | 'border' | 'reflection'. Default: 'zeros'.

    Return:
        torch.Tensor: Patch sampled at locations from source to destination.

    Example:
        >>> input = torch.rand(1, 3, 32, 32)
        >>> homography = torch.eye(3).view(1, 3, 3)
        >>> output = kornia.homography_warp(input, homography, (32, 32))
    """
    height, width = dsize
    warper = HomographyWarper(height, width, mode, padding_mode)
    return warper(patch_src, src_homo_dst)
