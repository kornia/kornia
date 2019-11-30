from typing import Tuple

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
    r"""Warps image patches or tensors by homographies.

    .. math::

        X_{dst} = H_{src}^{\{dst\}} * X_{src}

    Args:
        height (int): The height of the image to warp.
        width (int): The width of the image to warp.
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

    def warp_grid(self, dst_homo_src: torch.Tensor) -> torch.Tensor:
        r"""Computes the grid to warp the coordinates grid by an homography.

        Args:
            dst_homo_src (torch.Tensor): Homography or homographies (stacked) to
                              transform all points in the grid. Shape of the
                              homography has to be :math:`(N, 3, 3)`.

        Returns:
            torch.Tensor: the transformed grid of shape :math:`(N, H, W, 2)`.
        """
        batch_size: int = dst_homo_src.shape[0]
        device: torch.device = dst_homo_src.device
        dtype: torch.dtype = dst_homo_src.dtype
        # expand grid to match the input batch size
        grid: torch.Tensor = self.grid.expand(batch_size, -1, -1, -1)  # NxHxWx2
        if len(dst_homo_src.shape) == 3:  # local homography case
            dst_homo_src = dst_homo_src.view(batch_size, 1, 3, 3)  # NxHxWx3x3
        # perform the actual grid transformation,
        # the grid is copied to input device and casted to the same type
        flow: torch.Tensor = transform_points(
            dst_homo_src, grid.to(device).to(dtype))  # NxHxWx2
        return flow.view(batch_size, self.height, self.width, 2)  # NxHxWx2

    def forward(  # type: ignore
            self,
            patch_src: torch.Tensor,
            dst_homo_src: torch.Tensor) -> torch.Tensor:
        r"""Warps an image or tensor from source into reference frame.

        Args:
            patch_src (torch.Tensor): The image or tensor to warp.
                                      Should be from source.
            dst_homo_src (torch.Tensor): The homography or stack of homographies
             from source to destination. The homography assumes normalized
             coordinates [-1, 1].

        Return:
            torch.Tensor: Patch sampled at locations from source to destination.

        Shape:
            - Input: :math:`(N, C, H, W)` and :math:`(N, 3, 3)`
            - Output: :math:`(N, C, H, W)`

        Example:
            >>> input = torch.rand(1, 3, 32, 32)
            >>> homography = torch.eye(3).view(1, 3, 3)
            >>> warper = kornia.HomographyWarper(32, 32)
            >>> output = warper(input, homography)  # NxCxHxW
        """
        if not dst_homo_src.device == patch_src.device:
            raise TypeError("Patch and homography must be on the same device. \
                            Got patch.device: {} dst_H_src.device: {}."
                            .format(patch_src.device, dst_homo_src.device))
        return F.grid_sample(patch_src, self.warp_grid(dst_homo_src),
                             mode=self.mode, padding_mode=self.padding_mode)


# functional api


def homography_warp(patch_src: torch.Tensor,
                    dst_homo_src: torch.Tensor,
                    dsize: Tuple[int, int],
                    mode: str = 'bilinear',
                    padding_mode: str = 'zeros') -> torch.Tensor:
    r"""Function that warps image patchs or tensors by homographies.

    See :class:`~kornia.geometry.warp.HomographyWarper` for details.

    Args:
        patch_src (torch.Tensor): The image or tensor to warp. Should be from
                                  source of shape :math:`(N, C, H, W)`.
        dst_homo_src (torch.Tensor): The homography or stack of homographies
                                     from source to destination of shape
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
    return warper(patch_src, dst_homo_src)
