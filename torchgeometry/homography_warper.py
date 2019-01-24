import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import create_meshgrid
from .conversions import transform_points


__all__ = [
    "HomographyWarper",
    "homography_warp",
]


# layer api

class HomographyWarper(nn.Module):
    """Warps patches by homographies.

    .. math::

        X_{dst} = H_{dst}^{src} * X_{src}

    Args:
        height (int): The height of the image to warp.
        width (int): The width of the image to warp.
        padding_mode (string): Either 'zeros' to replace out of bounds with
                               zeros or 'border' to choose the closest
                               border data.
    """

    def __init__(self, height, width, padding_mode='zeros'):
        super(HomographyWarper, self).__init__()
        self.width = width
        self.height = height
        self.padding_mode = padding_mode

        # create base grid to compute the flow
        self.grid = create_meshgrid(
            height, width, normalized_coordinates=True)

    def warp_grid(self, H):
        """
        :param H: Homography or homographies (stacked) to transform all points
                  in the grid.
        :returns: Tensor[1, Height, Width, 2] containing transformed points in
                  normalized images space.
        """
        batch_size = H.shape[0]  # expand grid to match the input batch size
        grid = self.grid.repeat(batch_size, 1, 1, 1)  # NxHxWx2
        if len(H.shape) == 3:  # local homography case
            H = H.view(batch_size, 1, 3, 3)        # NxHxWx3x3
        # perform the actual grid transformation,
        # the grid is copied to input device and casted to the same type
        flow = transform_points(H, grid.to(H.device).type_as(H))    # NxHxWx2
        return flow.view(batch_size, self.height, self.width, 2)    # NxHxWx2

    def forward(self, patch_src, dst_homo_src):
        """Warps an image or tensor from source into reference frame.

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
            >>> warper = tgm.HomographyWarper(32, 32)
            >>> output = warper(input, homography)  # NxCxHxW
        """
        if not dst_homo_src.device == patch_src.device:
            raise TypeError("Patch and homography must be on the same device. \
                            Got patch.device: {} dst_H_src.device: {}."
                            .format(patch_src.device, dst_homo_src.device))
        return torch.nn.functional.grid_sample(
            patch_src, self.warp_grid(dst_homo_src), mode='bilinear',
            padding_mode=self.padding_mode)


# functional api


def homography_warp(patch, dst_H_src, dsize, padding_mode='zeros'):
    """
    .. note:: Functional API for :class:`torgeometry.HomographyWarper`

    Warps patches by homographies.

    Args:
        patch (Tensor): The image or tensor to warp. Should be from source.
        dst_homo_src (Tensor): The homography or stack of homographies from
                               source to destination.
        dsize (tuple): The height and width of the image to warp.
        padding_mode (string): Either 'zeros' to replace out of bounds with
                               zeros or 'border' to choose the closest border
                               data.

    Return:
        Tensor: Patch sampled at locations from source to destination.

    Shape:
        - Input: :math:`(N, C, H, W)` and :math:`(N, 3, 3)`
        - Output: :math:`(N, C, H, W)`

    Example:
        >>> input = torch.rand(1, 3, 32, 32)
        >>> homography = torch.eye(3).view(1, 3, 3)
        >>> output = tgm.homography_warp(input, homography, (32, 32))  # NxCxHxW
    """
    height, width = dsize
    warper = HomographyWarper(height, width, padding_mode)
    return warper(patch, dst_H_src)
