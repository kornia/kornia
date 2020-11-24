from typing import Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from kornia.geometry.linalg import transform_points, inverse_transformation, compose_transformations
from kornia.geometry.conversions import convert_points_to_homogeneous
from kornia.geometry.conversions import normalize_pixel_coordinates
from kornia.geometry.camera import PinholeCamera
from kornia.geometry.camera import cam2pixel, pixel2cam
from kornia.utils import create_meshgrid


__all__ = [
    "depth_warp",
    "DepthWarper",
]


class DepthWarper(nn.Module):
    r"""Warps a patch by depth.

    .. math::
        P_{src}^{\{dst\}} = K_{dst} * T_{src}^{\{dst\}}

        I_{src} = \\omega(I_{dst}, P_{src}^{\{dst\}}, D_{src})

    Args:
        pinholes_dst (PinholeCamera): the pinhole models for the destination
          frame.
        height (int): the height of the image to warp.
        width (int): the width of the image to warp.
        mode (str): interpolation mode to calculate output values
          'bilinear' | 'nearest'. Default: 'bilinear'.
        padding_mode (str): padding mode for outside grid values
           'zeros' | 'border' | 'reflection'. Default: 'zeros'.
        align_corners(bool): interpolation flag. Default: True. See
          https://pytorch.org/docs/stable/nn.functional.html#torch.nn.functional.interpolate for detail
    """

    def __init__(self,
                 pinhole_dst: PinholeCamera,
                 height: int, width: int,
                 mode: str = 'bilinear',
                 padding_mode: str = 'zeros',
                 align_corners: bool = True):
        super(DepthWarper, self).__init__()
        # constructor members
        self.width: int = width
        self.height: int = height
        self.mode: str = mode
        self.padding_mode: str = padding_mode
        self.eps = 1e-6
        self.align_corners: bool = align_corners

        # state members
        self._pinhole_dst: PinholeCamera = pinhole_dst
        self._pinhole_src: Union[None, PinholeCamera] = None
        self._dst_proj_src: Union[None, torch.Tensor] = None

        self.grid: torch.Tensor = self._create_meshgrid(height, width)

    @staticmethod
    def _create_meshgrid(height: int, width: int) -> torch.Tensor:
        grid: torch.Tensor = create_meshgrid(
            height, width, normalized_coordinates=False)  # 1xHxWx2
        return convert_points_to_homogeneous(grid)  # append ones to last dim

    def compute_projection_matrix(
            self, pinhole_src: PinholeCamera) -> 'DepthWarper':
        r"""Computes the projection matrix from the source to destinaion frame.
        """
        if not isinstance(self._pinhole_dst, PinholeCamera):
            raise TypeError("Member self._pinhole_dst expected to be of class "
                            "PinholeCamera. Got {}"
                            .format(type(self._pinhole_dst)))
        if not isinstance(pinhole_src, PinholeCamera):
            raise TypeError("Argument pinhole_src expected to be of class "
                            "PinholeCamera. Got {}".format(type(pinhole_src)))
        # compute the relative pose between the non reference and the reference
        # camera frames.
        dst_trans_src: torch.Tensor = compose_transformations(
            self._pinhole_dst.extrinsics,
            inverse_transformation(pinhole_src.extrinsics),
        )

        # compute the projection matrix between the non reference cameras and
        # the reference.
        dst_proj_src: torch.Tensor = torch.matmul(
            self._pinhole_dst.intrinsics, dst_trans_src)

        # update class members
        self._pinhole_src = pinhole_src
        self._dst_proj_src = dst_proj_src
        return self

    def _compute_projection(self, x, y, invd):
        point = torch.FloatTensor([[[x], [y], [1.0], [invd]]])
        flow = torch.matmul(
            self._dst_proj_src, point.to(self._dst_proj_src.device))
        z = 1. / flow[:, 2]
        x = (flow[:, 0] * z)
        y = (flow[:, 1] * z)
        return torch.cat([x, y], 1)

    def compute_subpixel_step(self) -> torch.Tensor:
        """This computes the required inverse depth step to achieve sub pixel
        accurate sampling of the depth cost volume, per camera.

        Szeliski, Richard, and Daniel Scharstein.
        "Symmetric sub-pixel stereo matching." European Conference on Computer
        Vision. Springer Berlin Heidelberg, 2002.
        """
        delta_d = 0.01
        xy_m1 = self._compute_projection(self.width / 2, self.height / 2,
                                         1.0 - delta_d)
        xy_p1 = self._compute_projection(self.width / 2, self.height / 2,
                                         1.0 + delta_d)
        dx = torch.norm((xy_p1 - xy_m1), 2, dim=-1) / 2.0
        dxdd = dx / (delta_d)  # pixel*(1/meter)
        # half pixel sampling, we're interested in the min for all cameras
        return torch.min(0.5 / dxdd)

    def warp_grid(self, depth_src: torch.Tensor) -> torch.Tensor:
        """Computes a grid for warping a given the depth from the reference
        pinhole camera.

        The function `compute_projection_matrix` has to be called beforehand in
        order to have precomputed the relative projection matrices encoding the
        relative pose and the intrinsics between the reference and a non
        reference camera.
        """
        # TODO: add type and value checkings
        if self._dst_proj_src is None or self._pinhole_src is None:
            raise ValueError("Please, call compute_projection_matrix.")

        if len(depth_src.shape) != 4:
            raise ValueError("Input depth_src has to be in the shape of "
                             "Bx1xHxW. Got {}".format(depth_src.shape))

        # unpack depth attributes
        batch_size, _, height, width = depth_src.shape
        device: torch.device = depth_src.device
        dtype: torch.dtype = depth_src.dtype

        # expand the base coordinate grid according to the input batch size
        pixel_coords: torch.Tensor = self.grid.to(device).to(dtype).expand(
            batch_size, -1, -1, -1)  # BxHxWx3

        # reproject the pixel coordinates to the camera frame
        cam_coords_src: torch.Tensor = pixel2cam(
            depth_src,
            self._pinhole_src.intrinsics_inverse().to(dtype),
            pixel_coords)  # BxHxWx3

        # reproject the camera coordinates to the pixel
        pixel_coords_src: torch.Tensor = cam2pixel(
            cam_coords_src, self._dst_proj_src.to(dtype))  # (B*N)xHxWx2

        # normalize between -1 and 1 the coordinates
        pixel_coords_src_norm: torch.Tensor = normalize_pixel_coordinates(
            pixel_coords_src, self.height, self.width)
        return pixel_coords_src_norm

    def forward(  # type: ignore
            self,
            depth_src: torch.Tensor,
            patch_dst: torch.Tensor) -> torch.Tensor:
        """Warps a tensor from destination frame to reference given the depth
        in the reference frame.

        Args:
            depth_src (torch.Tensor): the depth in the reference frame. The
              tensor must have a shape :math:`(B, 1, H, W)`.
            patch_dst (torch.Tensor): the patch in the destination frame. The
              tensor must have a shape :math:`(B, C, H, W)`.

        Return:
            torch.Tensor: the warped patch from destination frame to reference.

        Shape:
            - Output: :math:`(N, C, H, W)` where C = number of channels.

        Example:
            >>> # pinholes camera models
            >>> pinhole_dst = PinholeCamera(torch.randn(1, 4, 4), torch.randn(1, 4, 4),
            ... torch.tensor([32]), torch.tensor([32]))
            >>> pinhole_src = PinholeCamera(torch.randn(1, 4, 4), torch.randn(1, 4, 4),
            ... torch.tensor([32]), torch.tensor([32]))
            >>> # create the depth warper, compute the projection matrix
            >>> warper = DepthWarper(pinhole_dst, 32, 32)
            >>> _ = warper.compute_projection_matrix(pinhole_src)
            >>> # warp the destionation frame to reference by depth
            >>> depth_src = torch.ones(1, 1, 32, 32)  # Nx1xHxW
            >>> image_dst = torch.rand(1, 3, 32, 32)  # NxCxHxW
            >>> image_src = warper(depth_src, image_dst)  # NxCxHxW
        """
        return F.grid_sample(patch_dst, self.warp_grid(depth_src),  # type: ignore
                             mode=self.mode, padding_mode=self.padding_mode,
                             align_corners=self.align_corners)


# functional api

def depth_warp(pinhole_dst: PinholeCamera,
               pinhole_src: PinholeCamera,
               depth_src: torch.Tensor,
               patch_dst: torch.Tensor,
               height: int, width: int,
               align_corners: bool = True):
    r"""Function that warps a tensor from destination frame to reference
    given the depth in the reference frame.

    See :class:`~kornia.geometry.warp.DepthWarper` for details.

    Example:
        >>> # pinholes camera models
        >>> pinhole_dst = PinholeCamera(torch.randn(1, 4, 4), torch.randn(1, 4, 4),
        ... torch.tensor([32]), torch.tensor([32]))
        >>> pinhole_src = PinholeCamera(torch.randn(1, 4, 4), torch.randn(1, 4, 4),
        ... torch.tensor([32]), torch.tensor([32]))
        >>> # warp the destionation frame to reference by depth
        >>> depth_src = torch.ones(1, 1, 32, 32)  # Nx1xHxW
        >>> image_dst = torch.rand(1, 3, 32, 32)  # NxCxHxW
        >>> image_src = depth_warp(pinhole_dst, pinhole_src, depth_src, image_dst, 32, 32)  # NxCxHxW
    """
    warper = DepthWarper(pinhole_dst, height, width, align_corners=align_corners)
    warper.compute_projection_matrix(pinhole_src)
    return warper(depth_src, patch_dst)
