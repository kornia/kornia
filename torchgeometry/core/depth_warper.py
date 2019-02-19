from typing import Optional, Iterable, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchgeometry.utils import create_meshgrid
from torchgeometry.core.transformations import relative_pose
from torchgeometry.core.conversions import transform_points
from torchgeometry.core.conversions import convert_points_to_homogeneous
from torchgeometry.core.pinhole import PinholeCamera, PinholeCamerasList


__all__ = [
    "depth_warp",
    "DepthWarper",
]


def normalize_pixel_coordinates(
        pixel_coordinates: torch.Tensor,
        height: float,
        width: float) -> torch.Tensor:
    r"""Normalize pixel coordinates between -1 and 1.

    Normalized, -1 if on extreme left, 1 if on extreme right (x = w-1).

    Args:
        pixel_coordinate (torch.Tensor): the grid with pixel coordinates.
          Shape must be BxHxWx2.
        width (int): the maximum width in the x-axis.
        height (int): the maximum height in the y-axis.

    Return:
        torch.Tensor: the nornmalized pixel coordinates.
    """
    if len(pixel_coordinates.shape) != 4 and pixel_coordinates.shape[-1] != 2:
        raise ValueError("Input pixel_coordinates must be of shape BxHxWx2. "
                         "Got {}".format(pixel_coordinates.shape))

    # unpack pixel coordinates
    u_coord, v_coord = torch.chunk(pixel_coordinates, dim=-1, chunks=2)

    # apply actual normalization
    factor_u: float = 2. / (width - 1)
    factor_v: float = 2. / (height - 1)
    u_coord_norm: torch.Tensor = factor_u * u_coord - 1.
    v_coord_norm: torch.Tensor = factor_v * v_coord - 1.

    # stack normalized coordinates and return
    pixel_coordinates_norm: torch.Tensor = torch.cat(
        [u_coord_norm, v_coord_norm], dim=-1)
    return pixel_coordinates_norm


# based on:
# https://github.com/ClementPinard/SfmLearner-Pytorch/blob/master/inverse_warp.py#L26

def pixel2cam(depth: torch.Tensor, intrinsics_inv: torch.Tensor,
              pixel_coords: torch.Tensor) -> torch.Tensor:
    r"""Transform coordinates in the pixel frame to the camera frame.

    Args:
        depth (torch.Tensor): the source depth maps. Shape must be Bx1xHxW.
        intrinsics_inv (torch.Tensor): the inverse intrinsics camera matrix.
          Shape must be Bx4x4.
        pixel_coords (torch.Tensor): the grid with the homogeneous camera
          coordinates. Shape must be BxHxWx3.

    Returns:
        torch.Tensor: array of (u, v, 1) cam coordinates with shape BxHxWx3.
    """
    if not len(depth.shape) == 4 and depth.shape[1] == 1:
        raise ValueError("Input depth has to be in the shape of "
                         "Bx1xHxW. Got {}".format(depth.shape))
    if not len(intrinsics_inv.shape) == 3:
        raise ValueError("Input intrinsics_inv has to be in the shape of "
                         "Bx4x4. Got {}".format(intrinsics_inv.shape))
    if not len(pixel_coords.shape) == 4 and pixel_coords.shape[3] == 3:
        raise ValueError("Input pixel_coords has to be in the shape of "
                         "BxHxWx3. Got {}".format(intrinsics_inv.shape))
    cam_coords: torch.Tensor = transform_points(
        intrinsics_inv[:, None], pixel_coords)
    return cam_coords * depth.permute(0, 2, 3, 1)


# based on
# https://github.com/ClementPinard/SfmLearner-Pytorch/blob/master/inverse_warp.py#L43

def cam2pixel(
        cam_coords_src: torch.Tensor,
        dst_proj_src: torch.Tensor,
        eps: Optional[float] = 1e-6) -> torch.Tensor:
    r"""Transform coordinates in the camera frame to the pixel frame.

    Args:
        cam_coords (torch.Tensor): pixel coordinates defined in the first
          camera coordinates system. Shape must be BxHxWx3.
        dst_proj_src (torch.Tensor): the projection matrix between the
          reference and the non reference camera frame. Shape must be Bx4x4.

    Returns:
        torch.Tensor: array of [-1, 1] coordinates of shape BxHxWx2.
    """
    if not len(cam_coords_src.shape) == 4 and cam_coords_src.shape[3] == 3:
        raise ValueError("Input cam_coords_src has to be in the shape of "
                         "BxHxWx3. Got {}".format(cam_coords_src.shape))
    if not len(dst_proj_src.shape) == 3 and dst_proj_src.shape[-2:] == (4, 4):
        raise ValueError("Input dst_proj_src has to be in the shape of "
                         "Bx4x4. Got {}".format(dst_proj_src.shape))
    b, h, w, _ = cam_coords_src.shape
    # apply projection matrix to points
    point_coords: torch.Tensor = transform_points(
        dst_proj_src[:, None], cam_coords_src)
    x_coord: torch.Tensor = point_coords[..., 0]
    y_coord: torch.Tensor = point_coords[..., 1]
    z_coord: torch.Tensor = point_coords[..., 2]

    # compute pixel coordinates
    u_coord: torch.Tensor = x_coord / (z_coord + eps)
    v_coord: torch.Tensor = y_coord / (z_coord + eps)

    # stack and return the coordinates, that's the actual flow
    pixel_coords_dst: torch.Tensor = torch.stack([u_coord, v_coord], dim=-1)
    return pixel_coords_dst  # (B*N)xHxWx2


class DepthWarper(nn.Module):
    """Warps a patch by depth.

    .. math::
        H_{src}^{dst} = K_{dst} * T_{src}^{dst} * K_{src}^{-1}

        I_{src} = \\omega(I_{dst}, H_{src}^{dst}, D_{src})

    Args:
        pinholes_dst (Iterable[PinholeCamera]): The pinhole models for the
          destination frame.
        height (int): The height of the image to warp.
        width (int): The width of the image to warp.

    """

    def __init__(self,
                 pinholes_dst: Iterable[PinholeCamera],
                 height: int, width: int):
        super(DepthWarper, self).__init__()
        self.width: int = width
        self.height: int = height
        self.eps = 1e-6

        self._pinhole_dst: PinholeCamerasList = PinholeCamerasList(pinholes_dst)
        self._pinhole_src: Union[None, PinholeCamera] = None
        self._dst_proj_src: Union[None, torch.Tensor] = None

        self.grid: torch.Tensor = self._create_meshgrid(height, width)

    @staticmethod
    def _create_meshgrid(height: int, width: int) -> torch.Tensor:
        grid: torch.Tensor = create_meshgrid(
            height, width, normalized_coordinates=False)  # 1xHxWx2
        return convert_points_to_homogeneous(grid)  # append ones to last dim

    # TODO(edgar): remove
    '''def compute_homographies(self, pinhole, scale=None):
        if scale is None:
            batch_size = pinhole.shape[0]
            scale = torch.ones(
                batch_size).to(
                pinhole.device).type_as(pinhole)
        # TODO: add type and value checkings
        pinhole_ref = scale_pinhole(pinhole, scale)
        if self.width is None:
            self.width = pinhole_ref[..., 5:6]
        if self.height is None:
            self.height = pinhole_ref[..., 4:5]
        self._pinhole_ref = pinhole_ref
        # scale pinholes_i and compute homographies
        pinhole_i = scale_pinhole(self._pinholes, scale)
        self._i_Hs_ref = homography_i_H_ref(pinhole_i, pinhole_ref)'''

    def compute_projection_matrix(self, pinhole_src: PinholeCamera):
        if not isinstance(pinhole_src, PinholeCamera):
            raise TypeError("Argument pinhole_src expected to be of class "
                            "PinholeCamera. Got {}".format(type(pinhole_src)))
        # compute the relative pose between the non reference and the reference
        # camera frames.
        batch_size: int = pinhole_src.batch_size
        num_cameras: int = self._pinhole_dst.num_cameras

        extrinsics_src: torch.Tensor = pinhole_src.extrinsics[:, None].expand(
            -1, num_cameras, -1, -1)  # BxNx4x4
        extrinsics_src = extrinsics_src.contiguous().view(-1, 4, 4)  # (B*N)x4x4
        extrinsics_dst: torch.Tensor = self._pinhole_dst.extrinsics.view(
            -1, 4, 4)  # (B*N)x4x4
        dst_trans_src: torch.Tensor = relative_pose(
            extrinsics_src, extrinsics_dst)

        # compute the projection matrix between the non reference cameras and
        # the reference.
        intrinsics_dst: torch.Tensor = self._pinhole_dst.intrinsics.view(
            -1, 4, 4)
        dst_proj_src: torch.Tensor = torch.matmul(
            intrinsics_dst, dst_trans_src)

        # update class member
        self._pinhole_src = pinhole_src
        self._dst_proj_src = dst_proj_src.view(
            pinhole_src.batch_size, -1, 4, 4)
        return self

    def _compute_projection(self, x, y, invd):
        point = torch.FloatTensor([[[x], [y], [1.0], [invd]]])
        flow = torch.matmul(
            self._dst_proj_src[:, 0], point.to(self._dst_proj_src.device))
        z = 1. / flow[:, 2]
        x = (flow[:, 0] * z)
        y = (flow[:, 1] * z)
        return torch.cat([x, y], 1)

    def compute_subpixel_step(self):
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

    # compute grids
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

        # create views from tensors to match with cam2pixel expected shapes
        cam_coords_src = cam_coords_src[:, None].expand(
            -1, self._pinhole_dst.num_cameras, -1, -1, -1)  # BxNxHxWx3
        cam_coords_src = cam_coords_src.contiguous().view(
            -1, height, width, 3)                          # (B*N)xHxWx3
        dst_proj_src: torch.Tensor = self._dst_proj_src.view(-1, 4, 4)

        # reproject the camera coordinates to the pixel
        pixel_coords_src: torch.Tensor = cam2pixel(
            cam_coords_src, dst_proj_src.to(dtype))  # (B*N)xHxWx2

        # normalize between -1 and 1 the coordinates
        pixel_coords_src_norm: torch.Tensor = normalize_pixel_coordinates(
            pixel_coords_src, self.height, self.width)
        return pixel_coords_src_norm

    def forward(
            self,
            depth_src: torch.Tensor,
            patch_dst: torch.Tensor) -> torch.Tensor:
        """Warps a tensor from destination frame to reference given the depth
        in the reference frame.

        Args:
            depth_src (torch.Tensor): The depth in the reference frame. The
              tensor must have a shape :math:`(B, 1, H, W)`.
            patch_dst (torch.Tensor): The patch in the destination frame. The
              tensor must have a shape :math:`(B, C, H, W)`.

        Return:
            torch.Tensor: The warped patch from destination frame to reference.

        Shape:
            - Output: :math:`(N, C, H, W)` where C = number of channels.

        Example:
            >>> # pinholes camera models
            >>> pinhole_dst = tgm.PinholeCamera(...)
            >>> pinhole_src = tgm.PinholeCamera(...)
            >>> # create the depth warper, compute the projection matrix
            >>> warper = tgm.DepthWarper([pinhole_dst, ], height, width)
            >>> warper.compute_projection_matrix(pinhole_src)
            >>> # warp the destionation frame to reference by depth
            >>> depth_src = torch.ones(1, 1, 32, 32)  # Nx1xHxW
            >>> image_dst = torch.rand(1, 3, 32, 32)  # NxCxHxW
            >>> image_src = warper(depth_src, image_dst)    # NxCxHxW
        """
        return F.grid_sample(patch_dst, self.warp_grid(depth_src))


# functional api

def depth_warp(pinholes_dst: Iterable[PinholeCamera],
               pinhole_src: PinholeCamera,
               depth_src: torch.Tensor,
               patch_dst: torch.Tensor,
               height: int, width: int):
    r"""Function that warps a tensor from destination frame to reference
    given the depth in the reference frame.

    See :class:`~torchgeometry.DepthWarper` for details.

    Example:
        >>> # pinholes camera models
        >>> pinhole_dst = tgm.PinholeCamera(...)
        >>> pinhole_src = tgm.PinholeCamera(...)
        >>> # warp the destionation frame to reference by depth
        >>> depth_src = torch.ones(1, 1, 32, 32)  # Nx1xHxW
        >>> image_dst = torch.rand(1, 3, 32, 32)  # NxCxHxW
        >>> image_src = tgm.depth_warp([pinhole_dst, ], pinhole_src,
        >>>     depth_src, image_dst, height, width)  # NxCxHxW
    """
    warper = DepthWarper(pinholes_dst, height, width)
    warper.compute_projection_matrix(pinhole_src)
    return warper(depth_src, patch_dst)
