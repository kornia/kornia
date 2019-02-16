from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchgeometry.utils import create_meshgrid
from torchgeometry.core.conversions import transform_points
from torchgeometry.core.conversions import convert_points_to_homogeneous
from torchgeometry.core.transformations import relative_pose
# NOTE: remove later
from torchgeometry.core.pinhole import PinholeCamera
from torchgeometry.core.pinhole import scale_pinhole, homography_i_H_ref


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
    cam_coords: torch.Tensor = transform_points(intrinsics_inv, pixel_coords)
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
    point_coords: torch.Tensor = transform_points(dst_proj_src, cam_coords_src)
    x_coord: torch.Tensor = point_coords[..., 0]
    y_coord: torch.Tensor = point_coords[..., 1]
    z_coord: torch.Tensor = point_coords[..., 2]

    # compute pixel coordinates
    u_coord: torch.Tensor = x_coord / (z_coord + eps)
    v_coord: torch.Tensor = y_coord / (z_coord + eps)

    # stack and return the coordinates, that's the actual flow
    pixel_coords_dst: torch.Tensor = torch.stack([u_coord, v_coord], dim=-1)
    return pixel_coords_dst  # BxHxWx2


class DepthWarper(nn.Module):
    """Warps a patch by inverse depth.

    .. math::
        H_{ref}^{i} = K_{i} * T_{ref}^{i} * K_{ref}^{-1}

        I_{ref} = \\omega(I_{i}, H_{ref}^{i}, \\xi_{ref})

    Args:
        pinholes (torch.Tensor): The pinhole models for ith frame with shape `[Nx12]`.
        width (int): The width of the image to warp. Optional.
        height (int): The height of the image to warp. Optional.

    """

    def __init__(self,
                 pinholes: torch.Tensor,
                 height: int,
                 width: int,
                 pinhole_dst):
        super(DepthWarper, self).__init__()
        self.width: int = width
        self.height: int = height
        self.eps = 1e-6

        # old parameters
        self._pinholes = pinholes
        self._i_Hs_ref = None  # to be filled later
        self._pinhole_ref = None  # to be filled later

        # new refactor parameters
        self.grid: torch.Tensor = self._create_meshgrid(height, width)
        self._pinhole_src = None   # to be filled later
        self._dst_proj_src = None  # to be filled later

        if not isinstance(pinhole_dst, PinholeCamera):
            raise TypeError("Argument pinhole_dst expected to be of class "
                            "PinholeCamera. Got {}".format(type(pinhole_dst)))
        self._pinhole_dst = pinhole_dst

    @staticmethod
    def _create_meshgrid(height: int, width: int) -> torch.Tensor:
        grid: torch.Tensor = create_meshgrid(
            height, width, normalized_coordinates=False)  # 1xHxWx2
        return convert_points_to_homogeneous(grid)  # append ones to last dim

    # TODO(edgar): refactor this and potentially and potentially accept the
    # pinholes models as Nx2x4x4, where first will be the pinhole_matrix in the
    # 4x4 shape, and second the Rt matrix in the 4x4 shape. For this, we need
    # to update inverse_pinhole, scale_pinhole, etc. But seems the more generic
    # thing to do. Additionally, I wouldn't force the output shape(self.height/width)
    # to be the pinhole camera content. Instead I would stick to the values from
    # the class signature which I would force to the user to introduce (now is
    # optional and could be None).
    def compute_homographies(self, pinhole, scale=None):
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
        self._i_Hs_ref = homography_i_H_ref(pinhole_i, pinhole_ref)

    def compute_projection_matrix(self, pinhole_src):
        # TODO: add documentation
        if not isinstance(pinhole_src, PinholeCamera):
            raise TypeError("Argument pinhole_src expected to be of class "
                            "PinholeCamera. Got {}".format(type(pinhole_dst)))
        # compute the relative pose between the non reference and the reference
        # camera frames.
        dst_trans_src = relative_pose(
            pinhole_src.extrinsics, self._pinhole_dst.extrinsics)

        # compute the projection matrix between the non reference cameras and
        # the reference.
        self._dst_proj_src = torch.matmul(
            self._pinhole_dst.intrinsics, dst_trans_src)

        # update class member
        self._pinhole_src = pinhole_src
        return self

    def _compute_projection(self, x, y, invd):
        point = torch.FloatTensor([[[x], [y], [1.0], [invd]]])
        #flow = torch.matmul(self._i_Hs_ref, point.to(self._i_Hs_ref.device))
        flow = torch.matmul(
            self._dst_proj_src, point.to(self._dst_proj_src.device))
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
            raise ValueError("Input depth_src has to be in the shape of"
                             "Bx1xHxW. Got {}".format(depth_src.shape))

        # unpack depth attributes
        batch_size, _, height, width = depth_src.shape
        device: torch.device = depth_src.device
        dtype: torch.dtype = depth_src.dtype

        # expand the base coordinate grid according to the input batch size
        pixel_coords: torch.Tensor = self.grid.to(device).to(dtype).expand(
            batch_size, -1, -1, -1)

        # reproject the pixel coordinates to the camera frame
        cam_coords_src: torch.Tensor = pixel2cam(
            depth_src,
            self._pinhole_src.intrinsics_inverse(),
            pixel_coords)  # BxHxWx3

        # reproject the camera coordinates to the pixel
        pixel_coords_src: torch.Tensor = cam2pixel(
            cam_coords_src, self._dst_proj_src)  # BxHxWx2

        # normalize between -1 and 1 the coordinates
        pixel_coords_src_norm: torch.Tensor = normalize_pixel_coordinates(
            pixel_coords_src, self.height, self.width)
        return pixel_coords_src_norm

    def warp_grid_old(
            self,
            inv_depth_ref: torch.Tensor,
            roi=None) -> torch.Tensor:
        """Computes a grid for warping a given the inverse depth from
        a reference pinhole camera.

        The function `compute_homographies` has to be called beforehand in
        order to have precomputed the relative transformations encoding the
        relative pose and the intrinsics between the reference and a non
        reference camera.
        """
        # TODO: add type and value checkings
        if self._i_Hs_ref is None:
            raise ValueError("Please, call compute_homographies.")

        if len(inv_depth_ref.shape) != 4:
            raise ValueError("Input inv_depth_ref has to be in the shape of"
                             "Bx1xHxW. Got {}".format(inv_depth_ref.shape))

        # NOTE: exclude this by now. We will figure out later if we want to
        # support this feature again.
        '''if roi is None:
            roi = (0, int(self.height), 0, int(self.width))
        start_row, end_row, start_col, end_col = roi
        assert start_row < end_row
        assert start_col < end_col
        height, width = end_row - start_row, end_col - start_col
        area = width * height

        # take sub region
        inv_depth_ref = inv_depth_ref[..., start_row:end_row,
                                      start_col:end_col].contiguous()'''

        batch_size, _, height, width = inv_depth_ref.shape
        device: torch.device = inv_depth_ref.device
        dtype: torch.dtype = inv_depth_ref.dtype

        # expand the base according to the number of the input batch size
        # and concatenate it with the inverse depth [x, y, inv_depth]'
        import pdb
        pdb.set_trace()
        grid: torch.Tensor = self.grid.to(device).to(dtype).expand(
            batch_size, -1, -1, -1)
        grid = torch.cat([grid, inv_depth_ref.permute(0, 2, 3, 1)], dim=-1)
        grid = grid.view(batch_size, 1, 1, height, width, 3)  # Nx1x1xHxWx3

        # reshape the transforms to match with the grid shape: Nx1x1x4x4
        i_trans_ref: torch.Tensor = torch.unsqueeze(
            torch.unsqueeze(self._i_Hs_ref, dim=1), dim=1).to(device)

        # transform the actual grid
        flow: torch.Tensor = transform_points(i_trans_ref, grid).view(
            batch_size, height, width, 3)

        # recover the the coordinates
        z: torch.Tensor = 1. / (flow[..., 2] + self.eps)  # NxHxW
        x: torch.Tensor = flow[..., 0] * z
        y: torch.Tensor = flow[..., 1] * z

        # in case we are using normalized coordinates in the range of [-1, 1]
        # we have to move the coordinates to the grid center.
        x_norm: torch.Tensor = x
        y_norm: torch.Tensor = y

        if self.normalized_coordinates:
            factor_x: float = 2. / (self.width - 1.)
            factor_y: float = 2. / (self.height - 1.)
            x_norm = factor_x * x - 1.
            y_norm = factor_y * y - 1.

        # stack and return the values
        return torch.stack([x_norm, y_norm], dim=-1)  # NxHxWx2

    def forward(self, inv_depth_ref, patch_i):
        """Warps an image or tensor from ith frame to reference given the
        inverse depth in the reference frame.

        Args:
            inv_depth_ref (Tensor): The inverse depth in the reference frame.
            patch_i (Tensor): The patch in the it frame.

        Return:
            Tensor: The warped data from ith frame to reference.

        Shape:
            - Input: :math:`(N, 1, H, W)` and :math:`(N, C, H, W)`
            - Output: :math:`(N, C, H, W)`

        Example:
            >>> # image in ith frame
            >>> img_i = torch.rand(1, 3, 32, 32)       # NxCxHxW
            >>> # pinholes models for camera i and reference
            >>> pinholes_i = torch.Tensor([1, 12])     # Nx12
            >>> pinhole_ref = torch.Tensor([1, 12]),   # Nx12
            >>> # create the depth warper and compute the homographies
            >>> warper = tgm.DepthWarper(pinholes_i)
            >>> warper.compute_homographies(pinhole_ref)
            >>> # warp the ith frame to reference by inverse depth
            >>> inv_depth_ref = torch.ones(1, 1, 32, 32)  # Nx1xHxW
            >>> img_ref = warper(inv_depth_ref, img_i)    # NxCxHxW
        """
        # TODO: add type and value checkings
        return torch.nn.functional.grid_sample(
            patch_i, self.warp_grid(inv_depth_ref))


# functional api

def depth_warp(pinholes_i, pinhole_ref, inv_depth_ref, patch_i,
               width=None, height=None):
    """
    .. note::
        Functional API for :class:`torgeometry.DepthWarper`

    Warps a patch by inverse depth.

    Args:
        pinholes_i (Tensor): The pinhole models for ith frame with
                             shape `[Nx12]`.
        pinholes_ref (Tensor): The pinhole models for the reference frame
                               with shape `[1x12]`.
        inv_depth_ref (Tensor): The inverse depth in the reference frame.
        patch_i (Tensor): The patch data in the ith frame.

    Return:
        Tensor: The warped data from ith frame to reference.

    Example:
        >>> # image in ith frame
        >>> img_i = torch.rand(1, 3, 32, 32)          # NxCxHxW
        >>> # pinholes models for camera i and reference
        >>> pinholes_i = torch.Tensor([1, 12])        # Nx12
        >>> pinhole_ref = torch.Tensor([1, 12]),      # Nx12
        >>> # warp the ith frame to reference by inverse depth in the reference
        >>> inv_depth_ref = torch.ones(1, 1, 32, 32)  # Nx1xHxW
        >>> img_ref = tgm.depth_warp(
        >>>     pinholes_i, pinhole_ref, inv_depth_ref, img_i)  # NxCxHxW
    """
    warper = DepthWarper(pinholes_i, width=width, height=height)
    warper.compute_homographies(pinhole_ref)
    return warper(inv_depth_ref, patch_i)
