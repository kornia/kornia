from typing import Union, Optional

import torch
import torch.nn as nn
from torch.autograd import Variable

from torchgeometry.utils import create_meshgrid
from torchgeometry.core.conversions import transform_points
from torchgeometry.core.conversions import convert_points_to_homogeneous
from torchgeometry.core.pinhole import scale_pinhole, homography_i_H_ref


__all__ = [
    "depth_warp",
    "DepthWarper",
]


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
                 height: Optional[Union[None, int]] = None,
                 width:  Optional[Union[None, int]] = None,
                 normalized_coordinates: Optional[bool] = True):
        super(DepthWarper, self).__init__()
        # TODO: add type and value checkings
        #self.width = width if width is None else torch.tensor(width)
        #self.height = height if height is None else torch.tensor(height)
        self.width = width
        self.height = height
        self._pinholes = pinholes
        self._i_Hs_ref = None  # to be filled later
        self._pinhole_ref = None  # to be filled later
        self.eps = 1e-6
        self.normalized_coordinates = normalized_coordinates

        self.grid = None  # to be filled later
        if height is not None and width is not None:
            self.grid: torch.Tensor = create_meshgrid(height, width,
                normalized_coordinates=self.normalized_coordinates)
    
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

    def _compute_projection(self, x, y, invd):
        point = torch.FloatTensor([[[x], [y], [1.0], [invd]]])
        flow = torch.matmul(self._i_Hs_ref, point.to(self._i_Hs_ref.device))
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

    def warp_grid(self, inv_depth_ref: torch.Tensor, roi=None) -> torch.Tensor:
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

        # compute the base grid to transform in case it does not exist,
        # otherwise, use the existing one.
        if self.grid is None:
            grid: torch.Tensor = create_meshgrid(height, width,
                normalized_coordinates=self.normalized_coordinates)
        else:
            grid: torch.Tensor = self.grid

        # expand the base according to the number of the input batch size
        # and concatenate it with the inverse depth [x, y, inv_depth]'
        grid = grid.expand(batch_size, -1, -1, -1)
        grid = grid.to(device).to(inv_depth_ref.dtype)
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

        if self.normalized_coordinates:
            factor_x: float = (self.width - 1.) / 2
            factor_y: float = (self.height - 1.) / 2
            x = (x - factor_x) / factor_x
            y = (y - factor_y) / factor_y

        # stack and return the values
        return torch.stack([x, y], dim=-1)  # NxHxWx2

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
