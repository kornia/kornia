import torch
import torch.nn as nn
from torch.autograd import Variable

from .pinhole import scale_pinhole, homography_i_H_ref


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
        pinholes (Tensor): The pinhole models for ith frame with shape `[Nx12]`.
        width (int): The width of the image to warp. Optional.
        height (int): The height of the image to warp. Optional.

    """

    def __init__(self, pinholes, width=None, height=None):
        super(DepthWarper, self).__init__()
        # TODO: add type and value checkings
        self.width = width
        self.height = height
        self._pinholes = pinholes
        self._i_Hs_ref = None  # to be filled later
        self._pinhole_ref = None  # to be filled later

    def compute_homographies(self, pinhole, scale=None):
        if scale is None:
            batch_size = pinhole.shape[0]
            scale = torch.ones(
                batch_size, 1).to(
                pinhole.device).type_as(pinhole)
        # TODO: add type and value checkings
        pinhole_ref = scale_pinhole(pinhole, scale)
        if self.width is None:
            self.width = pinhole_ref[..., 5]
        if self.height is None:
            self.height = pinhole_ref[..., 4]
        self._pinhole_ref = pinhole_ref
        # scale pinholes_i and compute homographies
        pinhole_i = scale_pinhole(self._pinholes, scale)
        self._i_Hs_ref = homography_i_H_ref(pinhole_i, pinhole_ref)

    def _compute_projection(self, x, y, invd):
        point = torch.FloatTensor([[x], [y], [1.0], [invd]]).to(x.device)
        flow = torch.matmul(self._i_Hs_ref, point)
        z = 1. / flow[:, :, 2]
        x = (flow[:, :, 0] * z)
        y = (flow[:, :, 1] * z)
        return torch.stack([x, y], 1)

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
        dx = torch.norm((xy_p1 - xy_m1), 2, dim=2) / 2.0
        dxdd = dx / (delta_d)  # pixel*(1/meter)
        # half pixel sampling, we're interested in the min for all cameras
        return torch.min(0.5 / dxdd)

    # compute grids

    def warp(self, inv_depth_ref, roi=None):
        # TODO: add type and value checkings
        assert self._i_Hs_ref is not None, 'call compute_homographies'
        if roi is None:
            roi = (0, int(self.height), 0, int(self.width))
        start_row, end_row, start_col, end_col = roi
        assert start_row < end_row
        assert start_col < end_col
        height, width = end_row - start_row, end_col - start_col
        area = width * height

        # take sub region
        inv_depth_ref = inv_depth_ref[..., start_row:end_row,
                                      start_col:end_col].contiguous()

        device = inv_depth_ref.device
        ones_x = torch.ones(height).to(device)
        ones_y = torch.ones(width).to(device)
        ones = torch.ones(area).to(device)

        x = torch.linspace(start_col, end_col - 1, width).to(device)
        y = torch.linspace(start_row, end_row - 1, height).to(device)

        xv = torch.ger(ones_x, x).view(area)
        yv = torch.ger(y, ones_y).view(area)

        grid = [xv, yv, ones, inv_depth_ref.view(area)]
        grid = torch.stack(grid, 0)
        batch_size = inv_depth_ref.shape[0]
        grid = grid.unsqueeze(0).expand(batch_size, -1, -1)

        flow = torch.matmul(self._i_Hs_ref, grid)
        assert len(flow.shape) == 3, flow.shape

        factor_x = (self.width - 1) / 2
        factor_y = (self.height - 1) / 2

        z = 1. / flow[:, 2]  # Nx(H*W)
        x = (flow[:, 0] * z - factor_x) / factor_x
        y = (flow[:, 1] * z - factor_y) / factor_y

        flow = torch.stack([x, y], 1)  # Nx2x(H*W)

        n, c, a = flow.shape
        flows = flow.view(n, c, height, width)  # Nx2xHxW
        return flows.permute(0, 2, 3, 1)  # NxHxWx2

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
            patch_i, self.warp(inv_depth_ref))


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
        >>> img_ref = tgm.depth_warp( \
        >>>     pinholes_i, pinhole_ref, inv_depth_ref, img_i)  # NxCxHxW
    """
    warper = DepthWarper(pinholes_i, width=width, height=height)
    warper.compute_homographies(pinhole_ref)
    return warper(inv_depth_ref, patch_i)
