"""Module containing operators to work on RGB-Depth images."""
from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from kornia.filters.sobel import spatial_gradient
from kornia.geometry.camera import cam2pixel, PinholeCamera, pixel2cam, project_points, unproject_points
from kornia.geometry.conversions import normalize_pixel_coordinates
from kornia.geometry.linalg import (
    compose_transformations,
    convert_points_to_homogeneous,
    inverse_transformation,
    transform_points,
)
from kornia.utils import create_meshgrid

__all__ = ["depth_to_3d", "depth_to_normals", "warp_frame_depth", "depth_warp", "DepthWarper"]


def depth_to_3d(depth: torch.Tensor, camera_matrix: torch.Tensor, normalize_points: bool = False) -> torch.Tensor:
    """Compute a 3d point per pixel given its depth value and the camera intrinsics.

    Args:
        depth: image tensor containing a depth value per pixel.
        camera_matrix: tensor containing the camera intrinsics.
        normalize_points: whether to normalise the pointcloud. This must be set to `True` when the depth is
          represented as the Euclidean ray length from the camera position.

    Shape:
        - Input: :math:`(B, 1, H, W)` and :math:`(B, 3, 3)`
        - Output: :math:`(B, 3, H, W)`

    Return:
        tensor with a 3d point per pixel of the same resolution as the input.
    """
    if not isinstance(depth, torch.Tensor):
        raise TypeError(f"Input depht type is not a torch.Tensor. Got {type(depth)}.")

    if not len(depth.shape) == 4 and depth.shape[-3] == 1:
        raise ValueError(f"Input depth musth have a shape (B, 1, H, W). Got: {depth.shape}")

    if not isinstance(camera_matrix, torch.Tensor):
        raise TypeError(f"Input camera_matrix type is not a torch.Tensor. " f"Got {type(camera_matrix)}.")

    if not len(camera_matrix.shape) == 3 and camera_matrix.shape[-2:] == (3, 3):
        raise ValueError(f"Input camera_matrix must have a shape (B, 3, 3). " f"Got: {camera_matrix.shape}.")

    # create base coordinates grid
    batch_size, _, height, width = depth.shape
    points_2d: torch.Tensor = create_meshgrid(height, width, normalized_coordinates=False)  # 1xHxWx2
    points_2d = points_2d.to(depth.device).to(depth.dtype)

    # depth should come in Bx1xHxW
    points_depth: torch.Tensor = depth.permute(0, 2, 3, 1)  # 1xHxWx1

    # project pixels to camera frame
    camera_matrix_tmp: torch.Tensor = camera_matrix[:, None, None]  # Bx1x1x3x3
    points_3d: torch.Tensor = unproject_points(
        points_2d, points_depth, camera_matrix_tmp, normalize=normalize_points
    )  # BxHxWx3

    return points_3d.permute(0, 3, 1, 2)  # Bx3xHxW


def depth_to_normals(depth: torch.Tensor, camera_matrix: torch.Tensor, normalize_points: bool = False) -> torch.Tensor:
    """Compute the normal surface per pixel.

    Args:
        depth: image tensor containing a depth value per pixel.
        camera_matrix: tensor containing the camera intrinsics.
        normalize_points: whether to normalise the pointcloud. This must be set to `True` when the depth is
        represented as the Euclidean ray length from the camera position.

    Shape:
        - Input: :math:`(B, 1, H, W)` and :math:`(B, 3, 3)`
        - Output: :math:`(B, 3, H, W)`

    Return:
        tensor with a normal surface vector per pixel of the same resolution as the input.
    """
    if not isinstance(depth, torch.Tensor):
        raise TypeError(f"Input depht type is not a torch.Tensor. Got {type(depth)}.")

    if not len(depth.shape) == 4 and depth.shape[-3] == 1:
        raise ValueError(f"Input depth musth have a shape (B, 1, H, W). Got: {depth.shape}")

    if not isinstance(camera_matrix, torch.Tensor):
        raise TypeError(f"Input camera_matrix type is not a torch.Tensor. " f"Got {type(camera_matrix)}.")

    if not len(camera_matrix.shape) == 3 and camera_matrix.shape[-2:] == (3, 3):
        raise ValueError(f"Input camera_matrix must have a shape (B, 3, 3). " f"Got: {camera_matrix.shape}.")

    # compute the 3d points from depth
    xyz: torch.Tensor = depth_to_3d(depth, camera_matrix, normalize_points)  # Bx3xHxW

    # compute the pointcloud spatial gradients
    gradients: torch.Tensor = spatial_gradient(xyz)  # Bx3x2xHxW

    # compute normals
    a, b = gradients[:, :, 0], gradients[:, :, 1]  # Bx3xHxW

    normals: torch.Tensor = torch.cross(a, b, dim=1)  # Bx3xHxW
    return F.normalize(normals, dim=1, p=2)


def warp_frame_depth(
    image_src: torch.Tensor,
    depth_dst: torch.Tensor,
    src_trans_dst: torch.Tensor,
    camera_matrix: torch.Tensor,
    normalize_points: bool = False,
) -> torch.Tensor:
    """Warp a tensor from a source to destination frame by the depth in the destination.

    Compute 3d points from the depth, transform them using given transformation, then project the point cloud to an
    image plane.

    Args:
        image_src: image tensor in the source frame with shape :math:`(B,D,H,W)`.
        depth_dst: depth tensor in the destination frame with shape :math:`(B,1,H,W)`.
        src_trans_dst: transformation matrix from destination to source with shape :math:`(B,4,4)`.
        camera_matrix: tensor containing the camera intrinsics with shape :math:`(B,3,3)`.
        normalize_points: whether to normalise the pointcloud. This must be set to ``True`` when the depth
           is represented as the Euclidean ray length from the camera position.

    Return:
        the warped tensor in the source frame with shape :math:`(B,3,H,W)`.
    """
    if not isinstance(image_src, torch.Tensor):
        raise TypeError(f"Input image_src type is not a torch.Tensor. Got {type(image_src)}.")

    if not len(image_src.shape) == 4:
        raise ValueError(f"Input image_src musth have a shape (B, D, H, W). Got: {image_src.shape}")

    if not isinstance(depth_dst, torch.Tensor):
        raise TypeError(f"Input depht_dst type is not a torch.Tensor. Got {type(depth_dst)}.")

    if not len(depth_dst.shape) == 4 and depth_dst.shape[-3] == 1:
        raise ValueError(f"Input depth_dst musth have a shape (B, 1, H, W). Got: {depth_dst.shape}")

    if not isinstance(src_trans_dst, torch.Tensor):
        raise TypeError(f"Input src_trans_dst type is not a torch.Tensor. " f"Got {type(src_trans_dst)}.")

    if not len(src_trans_dst.shape) == 3 and src_trans_dst.shape[-2:] == (3, 3):
        raise ValueError(f"Input src_trans_dst must have a shape (B, 3, 3). " f"Got: {src_trans_dst.shape}.")

    if not isinstance(camera_matrix, torch.Tensor):
        raise TypeError(f"Input camera_matrix type is not a torch.Tensor. " f"Got {type(camera_matrix)}.")

    if not len(camera_matrix.shape) == 3 and camera_matrix.shape[-2:] == (3, 3):
        raise ValueError(f"Input camera_matrix must have a shape (B, 3, 3). " f"Got: {camera_matrix.shape}.")
    # unproject source points to camera frame
    points_3d_dst: torch.Tensor = depth_to_3d(depth_dst, camera_matrix, normalize_points)  # Bx3xHxW

    # transform points from source to destionation
    points_3d_dst = points_3d_dst.permute(0, 2, 3, 1)  # BxHxWx3

    # apply transformation to the 3d points
    points_3d_src = transform_points(src_trans_dst[:, None], points_3d_dst)  # BxHxWx3

    # project back to pixels
    camera_matrix_tmp: torch.Tensor = camera_matrix[:, None, None]  # Bx1x1xHxW
    points_2d_src: torch.Tensor = project_points(points_3d_src, camera_matrix_tmp)  # BxHxWx2

    # normalize points between [-1 / 1]
    height, width = depth_dst.shape[-2:]
    points_2d_src_norm: torch.Tensor = normalize_pixel_coordinates(points_2d_src, height, width)  # BxHxWx2

    return F.grid_sample(image_src, points_2d_src_norm, align_corners=True)  # type: ignore


class DepthWarper(nn.Module):
    r"""Warps a patch by depth.

    .. math::
        P_{src}^{\{dst\}} = K_{dst} * T_{src}^{\{dst\}}

        I_{src} = \\omega(I_{dst}, P_{src}^{\{dst\}}, D_{src})

    Args:
        pinholes_dst: the pinhole models for the destination frame.
        height: the height of the image to warp.
        width: the width of the image to warp.
        mode: interpolation mode to calculate output values ``'bilinear'`` | ``'nearest'``.
        padding_mode: padding mode for outside grid values ``'zeros'`` | ``'border'`` | ``'reflection'``.
        align_corners: interpolation flag.
    """

    def __init__(
        self,
        pinhole_dst: PinholeCamera,
        height: int,
        width: int,
        mode: str = 'bilinear',
        padding_mode: str = 'zeros',
        align_corners: bool = True,
    ):
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
        grid: torch.Tensor = create_meshgrid(height, width, normalized_coordinates=False)  # 1xHxWx2
        return convert_points_to_homogeneous(grid)  # append ones to last dim

    def compute_projection_matrix(self, pinhole_src: PinholeCamera) -> 'DepthWarper':
        r"""Computes the projection matrix from the source to destination frame."""
        if not isinstance(self._pinhole_dst, PinholeCamera):
            raise TypeError(
                "Member self._pinhole_dst expected to be of class "
                "PinholeCamera. Got {}".format(type(self._pinhole_dst))
            )
        if not isinstance(pinhole_src, PinholeCamera):
            raise TypeError(
                "Argument pinhole_src expected to be of class " "PinholeCamera. Got {}".format(type(pinhole_src))
            )
        # compute the relative pose between the non reference and the reference
        # camera frames.
        dst_trans_src: torch.Tensor = compose_transformations(
            self._pinhole_dst.extrinsics, inverse_transformation(pinhole_src.extrinsics)
        )

        # compute the projection matrix between the non reference cameras and
        # the reference.
        dst_proj_src: torch.Tensor = torch.matmul(self._pinhole_dst.intrinsics, dst_trans_src)

        # update class members
        self._pinhole_src = pinhole_src
        self._dst_proj_src = dst_proj_src
        return self

    def _compute_projection(self, x, y, invd):
        point = torch.tensor(
            [[[x], [y], [1.0], [invd]]], device=self._dst_proj_src.device, dtype=self._dst_proj_src.dtype
        )
        flow = torch.matmul(self._dst_proj_src, point)
        z = 1.0 / flow[:, 2]
        x = flow[:, 0] * z
        y = flow[:, 1] * z
        return torch.cat([x, y], 1)

    def compute_subpixel_step(self) -> torch.Tensor:
        """This computes the required inverse depth step to achieve sub pixel
        accurate sampling of the depth cost volume, per camera.

        Szeliski, Richard, and Daniel Scharstein.
        "Symmetric sub-pixel stereo matching." European Conference on Computer
        Vision. Springer Berlin Heidelberg, 2002.
        """
        delta_d = 0.01
        xy_m1 = self._compute_projection(self.width / 2, self.height / 2, 1.0 - delta_d)
        xy_p1 = self._compute_projection(self.width / 2, self.height / 2, 1.0 + delta_d)
        dx = torch.norm((xy_p1 - xy_m1), 2, dim=-1) / 2.0
        dxdd = dx / (delta_d)  # pixel*(1/meter)
        # half pixel sampling, we're interested in the min for all cameras
        return torch.min(0.5 / dxdd)

    def warp_grid(self, depth_src: torch.Tensor) -> torch.Tensor:
        """Computes a grid for warping a given the depth from the reference pinhole camera.

        The function `compute_projection_matrix` has to be called beforehand in
        order to have precomputed the relative projection matrices encoding the
        relative pose and the intrinsics between the reference and a non
        reference camera.
        """
        # TODO: add type and value checkings
        if self._dst_proj_src is None or self._pinhole_src is None:
            raise ValueError("Please, call compute_projection_matrix.")

        if len(depth_src.shape) != 4:
            raise ValueError("Input depth_src has to be in the shape of " "Bx1xHxW. Got {}".format(depth_src.shape))

        # unpack depth attributes
        batch_size, _, height, width = depth_src.shape
        device: torch.device = depth_src.device
        dtype: torch.dtype = depth_src.dtype

        # expand the base coordinate grid according to the input batch size
        pixel_coords: torch.Tensor = self.grid.to(device=device, dtype=dtype).expand(batch_size, -1, -1, -1)  # BxHxWx3

        # reproject the pixel coordinates to the camera frame
        cam_coords_src: torch.Tensor = pixel2cam(
            depth_src, self._pinhole_src.intrinsics_inverse().to(device=device, dtype=dtype), pixel_coords
        )  # BxHxWx3

        # reproject the camera coordinates to the pixel
        pixel_coords_src: torch.Tensor = cam2pixel(
            cam_coords_src, self._dst_proj_src.to(device=device, dtype=dtype)
        )  # (B*N)xHxWx2

        # normalize between -1 and 1 the coordinates
        pixel_coords_src_norm: torch.Tensor = normalize_pixel_coordinates(pixel_coords_src, self.height, self.width)
        return pixel_coords_src_norm

    def forward(self, depth_src: torch.Tensor, patch_dst: torch.Tensor) -> torch.Tensor:
        """Warps a tensor from destination frame to reference given the depth in the reference frame.

        Args:
            depth_src: the depth in the reference frame. The tensor must have a shape :math:`(B, 1, H, W)`.
            patch_dst: the patch in the destination frame. The tensor must have a shape :math:`(B, C, H, W)`.

        Return:
            the warped patch from destination frame to reference.

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
        return F.grid_sample(
            patch_dst,
            self.warp_grid(depth_src),  # type: ignore
            mode=self.mode,
            padding_mode=self.padding_mode,
            align_corners=self.align_corners,
        )


def depth_warp(
    pinhole_dst: PinholeCamera,
    pinhole_src: PinholeCamera,
    depth_src: torch.Tensor,
    patch_dst: torch.Tensor,
    height: int,
    width: int,
    align_corners: bool = True,
):
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
