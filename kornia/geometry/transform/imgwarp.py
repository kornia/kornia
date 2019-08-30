from typing import Tuple, Optional

import torch
import torch.nn.functional as F

import kornia
from kornia.geometry.warp.homography_warper import homography_warp
# TODO: move to utils or conversions
from kornia.geometry.conversions import (
    deg2rad, normalize_pixel_coordinates
)

__all__ = [
    "warp_perspective",
    "warp_affine",
    "get_perspective_transform",
    "get_rotation_matrix2d",
    "normal_transform_pixel",
    "remap",
    "invert_affine_transform",
    "angle_to_rotation_matrix"
]


def normal_transform_pixel(height, width):

    tr_mat = torch.Tensor([[1.0, 0.0, -1.0],
                           [0.0, 1.0, -1.0],
                           [0.0, 0.0, 1.0]])  # 1x3x3

    tr_mat[0, 0] = tr_mat[0, 0] * 2.0 / (width - 1.0)
    tr_mat[1, 1] = tr_mat[1, 1] * 2.0 / (height - 1.0)

    tr_mat = tr_mat.unsqueeze(0)

    return tr_mat


def dst_norm_to_dst_norm(dst_pix_trans_src_pix, dsize_src, dsize_dst):
    # source and destination sizes
    src_h, src_w = dsize_src
    dst_h, dst_w = dsize_dst
    # the devices and types
    device = dst_pix_trans_src_pix.device
    dtype = dst_pix_trans_src_pix.dtype
    # compute the transformation pixel/norm for src/dst
    src_norm_trans_src_pix = normal_transform_pixel(
        src_h, src_w).to(device).to(dtype)
    src_pix_trans_src_norm = torch.inverse(src_norm_trans_src_pix)
    dst_norm_trans_dst_pix = normal_transform_pixel(
        dst_h, dst_w).to(device).to(dtype)
    # compute chain transformations
    dst_norm_trans_src_norm = torch.matmul(
        dst_norm_trans_dst_pix, torch.matmul(
            dst_pix_trans_src_pix, src_pix_trans_src_norm))
    return dst_norm_trans_src_norm


def transform_warp_impl(src, dst_pix_trans_src_pix, dsize_src, dsize_dst):
    """Compute the transform in normalized cooridnates and perform the warping.
    """
    dst_norm_trans_dst_norm = dst_norm_to_dst_norm(
        dst_pix_trans_src_pix, dsize_src, dsize_dst)
    return homography_warp(src, torch.inverse(
        dst_norm_trans_dst_norm), dsize_dst)


def warp_perspective(src, M, dsize, flags='bilinear', border_mode=None,
                     border_value=0):
    r"""Applies a perspective transformation to an image.

    The function warp_perspective transforms the source image using
    the specified matrix:

    .. math::
        \text{dst} (x, y) = \text{src} \left(
        \frac{M_{11} x + M_{12} y + M_{13}}{M_{31} x + M_{32} y + M_{33}} ,
        \frac{M_{21} x + M_{22} y + M_{23}}{M_{31} x + M_{32} y + M_{33}}
        \right )

    Args:
        src (torch.Tensor): input image.
        M (Tensor): transformation matrix.
        dsize (tuple): size of the output image (height, width).

    Returns:
        Tensor: the warped input image.

    Shape:
        - Input: :math:`(B, C, H, W)` and :math:`(B, 3, 3)`
        - Output: :math:`(B, C, H, W)`

    .. note::
       See a working example `here <https://github.com/arraiyopensource/kornia/
       blob/master/docs/source/warp_perspective.ipynb>`_.
    """
    if not torch.is_tensor(src):
        raise TypeError("Input src type is not a torch.Tensor. Got {}"
                        .format(type(src)))
    if not torch.is_tensor(M):
        raise TypeError("Input M type is not a torch.Tensor. Got {}"
                        .format(type(M)))
    if not len(src.shape) == 4:
        raise ValueError("Input src must be a BxCxHxW tensor. Got {}"
                         .format(src.shape))
    if not (len(M.shape) == 3 or M.shape[-2:] == (3, 3)):
        raise ValueError("Input M must be a Bx3x3 tensor. Got {}"
                         .format(src.shape))
    # launches the warper
    return transform_warp_impl(src, M, (src.shape[-2:]), dsize)


def warp_affine(src: torch.Tensor,
                M: torch.Tensor,
                dsize: Tuple[int,
                             int],
                flags: Optional[str] = 'bilinear',
                padding_mode: Optional[str] = 'zeros') -> torch.Tensor:
    r"""Applies an affine transformation to a tensor.

    The function warp_affine transforms the source tensor using
    the specified matrix:

    .. math::
        \text{dst}(x, y) = \text{src} \left( M_{11} x + M_{12} y + M_{13} ,
        M_{21} x + M_{22} y + M_{23} \right )

    Args:
        src (torch.Tensor): input tensor of shape :math:`(B, C, H, W)`.
        M (torch.Tensor): affine transformation of shape :math:`(B, 2, 3)`.
        dsize (Tuple[int, int]): size of the output image (height, width).
        mode (Optional[str]): interpolation mode to calculate output values
          'bilinear' | 'nearest'. Default: 'bilinear'.
        padding_mode (Optional[str]): padding mode for outside grid values
          'zeros' | 'border' | 'reflection'. Default: 'zeros'.

    Returns:
        torch.Tensor: the warped tensor.

    Shape:
        - Output: :math:`(B, C, H, W)`

    .. note::
       See a working example `here <https://github.com/arraiyopensource/
       kornia/blob/master/docs/source/warp_affine.ipynb>`__.
    """
    if not torch.is_tensor(src):
        raise TypeError("Input src type is not a torch.Tensor. Got {}"
                        .format(type(src)))
    if not torch.is_tensor(M):
        raise TypeError("Input M type is not a torch.Tensor. Got {}"
                        .format(type(M)))
    if not len(src.shape) == 4:
        raise ValueError("Input src must be a BxCxHxW tensor. Got {}"
                         .format(src.shape))
    if not (len(M.shape) == 3 or M.shape[-2:] == (2, 3)):
        raise ValueError("Input M must be a Bx2x3 tensor. Got {}"
                         .format(src.shape))
    # we generate a 3x3 transformation matrix from 2x3 affine
    M_3x3: torch.Tensor = F.pad(M, [0, 0, 0, 1, 0, 0],
                                mode="constant", value=0)
    M_3x3[:, 2, 2] += 1.0

    # launches the warper
    return transform_warp_impl(src, M_3x3, (src.shape[-2:]), dsize)


def get_perspective_transform(src, dst):
    r"""Calculates a perspective transform from four pairs of the corresponding
    points.

    The function calculates the matrix of a perspective transform so that:

    .. math ::

        \begin{bmatrix}
        t_{i}x_{i}^{'} \\
        t_{i}y_{i}^{'} \\
        t_{i} \\
        \end{bmatrix}
        =
        \textbf{map_matrix} \cdot
        \begin{bmatrix}
        x_{i} \\
        y_{i} \\
        1 \\
        \end{bmatrix}

    where

    .. math ::
        dst(i) = (x_{i}^{'},y_{i}^{'}), src(i) = (x_{i}, y_{i}), i = 0,1,2,3

    Args:
        src (Tensor): coordinates of quadrangle vertices in the source image.
        dst (Tensor): coordinates of the corresponding quadrangle vertices in
            the destination image.

    Returns:
        Tensor: the perspective transformation.

    Shape:
        - Input: :math:`(B, 4, 2)` and :math:`(B, 4, 2)`
        - Output: :math:`(B, 3, 3)`
    """
    if not torch.is_tensor(src):
        raise TypeError("Input type is not a torch.Tensor. Got {}"
                        .format(type(src)))
    if not torch.is_tensor(dst):
        raise TypeError("Input type is not a torch.Tensor. Got {}"
                        .format(type(dst)))
    if not src.shape[-2:] == (4, 2):
        raise ValueError("Inputs must be a Bx4x2 tensor. Got {}"
                         .format(src.shape))
    if not src.shape == dst.shape:
        raise ValueError("Inputs must have the same shape. Got {}"
                         .format(dst.shape))
    if not (src.shape[0] == dst.shape[0]):
        raise ValueError("Inputs must have same batch size dimension. Got {}"
                         .format(src.shape, dst.shape))

    def ax(p, q):
        ones = torch.ones_like(p)[..., 0:1]
        zeros = torch.zeros_like(p)[..., 0:1]
        return torch.cat(
            [p[:, 0:1], p[:, 1:2], ones, zeros, zeros, zeros,
             -p[:, 0:1] * q[:, 0:1], -p[:, 1:2] * q[:, 0:1]
             ], dim=1)

    def ay(p, q):
        ones = torch.ones_like(p)[..., 0:1]
        zeros = torch.zeros_like(p)[..., 0:1]
        return torch.cat(
            [zeros, zeros, zeros, p[:, 0:1], p[:, 1:2], ones,
             -p[:, 0:1] * q[:, 1:2], -p[:, 1:2] * q[:, 1:2]], dim=1)
    # we build matrix A by using only 4 point correspondence. The linear
    # system is solved with the least square method, so here
    # we could even pass more correspondence
    p = []
    p.append(ax(src[:, 0], dst[:, 0]))
    p.append(ay(src[:, 0], dst[:, 0]))

    p.append(ax(src[:, 1], dst[:, 1]))
    p.append(ay(src[:, 1], dst[:, 1]))

    p.append(ax(src[:, 2], dst[:, 2]))
    p.append(ay(src[:, 2], dst[:, 2]))

    p.append(ax(src[:, 3], dst[:, 3]))
    p.append(ay(src[:, 3], dst[:, 3]))

    # A is Bx8x8
    A = torch.stack(p, dim=1)

    # b is a Bx8x1
    b = torch.stack([
        dst[:, 0:1, 0], dst[:, 0:1, 1],
        dst[:, 1:2, 0], dst[:, 1:2, 1],
        dst[:, 2:3, 0], dst[:, 2:3, 1],
        dst[:, 3:4, 0], dst[:, 3:4, 1],
    ], dim=1)

    # solve the system Ax = b
    X, LU = torch.solve(b, A)

    # create variable to return
    batch_size = src.shape[0]
    M = torch.ones(batch_size, 9, device=src.device, dtype=src.dtype)
    M[..., :8] = torch.squeeze(X, dim=-1)
    return M.view(-1, 3, 3)  # Bx3x3


def angle_to_rotation_matrix(angle: torch.Tensor) -> torch.Tensor:
    """
    Creates a rotation matrix out of angles in degrees
    Args:
        angle: (torch.Tensor): tensor of angles in degrees, any shape.

    Returns:
        torch.Tensor: tensor of *x2x2 rotation matrices.

    Shape:
        - Input: :math:`(*)`
        - Output: :math:`(*, 2, 2)`

    Example:
        >>> input = torch.rand(1, 3)  # Nx3
        >>> output = kornia.angle_to_rotation_matrix(input)  # Nx3x2x2
    """
    ang_rad = kornia.deg2rad(angle)
    cos_a: torch.Tensor = torch.cos(ang_rad)
    sin_a: torch.Tensor = torch.sin(ang_rad)
    return torch.stack([cos_a, sin_a, -sin_a, cos_a], dim=-1).view(*angle.shape, 2, 2)


def get_rotation_matrix2d(
        center: torch.Tensor,
        angle: torch.Tensor,
        scale: torch.Tensor) -> torch.Tensor:
    r"""Calculates an affine matrix of 2D rotation.

    The function calculates the following matrix:

    .. math::
        \begin{bmatrix}
            \alpha & \beta & (1 - \alpha) \cdot \text{x}
            - \beta \cdot \text{y} \\
            -\beta & \alpha & \beta \cdot \text{x}
            + (1 - \alpha) \cdot \text{y}
        \end{bmatrix}

    where

    .. math::
        \alpha = \text{scale} \cdot cos(\text{angle}) \\
        \beta = \text{scale} \cdot sin(\text{angle})

    The transformation maps the rotation center to itself
    If this is not the target, adjust the shift.

    Args:
        center (Tensor): center of the rotation in the source image.
        angle (Tensor): rotation angle in degrees. Positive values mean
            counter-clockwise rotation (the coordinate origin is assumed to
            be the top-left corner).
        scale (Tensor): isotropic scale factor.

    Returns:
        Tensor: the affine matrix of 2D rotation.

    Shape:
        - Input: :math:`(B, 2)`, :math:`(B)` and :math:`(B)`
        - Output: :math:`(B, 2, 3)`

    Example:
        >>> center = torch.zeros(1, 2)
        >>> scale = torch.ones(1)
        >>> angle = 45. * torch.ones(1)
        >>> M = kornia.get_rotation_matrix2d(center, angle, scale)
        tensor([[[ 0.7071,  0.7071,  0.0000],
                 [-0.7071,  0.7071,  0.0000]]])
    """
    if not torch.is_tensor(center):
        raise TypeError("Input center type is not a torch.Tensor. Got {}"
                        .format(type(center)))
    if not torch.is_tensor(angle):
        raise TypeError("Input angle type is not a torch.Tensor. Got {}"
                        .format(type(angle)))
    if not torch.is_tensor(scale):
        raise TypeError("Input scale type is not a torch.Tensor. Got {}"
                        .format(type(scale)))
    if not (len(center.shape) == 2 and center.shape[1] == 2):
        raise ValueError("Input center must be a Bx2 tensor. Got {}"
                         .format(center.shape))
    if not len(angle.shape) == 1:
        raise ValueError("Input angle must be a B tensor. Got {}"
                         .format(angle.shape))
    if not len(scale.shape) == 1:
        raise ValueError("Input scale must be a B tensor. Got {}"
                         .format(scale.shape))
    if not (center.shape[0] == angle.shape[0] == scale.shape[0]):
        raise ValueError("Inputs must have same batch size dimension. Got {}"
                         .format(center.shape, angle.shape, scale.shape))
    # convert angle and apply scale
    scaled_rotation: torch.Tensor = angle_to_rotation_matrix(angle) * scale.view(-1, 1, 1)
    alpha: torch.Tensor = scaled_rotation[:, 0, 0]
    beta: torch.Tensor = scaled_rotation[:, 0, 1]

    # unpack the center to x, y coordinates
    x: torch.Tensor = center[..., 0]
    y: torch.Tensor = center[..., 1]

    # create output tensor
    batch_size: int = center.shape[0]
    M: torch.Tensor = torch.zeros(
        batch_size, 2, 3, device=center.device, dtype=center.dtype)
    M[..., 0:2, 0:2] = scaled_rotation
    M[..., 0, 2] = (torch.tensor(1.) - alpha) * x - beta * y
    M[..., 1, 2] = beta * x + (torch.tensor(1.) - alpha) * y
    return M


def remap(tensor: torch.Tensor, map_x: torch.Tensor,
          map_y: torch.Tensor) -> torch.Tensor:
    r"""Applies a generic geometrical transformation to a tensor.

    The function remap transforms the source tensor using the specified map:

    .. math::
        \text{dst}(x, y) = \text{src}(map_x(x, y), map_y(x, y))

    Args:
        tensor (torch.Tensor): the tensor to remap with shape (B, D, H, W).
          Where D is the number of channels.
        map_x (torch.Tensor): the flow in the x-direction in pixel coordinates.
          The tensor must be in the shape of (B, H, W).
        map_y (torch.Tensor): the flow in the y-direction in pixel coordinates.
          The tensor must be in the shape of (B, H, W).

    Returns:
        torch.Tensor: the warped tensor.

    Example:
        >>> grid = kornia.utils.create_meshgrid(2, 2, False)  # 1x2x2x2
        >>> grid += 1  # apply offset in both directions
        >>> input = torch.ones(1, 1, 2, 2)
        >>> kornia.remap(input, grid[..., 0], grid[..., 1])   # 1x1x2x2
        tensor([[[[1., 0.],
                  [0., 0.]]]])

    """
    if not torch.is_tensor(tensor):
        raise TypeError("Input tensor type is not a torch.Tensor. Got {}"
                        .format(type(tensor)))
    if not torch.is_tensor(map_x):
        raise TypeError("Input map_x type is not a torch.Tensor. Got {}"
                        .format(type(map_x)))
    if not torch.is_tensor(map_y):
        raise TypeError("Input map_y type is not a torch.Tensor. Got {}"
                        .format(type(map_y)))
    if not tensor.shape[-2:] == map_x.shape[-2:] == map_y.shape[-2:]:
        raise ValueError("Inputs last two dimensions must match.")

    batch_size, _, height, width = tensor.shape

    # grid_sample need the grid between -1/1
    map_xy: torch.Tensor = torch.stack([map_x, map_y], dim=-1)
    map_xy_norm: torch.Tensor = normalize_pixel_coordinates(
        map_xy, height, width)

    # simulate broadcasting since grid_sample does not support it
    map_xy_norm = map_xy_norm.expand(batch_size, -1, -1, -1)

    # warp ans return
    tensor_warped: torch.Tensor = F.grid_sample(tensor, map_xy_norm)
    return tensor_warped


def invert_affine_transform(matrix: torch.Tensor) -> torch.Tensor:
    r"""Inverts an affine transformation.

    The function computes an inverse affine transformation represented by
    2×3 matrix:

    .. math::
        \begin{bmatrix}
            a_{11} & a_{12} & b_{1} \\
            a_{21} & a_{22} & b_{2} \\
        \end{bmatrix}

    The result is also a 2×3 matrix of the same type as M.

    Args:
        matrix (torch.Tensor): original affine transform. The tensor musth be
          in the shape of (B, 2, 3).

    Return:
        torch.Tensor: the reverse affine transform.
    """
    if not torch.is_tensor(matrix):
        raise TypeError("Input matrix type is not a torch.Tensor. Got {}"
                        .format(type(matrix)))
    if not (len(matrix.shape) == 3 or matrix.shape[-2:] == (2, 3)):
        raise ValueError("Input matrix must be a Bx2x3 tensor. Got {}"
                         .format(matrix.shape))
    matrix_tmp: torch.Tensor = F.pad(matrix, [0, 0, 0, 1], "constant", 0.0)
    matrix_tmp[..., 2, 2] += 1.0

    matrix_inv: torch.Tensor = torch.inverse(matrix_tmp)
    return matrix_inv[..., :2, :3]
