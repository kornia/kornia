from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, stack
from torch.nn.functional import grid_sample

from kornia.geometry.conversions import (
    angle_axis_to_rotation_matrix,
    angle_to_rotation_matrix,
    convert_affinematrix_to_homography,
    convert_affinematrix_to_homography3d,
    deg2rad,
    normalize_homography,
    normalize_homography3d,
    normalize_pixel_coordinates,
)
from kornia.geometry.linalg import transform_points
from kornia.testing import KORNIA_CHECK_IS_TENSOR
from kornia.utils import create_meshgrid, create_meshgrid3d, eye_like
from kornia.utils.helpers import _torch_inverse_cast, _torch_solve_cast

__all__ = [
    "warp_perspective",
    "warp_affine",
    "get_perspective_transform",
    "get_rotation_matrix2d",
    "remap",
    "invert_affine_transform",
    "get_affine_matrix2d",
    "get_affine_matrix3d",
    "get_shear_matrix2d",
    "get_shear_matrix3d",
    "warp_affine3d",
    "get_projective_transform",
    "projection_from_Rt",
    "get_perspective_transform3d",
    "warp_perspective3d",
    "warp_grid",
    "warp_grid3d",
    "homography_warp",
    "homography_warp3d",
]


def warp_perspective(
    src: torch.Tensor,
    M: torch.Tensor,
    dsize: Tuple[int, int],
    mode: str = 'bilinear',
    padding_mode: str = 'zeros',
    align_corners: bool = True,
    fill_value: torch.Tensor = torch.zeros(3),  # needed for jit
) -> torch.Tensor:
    r"""Apply a perspective transformation to an image.

    .. image:: https://kornia-tutorials.readthedocs.io/en/latest/_images/warp_perspective_10_1.png

    The function warp_perspective transforms the source image using
    the specified matrix:

    .. math::
        \text{dst} (x, y) = \text{src} \left(
        \frac{M^{-1}_{11} x + M^{-1}_{12} y + M^{-1}_{13}}{M^{-1}_{31} x + M^{-1}_{32} y + M^{-1}_{33}} ,
        \frac{M^{-1}_{21} x + M^{-1}_{22} y + M^{-1}_{23}}{M^{-1}_{31} x + M^{-1}_{32} y + M^{-1}_{33}}
        \right )

    Args:
        src: input image with shape :math:`(B, C, H, W)`.
        M: transformation matrix with shape :math:`(B, 3, 3)`.
        dsize: size of the output image (height, width).
        mode: interpolation mode to calculate output values ``'bilinear'`` | ``'nearest'``.
        padding_mode: padding mode for outside grid values ``'zeros'`` | ``'border'`` | ``'reflection'`` | ``'fill'``.
        align_corners: interpolation flag.
        fill_value: tensor of shape :math:`(3)` that fills the padding area. Only supported for RGB.

    Returns:
        the warped input image :math:`(B, C, H, W)`.

    Example:
       >>> img = torch.rand(1, 4, 5, 6)
       >>> H = torch.eye(3)[None]
       >>> out = warp_perspective(img, H, (4, 2), align_corners=True)
       >>> print(out.shape)
       torch.Size([1, 4, 4, 2])

    .. note::
        This function is often used in conjunction with :func:`get_perspective_transform`.

    .. note::
        See a working example `here <https://kornia-tutorials.readthedocs.io/en/
        latest/warp_perspective.html>`_.
    """
    if not isinstance(src, torch.Tensor):
        raise TypeError(f"Input src type is not a torch.Tensor. Got {type(src)}")

    if not isinstance(M, torch.Tensor):
        raise TypeError(f"Input M type is not a torch.Tensor. Got {type(M)}")

    if not len(src.shape) == 4:
        raise ValueError(f"Input src must be a BxCxHxW tensor. Got {src.shape}")

    if not (len(M.shape) == 3 and M.shape[-2:] == (3, 3)):
        raise ValueError(f"Input M must be a Bx3x3 tensor. Got {M.shape}")

    # fill padding is only supported for 3 channels because we can't set fill_value default
    # to None as this gives jit issues.
    if padding_mode == "fill" and fill_value.shape != torch.Size([3]):
        raise ValueError(f"Padding_tensor only supported for 3 channels. Got {fill_value.shape}")

    B, _, H, W = src.size()
    h_out, w_out = dsize

    # we normalize the 3x3 transformation matrix and convert to 3x4
    dst_norm_trans_src_norm: torch.Tensor = normalize_homography(M, (H, W), (h_out, w_out))  # Bx3x3

    src_norm_trans_dst_norm = _torch_inverse_cast(dst_norm_trans_src_norm)  # Bx3x3

    # this piece of code substitutes F.affine_grid since it does not support 3x3
    grid = (
        create_meshgrid(h_out, w_out, normalized_coordinates=True, device=src.device).to(src.dtype).repeat(B, 1, 1, 1)
    )
    grid = transform_points(src_norm_trans_dst_norm[:, None, None], grid)

    if padding_mode == "fill":
        return _fill_and_warp(src, grid, align_corners=align_corners, mode=mode, fill_value=fill_value)
    return F.grid_sample(src, grid, align_corners=align_corners, mode=mode, padding_mode=padding_mode)


def warp_affine(
    src: torch.Tensor,
    M: torch.Tensor,
    dsize: Tuple[int, int],
    mode: str = 'bilinear',
    padding_mode: str = 'zeros',
    align_corners: bool = True,
    fill_value: torch.Tensor = torch.zeros(3),  # needed for jit
) -> torch.Tensor:
    r"""Apply an affine transformation to a tensor.

    .. image:: _static/img/warp_affine.png

    The function warp_affine transforms the source tensor using
    the specified matrix:

    .. math::
        \text{dst}(x, y) = \text{src} \left( M_{11} x + M_{12} y + M_{13} ,
        M_{21} x + M_{22} y + M_{23} \right )

    Args:
        src: input tensor of shape :math:`(B, C, H, W)`.
        M: affine transformation of shape :math:`(B, 2, 3)`.
        dsize: size of the output image (height, width).
        mode: interpolation mode to calculate output values ``'bilinear'`` | ``'nearest'``.
        padding_mode: padding mode for outside grid values ``'zeros'`` | ``'border'`` | ``'reflection'`` | ``'fill'``.
        align_corners : mode for grid_generation.
        fill_value: tensor of shape :math:`(3)` that fills the padding area. Only supported for RGB.

    Returns:
        the warped tensor with shape :math:`(B, C, H, W)`.

    .. note::
        This function is often used in conjunction with :func:`get_rotation_matrix2d`,
        :func:`get_shear_matrix2d`, :func:`get_affine_matrix2d`, :func:`invert_affine_transform`.

    .. note::
       See a working example `here <https://kornia-tutorials.readthedocs.io/en/latest/
       rotate_affine.html>`__.

    Example:
       >>> img = torch.rand(1, 4, 5, 6)
       >>> A = torch.eye(2, 3)[None]
       >>> out = warp_affine(img, A, (4, 2), align_corners=True)
       >>> print(out.shape)
       torch.Size([1, 4, 4, 2])
    """
    if not isinstance(src, torch.Tensor):
        raise TypeError(f"Input src type is not a torch.Tensor. Got {type(src)}")

    if not isinstance(M, torch.Tensor):
        raise TypeError(f"Input M type is not a torch.Tensor. Got {type(M)}")

    if not len(src.shape) == 4:
        raise ValueError(f"Input src must be a BxCxHxW tensor. Got {src.shape}")

    if not (len(M.shape) == 3 or M.shape[-2:] == (2, 3)):
        raise ValueError(f"Input M must be a Bx2x3 tensor. Got {M.shape}")

    # fill padding is only supported for 3 channels because we can't set fill_value default
    # to None as this gives jit issues.
    if padding_mode == "fill" and fill_value.shape != torch.Size([3]):
        raise ValueError(f"Padding_tensor only supported for 3 channels. Got {fill_value.shape}")

    B, C, H, W = src.size()

    # we generate a 3x3 transformation matrix from 2x3 affine
    M_3x3: torch.Tensor = convert_affinematrix_to_homography(M)
    dst_norm_trans_src_norm: torch.Tensor = normalize_homography(M_3x3, (H, W), dsize)

    # src_norm_trans_dst_norm = torch.inverse(dst_norm_trans_src_norm)
    src_norm_trans_dst_norm = _torch_inverse_cast(dst_norm_trans_src_norm)

    grid = F.affine_grid(src_norm_trans_dst_norm[:, :2, :], [B, C, dsize[0], dsize[1]], align_corners=align_corners)

    if padding_mode == "fill":
        return _fill_and_warp(src, grid, align_corners=align_corners, mode=mode, fill_value=fill_value)
    return F.grid_sample(src, grid, align_corners=align_corners, mode=mode, padding_mode=padding_mode)


def _fill_and_warp(
    src: torch.Tensor, grid: torch.Tensor, mode: str, align_corners: bool, fill_value: torch.Tensor
) -> torch.Tensor:
    r"""Warp a mask of ones, then multiple with fill_value and add to default warp.

    Args:
        src: input tensor of shape :math:`(B, 3, H, W)`.
        grid: grid tensor from `transform_points`.
        mode: interpolation mode to calculate output values ``'bilinear'`` | ``'nearest'``.
        align_corners: interpolation flag.
        fill_value: tensor of shape :math:`(3)` that fills the padding area. Only supported for RGB.

    Returns:
        the warped and filled tensor with shape :math:`(B, 3, H, W)`.
    """
    ones_mask = torch.ones_like(src)
    fill_value = fill_value.to(ones_mask)[None, :, None, None]  # cast and add dimensions for broadcasting
    inv_ones_mask = 1 - F.grid_sample(ones_mask, grid, align_corners=align_corners, mode=mode, padding_mode="zeros")
    inv_color_mask = inv_ones_mask * fill_value
    return F.grid_sample(src, grid, align_corners=align_corners, mode=mode, padding_mode="zeros") + inv_color_mask


def warp_grid(grid: torch.Tensor, src_homo_dst: torch.Tensor) -> torch.Tensor:
    r"""Compute the grid to warp the coordinates grid by the homography/ies.

    Args:
        grid: Unwrapped grid of the shape :math:`(1, N, W, 2)`.
        src_homo_dst: Homography or homographies (stacked) to
          transform all points in the grid. Shape of the homography
          has to be :math:`(1, 3, 3)` or :math:`(N, 1, 3, 3)`.

    Returns:
        the transformed grid of shape :math:`(N, H, W, 2)`.
    """
    batch_size: int = src_homo_dst.size(0)
    _, height, width, _ = grid.size()
    # expand grid to match the input batch size
    grid = grid.expand(batch_size, -1, -1, -1)  # NxHxWx2
    if len(src_homo_dst.shape) == 3:  # local homography case
        src_homo_dst = src_homo_dst.view(batch_size, 1, 3, 3)  # Nx1x3x3
    # perform the actual grid transformation,
    # the grid is copied to input device and casted to the same type
    flow: torch.Tensor = transform_points(src_homo_dst, grid.to(src_homo_dst))  # NxHxWx2
    return flow.view(batch_size, height, width, 2)  # NxHxWx2


def warp_grid3d(grid: torch.Tensor, src_homo_dst: torch.Tensor) -> torch.Tensor:
    r"""Compute the grid to warp the coordinates grid by the homography/ies.

    Args:
        grid: Unwrapped grid of the shape :math:`(1, D, H, W, 3)`.
        src_homo_dst: Homography or homographies (stacked) to
          transform all points in the grid. Shape of the homography
          has to be :math:`(1, 4, 4)` or :math:`(N, 1, 4, 4)`.

    Returns:
        the transformed grid of shape :math:`(N, H, W, 3)`.
    """
    batch_size: int = src_homo_dst.size(0)
    _, depth, height, width, _ = grid.size()
    # expand grid to match the input batch size
    grid = grid.expand(batch_size, -1, -1, -1, -1)  # NxDxHxWx3
    if len(src_homo_dst.shape) == 3:  # local homography case
        src_homo_dst = src_homo_dst.view(batch_size, 1, 4, 4)  # Nx1x3x3
    # perform the actual grid transformation,
    # the grid is copied to input device and casted to the same type
    flow: torch.Tensor = transform_points(src_homo_dst, grid.to(src_homo_dst))  # NxDxHxWx3
    return flow.view(batch_size, depth, height, width, 3)  # NxDxHxWx3


def get_perspective_transform(src, dst):
    r"""Calculate a perspective transform from four pairs of the corresponding
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
        src: coordinates of quadrangle vertices in the source image with shape :math:`(B, 4, 2)`.
        dst: coordinates of the corresponding quadrangle vertices in
            the destination image with shape :math:`(B, 4, 2)`.

    Returns:
        the perspective transformation with shape :math:`(B, 3, 3)`.

    .. note::
        This function is often used in conjunction with :func:`warp_perspective`.
    """
    if not isinstance(src, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(src)}")

    if not isinstance(dst, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(dst)}")

    if not src.dtype == dst.dtype:
        raise TypeError(f"Source data type {src.dtype} must match Destination data type {dst.dtype}")

    if not src.shape[-2:] == (4, 2):
        raise ValueError(f"Inputs must be a Bx4x2 tensor. Got {src.shape}")

    if not src.shape == dst.shape:
        raise ValueError(f"Inputs must have the same shape. Got {dst.shape}")

    if not (src.shape[0] == dst.shape[0]):
        raise ValueError(f"Inputs must have same batch size dimension. Expect {src.shape} but got {dst.shape}")

    # we build matrix A by using only 4 point correspondence. The linear
    # system is solved with the least square method, so here
    # we could even pass more correspondence
    p = []
    for i in [0, 1, 2, 3]:
        p.append(_build_perspective_param(src[:, i], dst[:, i], 'x'))
        p.append(_build_perspective_param(src[:, i], dst[:, i], 'y'))

    # A is Bx8x8
    A = torch.stack(p, dim=1)

    # b is a Bx8x1
    b = torch.stack(
        [
            dst[:, 0:1, 0],
            dst[:, 0:1, 1],
            dst[:, 1:2, 0],
            dst[:, 1:2, 1],
            dst[:, 2:3, 0],
            dst[:, 2:3, 1],
            dst[:, 3:4, 0],
            dst[:, 3:4, 1],
        ],
        dim=1,
    )

    # solve the system Ax = b
    X, _ = _torch_solve_cast(b, A)

    # create variable to return
    batch_size = src.shape[0]
    M = torch.ones(batch_size, 9, device=src.device, dtype=src.dtype)
    M[..., :8] = torch.squeeze(X, dim=-1)

    return M.view(-1, 3, 3)  # Bx3x3


def _build_perspective_param(p: torch.Tensor, q: torch.Tensor, axis: str) -> torch.Tensor:
    ones = torch.ones_like(p)[..., 0:1]
    zeros = torch.zeros_like(p)[..., 0:1]
    if axis == 'x':
        return torch.cat(
            [p[:, 0:1], p[:, 1:2], ones, zeros, zeros, zeros, -p[:, 0:1] * q[:, 0:1], -p[:, 1:2] * q[:, 0:1]], dim=1
        )

    if axis == 'y':
        return torch.cat(
            [zeros, zeros, zeros, p[:, 0:1], p[:, 1:2], ones, -p[:, 0:1] * q[:, 1:2], -p[:, 1:2] * q[:, 1:2]], dim=1
        )

    raise NotImplementedError(f"perspective params for axis `{axis}` is not implemented.")


def get_rotation_matrix2d(center: torch.Tensor, angle: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    r"""Calculate an affine matrix of 2D rotation.

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
        center: center of the rotation in the source image with shape :math:`(B, 2)`.
        angle: rotation angle in degrees. Positive values mean
            counter-clockwise rotation (the coordinate origin is assumed to
            be the top-left corner) with shape :math:`(B)`.
        scale: scale factor for x, y scaling with shape :math:`(B, 2)`.

    Returns:
        the affine matrix of 2D rotation with shape :math:`(B, 2, 3)`.

    Example:
        >>> center = torch.zeros(1, 2)
        >>> scale = torch.ones((1, 2))
        >>> angle = 45. * torch.ones(1)
        >>> get_rotation_matrix2d(center, angle, scale)
        tensor([[[ 0.7071,  0.7071,  0.0000],
                 [-0.7071,  0.7071,  0.0000]]])

    .. note::
        This function is often used in conjunction with :func:`warp_affine`.
    """
    if not isinstance(center, torch.Tensor):
        raise TypeError(f"Input center type is not a torch.Tensor. Got {type(center)}")

    if not isinstance(angle, torch.Tensor):
        raise TypeError(f"Input angle type is not a torch.Tensor. Got {type(angle)}")

    if not isinstance(scale, torch.Tensor):
        raise TypeError(f"Input scale type is not a torch.Tensor. Got {type(scale)}")

    if not (len(center.shape) == 2 and center.shape[1] == 2):
        raise ValueError(f"Input center must be a Bx2 tensor. Got {center.shape}")

    if not len(angle.shape) == 1:
        raise ValueError(f"Input angle must be a B tensor. Got {angle.shape}")

    if not (len(scale.shape) == 2 and scale.shape[1] == 2):
        raise ValueError(f"Input scale must be a Bx2 tensor. Got {scale.shape}")

    if not (center.shape[0] == angle.shape[0] == scale.shape[0]):
        raise ValueError(
            "Inputs must have same batch size dimension. Got center {}, angle {} and scale {}".format(
                center.shape, angle.shape, scale.shape
            )
        )

    if not (center.device == angle.device == scale.device) or not (center.dtype == angle.dtype == scale.dtype):
        raise ValueError(
            "Inputs must have same device Got center ({}, {}), angle ({}, {}) and scale ({}, {})".format(
                center.device, center.dtype, angle.device, angle.dtype, scale.device, scale.dtype
            )
        )

    shift_m = eye_like(3, center)
    shift_m[:, :2, 2] = center

    shift_m_inv = eye_like(3, center)
    shift_m_inv[:, :2, 2] = -center

    scale_m = eye_like(3, center)
    scale_m[:, 0, 0] *= scale[:, 0]
    scale_m[:, 1, 1] *= scale[:, 1]

    rotat_m = eye_like(3, center)
    rotat_m[:, :2, :2] = angle_to_rotation_matrix(angle)

    affine_m = shift_m @ rotat_m @ scale_m @ shift_m_inv
    return affine_m[:, :2, :]  # Bx2x3


def remap(
    image: Tensor,
    map_x: Tensor,
    map_y: Tensor,
    mode: str = 'bilinear',
    padding_mode: str = 'zeros',
    align_corners: Optional[bool] = None,
    normalized_coordinates: bool = False,
) -> Tensor:
    r"""Apply a generic geometrical transformation to an image tensor.

    .. image:: _static/img/remap.png

    The function remap transforms the source tensor using the specified map:

    .. math::
        \text{dst}(x, y) = \text{src}(map_x(x, y), map_y(x, y))

    Args:
        image: the tensor to remap with shape (B, C, H, W).
          Where C is the number of channels.
        map_x: the flow in the x-direction in pixel coordinates.
          The tensor must be in the shape of (B, H, W).
        map_y: the flow in the y-direction in pixel coordinates.
          The tensor must be in the shape of (B, H, W).
        mode: interpolation mode to calculate output values
          ``'bilinear'`` | ``'nearest'``.
        padding_mode: padding mode for outside grid values
          ``'zeros'`` | ``'border'`` | ``'reflection'``.
        align_corners: mode for grid_generation.
        normalized_coordinates: whether the input coordinates are
           normalized in the range of [-1, 1].

    Returns:
        the warped tensor with same shape as the input grid maps.

    Example:
        >>> import torch
        >>> from kornia.utils import create_meshgrid
        >>> grid = create_meshgrid(2, 2, False)  # 1x2x2x2
        >>> grid += 1  # apply offset in both directions
        >>> input = torch.ones(1, 1, 2, 2)
        >>> remap(input, grid[..., 0], grid[..., 1], align_corners=True)   # 1x1x2x2
        tensor([[[[1., 0.],
                  [0., 0.]]]])

    .. note::
        This function is often used in conjunction with :func:`kornia.utils.create_meshgrid`.
    """
    KORNIA_CHECK_IS_TENSOR(image, "Expected shape BxCxHxW")
    KORNIA_CHECK_IS_TENSOR(map_x, "Expected shape BxHxW")
    KORNIA_CHECK_IS_TENSOR(map_y, "Expected shape BxHxW")

    batch_size, _, height, width = image.shape

    # grid_sample need the grid between -1/1
    map_xy: Tensor = stack([map_x, map_y], -1)

    # normalize coordinates if not already normalized
    if not normalized_coordinates:
        map_xy = normalize_pixel_coordinates(map_xy, height, width)

    # simulate broadcasting since grid_sample does not support it
    map_xy = map_xy.expand(batch_size, -1, -1, -1)

    # warp the image tensor and return
    warped = grid_sample(image, map_xy, mode=mode, padding_mode=padding_mode, align_corners=align_corners)
    return warped


def invert_affine_transform(matrix: torch.Tensor) -> torch.Tensor:
    r"""Invert an affine transformation.

    The function computes an inverse affine transformation represented by
    2×3 matrix:

    .. math::
        \begin{bmatrix}
            a_{11} & a_{12} & b_{1} \\
            a_{21} & a_{22} & b_{2} \\
        \end{bmatrix}

    The result is also a 2×3 matrix of the same type as M.

    Args:
        matrix: original affine transform. The tensor must be
          in the shape of :math:`(B, 2, 3)`.

    Return:
        the reverse affine transform with shape :math:`(B, 2, 3)`.

    .. note::
        This function is often used in conjunction with :func:`warp_affine`.
    """
    if not isinstance(matrix, torch.Tensor):
        raise TypeError(f"Input matrix type is not a torch.Tensor. Got {type(matrix)}")

    if not (len(matrix.shape) == 3 and matrix.shape[-2:] == (2, 3)):
        raise ValueError(f"Input matrix must be a Bx2x3 tensor. Got {matrix.shape}")

    matrix_tmp: torch.Tensor = convert_affinematrix_to_homography(matrix)
    matrix_inv: torch.Tensor = torch.inverse(matrix_tmp)

    return matrix_inv[..., :2, :3]


def get_affine_matrix2d(
    translations: torch.Tensor,
    center: torch.Tensor,
    scale: torch.Tensor,
    angle: torch.Tensor,
    sx: Optional[torch.Tensor] = None,
    sy: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    r"""Compose affine matrix from the components.

    Args:
        translations: tensor containing the translation vector with shape :math:`(B, 2)`.
        center: tensor containing the center vector with shape :math:`(B, 2)`.
        scale: tensor containing the scale factor with shape :math:`(B, 2)`.
        angle: tensor of angles in degrees :math:`(B)`.
        sx: tensor containing the shear factor in the x-direction with shape :math:`(B)`.
        sy: tensor containing the shear factor in the y-direction with shape :math:`(B)`.

    Returns:
        the affine transformation matrix :math:`(B, 3, 3)`.

    .. note::
        This function is often used in conjunction with :func:`warp_affine`, :func:`warp_perspective`.
    """
    transform: torch.Tensor = get_rotation_matrix2d(center, -angle, scale)
    transform[..., 2] += translations  # tx/ty

    # pad transform to get Bx3x3
    transform_h = convert_affinematrix_to_homography(transform)

    if any(s is not None for s in [sx, sy]):
        shear_mat = get_shear_matrix2d(center, sx, sy)
        transform_h = transform_h @ shear_mat

    return transform_h


def get_shear_matrix2d(center: torch.Tensor, sx: Optional[torch.Tensor] = None, sy: Optional[torch.Tensor] = None):
    r"""Compose shear matrix Bx4x4 from the components.

    Note: Ordered shearing, shear x-axis then y-axis.

    .. math::
        \begin{bmatrix}
            1 & b \\
            a & ab + 1 \\
        \end{bmatrix}

    Args:
        center: shearing center coordinates of (x, y).
        sx: shearing degree along x axis.
        sy: shearing degree along y axis.

    Returns:
        params to be passed to the affine transformation with shape :math:`(B, 3, 3)`.

    Examples:
        >>> rng = torch.manual_seed(0)
        >>> sx = torch.randn(1)
        >>> sx
        tensor([1.5410])
        >>> center = torch.tensor([[0., 0.]])  # Bx2
        >>> get_shear_matrix2d(center, sx=sx)
        tensor([[[  1.0000, -33.5468,   0.0000],
                 [ -0.0000,   1.0000,   0.0000],
                 [  0.0000,   0.0000,   1.0000]]])

    .. note::
        This function is often used in conjunction with :func:`warp_affine`, :func:`warp_perspective`.
    """
    sx = torch.tensor([0.0]).repeat(center.size(0)) if sx is None else sx
    sy = torch.tensor([0.0]).repeat(center.size(0)) if sy is None else sy

    x, y = torch.split(center, 1, dim=-1)
    x, y = x.view(-1), y.view(-1)

    sx_tan = torch.tan(sx)  # type: ignore
    sy_tan = torch.tan(sy)  # type: ignore
    ones = torch.ones_like(sx)  # type: ignore
    shear_mat = torch.stack(
        [ones, -sx_tan, sx_tan * y, -sy_tan, ones + sx_tan * sy_tan, sy_tan * (sx_tan * y + x)], dim=-1  # type: ignore
    ).view(-1, 2, 3)

    shear_mat = convert_affinematrix_to_homography(shear_mat)
    return shear_mat


def get_affine_matrix3d(
    translations: torch.Tensor,
    center: torch.Tensor,
    scale: torch.Tensor,
    angles: torch.Tensor,
    sxy: Optional[torch.Tensor] = None,
    sxz: Optional[torch.Tensor] = None,
    syx: Optional[torch.Tensor] = None,
    syz: Optional[torch.Tensor] = None,
    szx: Optional[torch.Tensor] = None,
    szy: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    r"""Compose 3d affine matrix from the components.

    Args:
        translations: tensor containing the translation vector (dx,dy,dz) with shape :math:`(B, 3)`.
        center: tensor containing the center vector (x,y,z) with shape :math:`(B, 3)`.
        scale: tensor containing the scale factor with shape :math:`(B)`.
        angle: angle axis vector containing the rotation angles in degrees in the form
            of (rx, ry, rz) with shape :math:`(B, 3)`. Internally it calls Rodrigues to compute
            the rotation matrix from axis-angle.
        sxy: tensor containing the shear factor in the xy-direction with shape :math:`(B)`.
        sxz: tensor containing the shear factor in the xz-direction with shape :math:`(B)`.
        syx: tensor containing the shear factor in the yx-direction with shape :math:`(B)`.
        syz: tensor containing the shear factor in the yz-direction with shape :math:`(B)`.
        szx: tensor containing the shear factor in the zx-direction with shape :math:`(B)`.
        szy: tensor containing the shear factor in the zy-direction with shape :math:`(B)`.

    Returns:
        the 3d affine transformation matrix :math:`(B, 3, 3)`.

    .. note::
        This function is often used in conjunction with :func:`warp_perspective`.
    """
    transform: torch.Tensor = get_projective_transform(center, -angles, scale)
    transform[..., 3] += translations  # tx/ty/tz

    # pad transform to get Bx3x3
    transform_h = convert_affinematrix_to_homography3d(transform)
    if any(s is not None for s in [sxy, sxz, syx, syz, szx, szy]):
        shear_mat = get_shear_matrix3d(center, sxy, sxz, syx, syz, szx, szy)
        transform_h = transform_h @ shear_mat

    return transform_h


def get_shear_matrix3d(
    center: torch.Tensor,
    sxy: Optional[torch.Tensor] = None,
    sxz: Optional[torch.Tensor] = None,
    syx: Optional[torch.Tensor] = None,
    syz: Optional[torch.Tensor] = None,
    szx: Optional[torch.Tensor] = None,
    szy: Optional[torch.Tensor] = None,
):
    r"""Compose shear matrix Bx4x4 from the components.
    Note: Ordered shearing, shear x-axis then y-axis then z-axis.

    .. math::
        \begin{bmatrix}
            1 & o & r & oy + rz \\
            m & p & s & mx + py + sz -y \\
            n & q & t & nx + qy + tz -z \\
            0 & 0 & 0 & 1  \\
        \end{bmatrix}
        Where:
        m = S_{xy}
        n = S_{xz}
        o = S_{yx}
        p = S_{xy}S_{yx} + 1
        q = S_{xz}S_{yx} + S_{yz}
        r = S_{zx} + S_{yx}S_{zy}
        s = S_{xy}S_{zx} + (S_{xy}S_{yx} + 1)S_{zy}
        t = S_{xz}S_{zx} + (S_{xz}S_{yx} + S_{yz})S_{zy} + 1

    Params:
        center: shearing center coordinates of (x, y, z).
        sxy: shearing degree along x axis, towards y plane.
        sxz: shearing degree along x axis, towards z plane.
        syx: shearing degree along y axis, towards x plane.
        syz: shearing degree along y axis, towards z plane.
        szx: shearing degree along z axis, towards x plane.
        szy: shearing degree along z axis, towards y plane.

    Returns:
        params to be passed to the affine transformation.

    Examples:
        >>> rng = torch.manual_seed(0)
        >>> sxy, sxz, syx, syz = torch.randn(4, 1)
        >>> sxy, sxz, syx, syz
        (tensor([1.5410]), tensor([-0.2934]), tensor([-2.1788]), tensor([0.5684]))
        >>> center = torch.tensor([[0., 0., 0.]])  # Bx3
        >>> get_shear_matrix3d(center, sxy=sxy, sxz=sxz, syx=syx, syz=syz)
        tensor([[[  1.0000,  -1.4369,   0.0000,   0.0000],
                 [-33.5468,  49.2039,   0.0000,   0.0000],
                 [  0.3022,  -1.0729,   1.0000,   0.0000],
                 [  0.0000,   0.0000,   0.0000,   1.0000]]])

    .. note::
        This function is often used in conjunction with :func:`warp_perspective3d`.
    """
    sxy = torch.tensor([0.0]).repeat(center.size(0)) if sxy is None else sxy
    sxz = torch.tensor([0.0]).repeat(center.size(0)) if sxz is None else sxz
    syx = torch.tensor([0.0]).repeat(center.size(0)) if syx is None else syx
    syz = torch.tensor([0.0]).repeat(center.size(0)) if syz is None else syz
    szx = torch.tensor([0.0]).repeat(center.size(0)) if szx is None else szx
    szy = torch.tensor([0.0]).repeat(center.size(0)) if szy is None else szy

    x, y, z = torch.split(center, 1, dim=-1)
    x, y, z = x.view(-1), y.view(-1), z.view(-1)
    # Prepare parameters
    sxy_tan = torch.tan(sxy)  # type: ignore
    sxz_tan = torch.tan(sxz)  # type: ignore
    syx_tan = torch.tan(syx)  # type: ignore
    syz_tan = torch.tan(syz)  # type: ignore
    szx_tan = torch.tan(szx)  # type: ignore
    szy_tan = torch.tan(szy)  # type: ignore

    # compute translation matrix
    m00, m10, m20, m01, m11, m21, m02, m12, m22 = _compute_shear_matrix_3d(
        sxy_tan, sxz_tan, syx_tan, syz_tan, szx_tan, szy_tan
    )

    m03 = m01 * y + m02 * z
    m13 = m10 * x + m11 * y + m12 * z - y
    m23 = m20 * x + m21 * y + m22 * z - z

    # shear matrix is implemented with negative values
    sxy_tan, sxz_tan, syx_tan, syz_tan, szx_tan, szy_tan = -sxy_tan, -sxz_tan, -syx_tan, -syz_tan, -szx_tan, -szy_tan
    m00, m10, m20, m01, m11, m21, m02, m12, m22 = _compute_shear_matrix_3d(
        sxy_tan, sxz_tan, syx_tan, syz_tan, szx_tan, szy_tan
    )

    shear_mat = torch.stack([m00, m01, m02, m03, m10, m11, m12, m13, m20, m21, m22, m23], dim=-1).view(-1, 3, 4)
    shear_mat = convert_affinematrix_to_homography3d(shear_mat)

    return shear_mat


def _compute_shear_matrix_3d(sxy_tan, sxz_tan, syx_tan, syz_tan, szx_tan, szy_tan):
    ones = torch.ones_like(sxy_tan)  # type: ignore

    m00, m10, m20 = ones, sxy_tan, sxz_tan
    m01, m11, m21 = syx_tan, sxy_tan * syx_tan + ones, sxz_tan * syx_tan + syz_tan
    m02 = syx_tan * szy_tan + szx_tan
    m12 = sxy_tan * szx_tan + szy_tan * m11
    m22 = sxz_tan * szx_tan + szy_tan * m21 + ones
    return m00, m10, m20, m01, m11, m21, m02, m12, m22


def warp_affine3d(
    src: torch.Tensor,
    M: torch.Tensor,
    dsize: Tuple[int, int, int],
    flags: str = 'bilinear',
    padding_mode: str = 'zeros',
    align_corners: bool = True,
) -> torch.Tensor:
    r"""Apply a projective transformation a to 3d tensor.

    .. warning::
        This API signature it is experimental and might suffer some changes in the future.

    Args:
        src : input tensor of shape :math:`(B, C, D, H, W)`.
        M: projective transformation matrix of shape :math:`(B, 3, 4)`.
        dsize: size of the output image (depth, height, width).
        mode: interpolation mode to calculate output values
          ``'bilinear'`` | ``'nearest'``.
        padding_mode: padding mode for outside grid values
          ``'zeros'`` | ``'border'`` | ``'reflection'``.
        align_corners : mode for grid_generation.

    Returns:
        torch.Tensor: the warped 3d tensor with shape :math:`(B, C, D, H, W)`.

    .. note::
        This function is often used in conjunction with :func:`get_perspective_transform3d`.
    """
    if len(src.shape) != 5:
        raise AssertionError(src.shape)
    if not (len(M.shape) == 3 and M.shape[-2:] == (3, 4)):
        raise AssertionError(M.shape)
    if len(dsize) != 3:
        raise AssertionError(dsize)
    B, C, D, H, W = src.size()

    size_src: Tuple[int, int, int] = (D, H, W)
    size_out: Tuple[int, int, int] = dsize

    M_4x4 = convert_affinematrix_to_homography3d(M)  # Bx4x4

    # we need to normalize the transformation since grid sample needs -1/1 coordinates
    dst_norm_trans_src_norm: torch.Tensor = normalize_homography3d(M_4x4, size_src, size_out)  # Bx4x4

    src_norm_trans_dst_norm = _torch_inverse_cast(dst_norm_trans_src_norm)
    P_norm: torch.Tensor = src_norm_trans_dst_norm[:, :3]  # Bx3x4

    # compute meshgrid and apply to input
    dsize_out: List[int] = [B, C] + list(size_out)
    grid = torch.nn.functional.affine_grid(P_norm, dsize_out, align_corners=align_corners)
    return torch.nn.functional.grid_sample(
        src, grid, align_corners=align_corners, mode=flags, padding_mode=padding_mode
    )


def projection_from_Rt(rmat: torch.Tensor, tvec: torch.Tensor) -> torch.Tensor:
    r"""Compute the projection matrix from Rotation and translation.

    .. warning::
        This API signature it is experimental and might suffer some changes in the future.

    Concatenates the batch of rotations and translations such that :math:`P = [R | t]`.

    Args:
       rmat: the rotation matrix with shape :math:`(*, 3, 3)`.
       tvec: the translation vector with shape :math:`(*, 3, 1)`.

    Returns:
       the projection matrix with shape :math:`(*, 3, 4)`.

    """
    if not (len(rmat.shape) >= 2 and rmat.shape[-2:] == (3, 3)):
        raise AssertionError(rmat.shape)
    if not (len(tvec.shape) >= 2 and tvec.shape[-2:] == (3, 1)):
        raise AssertionError(tvec.shape)

    return torch.cat([rmat, tvec], dim=-1)  # Bx3x4


def get_projective_transform(center: torch.Tensor, angles: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    r"""Calculate the projection matrix for a 3D rotation.

    .. warning::
        This API signature it is experimental and might suffer some changes in the future.

    The function computes the projection matrix given the center and angles per axis.

    Args:
        center: center of the rotation (x,y,z) in the source with shape :math:`(B, 3)`.
        angles: angle axis vector containing the rotation angles in degrees in the form
            of (rx, ry, rz) with shape :math:`(B, 3)`. Internally it calls Rodrigues to compute
            the rotation matrix from axis-angle.
        scales: scale factor for x-y-z-directions with shape :math:`(B, 3)`.

    Returns:
        the projection matrix of 3D rotation with shape :math:`(B, 3, 4)`.

    .. note::
        This function is often used in conjunction with :func:`warp_affine3d`.
    """
    if not (len(center.shape) == 2 and center.shape[-1] == 3):
        raise AssertionError(center.shape)
    if not (len(angles.shape) == 2 and angles.shape[-1] == 3):
        raise AssertionError(angles.shape)
    if center.device != angles.device:
        raise AssertionError(center.device, angles.device)
    if center.dtype != angles.dtype:
        raise AssertionError(center.dtype, angles.dtype)

    # create rotation matrix
    angle_axis_rad: torch.Tensor = deg2rad(angles)
    rmat: torch.Tensor = angle_axis_to_rotation_matrix(angle_axis_rad)  # Bx3x3
    scaling_matrix: torch.Tensor = eye_like(3, rmat)
    scaling_matrix = scaling_matrix * scales.unsqueeze(dim=1)
    rmat = rmat @ scaling_matrix.to(rmat)

    # define matrix to move forth and back to origin
    from_origin_mat = torch.eye(4)[None].repeat(rmat.shape[0], 1, 1).type_as(center)  # Bx4x4
    from_origin_mat[..., :3, -1] += center

    to_origin_mat = from_origin_mat.clone()
    to_origin_mat = _torch_inverse_cast(from_origin_mat)

    # append translation with zeros
    proj_mat = projection_from_Rt(rmat, torch.zeros_like(center)[..., None])  # Bx3x4

    # chain 4x4 transforms
    proj_mat = convert_affinematrix_to_homography3d(proj_mat)  # Bx4x4
    proj_mat = from_origin_mat @ proj_mat @ to_origin_mat

    return proj_mat[..., :3, :]  # Bx3x4


def get_perspective_transform3d(src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
    r"""Calculate a 3d perspective transform from four pairs of the corresponding points.

    The function calculates the matrix of a perspective transform so that:

    .. math ::

        \begin{bmatrix}
        t_{i}x_{i}^{'} \\
        t_{i}y_{i}^{'} \\
        t_{i}z_{i}^{'} \\
        t_{i} \\
        \end{bmatrix}
        =
        \textbf{map_matrix} \cdot
        \begin{bmatrix}
        x_{i} \\
        y_{i} \\
        z_{i} \\
        1 \\
        \end{bmatrix}

    where

    .. math ::
        dst(i) = (x_{i}^{'},y_{i}^{'},z_{i}^{'}), src(i) = (x_{i}, y_{i}, z_{i}), i = 0,1,2,5,7

    Concrete math is as below:

    .. math ::

        \[ u_i =\frac{c_{00} * x_i + c_{01} * y_i + c_{02} * z_i + c_{03}}
            {c_{30} * x_i + c_{31} * y_i + c_{32} * z_i + c_{33}} \]
        \[ v_i =\frac{c_{10} * x_i + c_{11} * y_i + c_{12} * z_i + c_{13}}
            {c_{30} * x_i + c_{31} * y_i + c_{32} * z_i + c_{33}} \]
        \[ w_i =\frac{c_{20} * x_i + c_{21} * y_i + c_{22} * z_i + c_{23}}
            {c_{30} * x_i + c_{31} * y_i + c_{32} * z_i + c_{33}} \]

    .. math ::

        \begin{pmatrix}
        x_0 & y_0 & z_0 & 1 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & -x_0*u_0 & -y_0*u_0 & -z_0 * u_0 \\
        x_1 & y_1 & z_1 & 1 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & -x_1*u_1 & -y_1*u_1 & -z_1 * u_1 \\
        x_2 & y_2 & z_2 & 1 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & -x_2*u_2 & -y_2*u_2 & -z_2 * u_2 \\
        x_5 & y_5 & z_5 & 1 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & -x_5*u_5 & -y_5*u_5 & -z_5 * u_5 \\
        x_7 & y_7 & z_7 & 1 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & -x_7*u_7 & -y_7*u_7 & -z_7 * u_7 \\
        0 & 0 & 0 & 0 & x_0 & y_0 & z_0 & 1 & 0 & 0 & 0 & 0 & -x_0*v_0 & -y_0*v_0 & -z_0 * v_0 \\
        0 & 0 & 0 & 0 & x_1 & y_1 & z_1 & 1 & 0 & 0 & 0 & 0 & -x_1*v_1 & -y_1*v_1 & -z_1 * v_1 \\
        0 & 0 & 0 & 0 & x_2 & y_2 & z_2 & 1 & 0 & 0 & 0 & 0 & -x_2*v_2 & -y_2*v_2 & -z_2 * v_2 \\
        0 & 0 & 0 & 0 & x_5 & y_5 & z_5 & 1 & 0 & 0 & 0 & 0 & -x_5*v_5 & -y_5*v_5 & -z_5 * v_5 \\
        0 & 0 & 0 & 0 & x_7 & y_7 & z_7 & 1 & 0 & 0 & 0 & 0 & -x_7*v_7 & -y_7*v_7 & -z_7 * v_7 \\
        0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & x_0 & y_0 & z_0 & 1 & -x_0*w_0 & -y_0*w_0 & -z_0 * w_0 \\
        0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & x_1 & y_1 & z_1 & 1 & -x_1*w_1 & -y_1*w_1 & -z_1 * w_1 \\
        0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & x_2 & y_2 & z_2 & 1 & -x_2*w_2 & -y_2*w_2 & -z_2 * w_2 \\
        0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & x_5 & y_5 & z_5 & 1 & -x_5*w_5 & -y_5*w_5 & -z_5 * w_5 \\
        0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & x_7 & y_7 & z_7 & 1 & -x_7*w_7 & -y_7*w_7 & -z_7 * w_7 \\
        \end{pmatrix}

    Args:
        src: coordinates of quadrangle vertices in the source image with shape :math:`(B, 8, 3)`.
        dst: coordinates of the corresponding quadrangle vertices in
            the destination image with shape :math:`(B, 8, 3)`.

    Returns:
        the perspective transformation with shape :math:`(B, 4, 4)`.

    .. note::
        This function is often used in conjunction with :func:`warp_perspective3d`.
    """
    if not isinstance(src, (torch.Tensor)):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(src)}")

    if not isinstance(dst, (torch.Tensor)):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(dst)}")

    if not src.shape[-2:] == (8, 3):
        raise ValueError(f"Inputs must be a Bx8x3 tensor. Got {src.shape}")

    if not src.shape == dst.shape:
        raise ValueError(f"Inputs must have the same shape. Got {dst.shape}")

    if not (src.shape[0] == dst.shape[0]):
        raise ValueError(f"Inputs must have same batch size dimension. Expect {src.shape} but got {dst.shape}")

    if not (src.device == dst.device and src.dtype == dst.dtype):
        raise AssertionError(
            f"Expect `src` and `dst` to be in the same device (Got {src.dtype}, {dst.dtype}) "
            f"with the same dtype (Got {src.dtype}, {dst.dtype})."
        )

    # we build matrix A by using only 4 point correspondence. The linear
    # system is solved with the least square method, so here
    # we could even pass more correspondence
    p = []

    # 000, 100, 110, 101, 011
    for i in [0, 1, 2, 5, 7]:
        p.append(_build_perspective_param3d(src[:, i], dst[:, i], 'x'))
        p.append(_build_perspective_param3d(src[:, i], dst[:, i], 'y'))
        p.append(_build_perspective_param3d(src[:, i], dst[:, i], 'z'))

    # A is Bx15x15
    A = torch.stack(p, dim=1)

    # b is a Bx15x1
    b = torch.stack(
        [
            dst[:, 0:1, 0],
            dst[:, 0:1, 1],
            dst[:, 0:1, 2],
            dst[:, 1:2, 0],
            dst[:, 1:2, 1],
            dst[:, 1:2, 2],
            dst[:, 2:3, 0],
            dst[:, 2:3, 1],
            dst[:, 2:3, 2],
            # dst[:, 3:4, 0], dst[:, 3:4, 1], dst[:, 3:4, 2],
            # dst[:, 4:5, 0], dst[:, 4:5, 1], dst[:, 4:5, 2],
            dst[:, 5:6, 0],
            dst[:, 5:6, 1],
            dst[:, 5:6, 2],
            # dst[:, 6:7, 0], dst[:, 6:7, 1], dst[:, 6:7, 2],
            dst[:, 7:8, 0],
            dst[:, 7:8, 1],
            dst[:, 7:8, 2],
        ],
        dim=1,
    )

    # solve the system Ax = b
    X, _ = _torch_solve_cast(b, A)

    # create variable to return
    batch_size = src.shape[0]
    M = torch.ones(batch_size, 16, device=src.device, dtype=src.dtype)
    M[..., :15] = torch.squeeze(X, dim=-1)
    return M.view(-1, 4, 4)  # Bx4x4


def _build_perspective_param3d(p: torch.Tensor, q: torch.Tensor, axis: str) -> torch.Tensor:
    ones = torch.ones_like(p)[..., 0:1]
    zeros = torch.zeros_like(p)[..., 0:1]

    if axis == 'x':
        return torch.cat(
            [
                p[:, 0:1],
                p[:, 1:2],
                p[:, 2:3],
                ones,
                zeros,
                zeros,
                zeros,
                zeros,
                zeros,
                zeros,
                zeros,
                zeros,
                -p[:, 0:1] * q[:, 0:1],
                -p[:, 1:2] * q[:, 0:1],
                -p[:, 2:3] * q[:, 0:1],
            ],
            dim=1,
        )

    if axis == 'y':
        return torch.cat(
            [
                zeros,
                zeros,
                zeros,
                zeros,
                p[:, 0:1],
                p[:, 1:2],
                p[:, 2:3],
                ones,
                zeros,
                zeros,
                zeros,
                zeros,
                -p[:, 0:1] * q[:, 1:2],
                -p[:, 1:2] * q[:, 1:2],
                -p[:, 2:3] * q[:, 1:2],
            ],
            dim=1,
        )

    if axis == 'z':
        return torch.cat(
            [
                zeros,
                zeros,
                zeros,
                zeros,
                zeros,
                zeros,
                zeros,
                zeros,
                p[:, 0:1],
                p[:, 1:2],
                p[:, 2:3],
                ones,
                -p[:, 0:1] * q[:, 2:3],
                -p[:, 1:2] * q[:, 2:3],
                -p[:, 2:3] * q[:, 2:3],
            ],
            dim=1,
        )

    raise NotImplementedError(f"perspective params for axis `{axis}` is not implemented.")


def warp_perspective3d(
    src: torch.Tensor,
    M: torch.Tensor,
    dsize: Tuple[int, int, int],
    flags: str = 'bilinear',
    border_mode: str = 'zeros',
    align_corners: bool = False,
) -> torch.Tensor:
    r"""Apply a perspective transformation to an image.

    The function warp_perspective transforms the source image using
    the specified matrix:

    .. math::
        \text{dst} (x, y) = \text{src} \left(
        \frac{M_{11} x + M_{12} y + M_{13}}{M_{31} x + M_{32} y + M_{33}} ,
        \frac{M_{21} x + M_{22} y + M_{23}}{M_{31} x + M_{32} y + M_{33}}
        \right )

    Args:
        src: input image with shape :math:`(B, C, D, H, W)`.
        M: transformation matrix with shape :math:`(B, 4, 4)`.
        dsize: size of the output image (height, width).
        flags: interpolation mode to calculate output values
          ``'bilinear'`` | ``'nearest'``.
        border_mode: padding mode for outside grid values
          ``'zeros'`` | ``'border'`` | ``'reflection'``.
        align_corners: interpolation flag.

    Returns:
        the warped input image :math:`(B, C, D, H, W)`.

    .. note::
        This function is often used in conjunction with :func:`get_perspective_transform3d`.
    """
    if not isinstance(src, torch.Tensor):
        raise TypeError(f"Input src type is not a torch.Tensor. Got {type(src)}")

    if not isinstance(M, torch.Tensor):
        raise TypeError(f"Input M type is not a torch.Tensor. Got {type(M)}")

    if not len(src.shape) == 5:
        raise ValueError(f"Input src must be a BxCxDxHxW tensor. Got {src.shape}")

    if not (len(M.shape) == 3 or M.shape[-2:] == (4, 4)):
        raise ValueError(f"Input M must be a Bx4x4 tensor. Got {M.shape}")

    # launches the warper
    d, h, w = src.shape[-3:]
    return _transform_warp_impl3d(src, M, (d, h, w), dsize, flags, border_mode, align_corners)


def homography_warp(
    patch_src: torch.Tensor,
    src_homo_dst: torch.Tensor,
    dsize: Tuple[int, int],
    mode: str = 'bilinear',
    padding_mode: str = 'zeros',
    align_corners: bool = False,
    normalized_coordinates: bool = True,
    normalized_homography: bool = True,
) -> torch.Tensor:
    r"""Warp image patches or tensors by normalized 2D homographies.

    See :class:`~kornia.geometry.warp.HomographyWarper` for details.

    Args:
        patch_src: The image or tensor to warp. Should be from source of shape :math:`(N, C, H, W)`.
        src_homo_dst: The homography or stack of homographies from destination to source of shape :math:`(N, 3, 3)`.
        dsize:
          if homography normalized: The height and width of the image to warp.
          if homography not normalized: size of the output image (height, width).
        mode: interpolation mode to calculate output values ``'bilinear'`` | ``'nearest'``.
        padding_mode: padding mode for outside grid values ``'zeros'`` | ``'border'`` | ``'reflection'``.
        align_corners: interpolation flag.
        normalized_coordinates: Whether the homography assumes [-1, 1] normalized coordinates or not.
        normalized_homography: show is homography normalized.

    Return:
        Patch sampled at locations from source to destination.

    Example:
        >>> input = torch.rand(1, 3, 32, 32)
        >>> homography = torch.eye(3).view(1, 3, 3)
        >>> output = homography_warp(input, homography, (32, 32))

    Example
        >>> img = torch.rand(1, 4, 5, 6)
        >>> H = torch.eye(3)[None]
        >>> out = homography_warp(img, H, (4, 2), align_corners=True, normalized_homography=False)
        >>> print(out.shape)
        torch.Size([1, 4, 4, 2])

    """
    if not src_homo_dst.device == patch_src.device:
        raise TypeError(
            "Patch and homography must be on the same device. \
                         Got patch.device: {} src_H_dst.device: {}.".format(
                patch_src.device, src_homo_dst.device
            )
        )
    if normalized_homography:
        height, width = dsize
        grid = create_meshgrid(
            height, width, normalized_coordinates=normalized_coordinates, device=patch_src.device, dtype=patch_src.dtype
        )
        warped_grid = warp_grid(grid, src_homo_dst)

        return F.grid_sample(patch_src, warped_grid, mode=mode, padding_mode=padding_mode, align_corners=align_corners)
    return warp_perspective(
        patch_src, src_homo_dst, dsize, mode='bilinear', padding_mode=padding_mode, align_corners=True
    )


def _transform_warp_impl3d(
    src: torch.Tensor,
    dst_pix_trans_src_pix: torch.Tensor,
    dsize_src: Tuple[int, int, int],
    dsize_dst: Tuple[int, int, int],
    grid_mode: str,
    padding_mode: str,
    align_corners: bool,
) -> torch.Tensor:
    """Compute the transform in normalized coordinates and perform the warping."""
    dst_norm_trans_src_norm: torch.Tensor = normalize_homography3d(dst_pix_trans_src_pix, dsize_src, dsize_dst)

    src_norm_trans_dst_norm = torch.inverse(dst_norm_trans_src_norm)
    return homography_warp3d(src, src_norm_trans_dst_norm, dsize_dst, grid_mode, padding_mode, align_corners, True)


def homography_warp3d(
    patch_src: torch.Tensor,
    src_homo_dst: torch.Tensor,
    dsize: Tuple[int, int, int],
    mode: str = 'bilinear',
    padding_mode: str = 'zeros',
    align_corners: bool = False,
    normalized_coordinates: bool = True,
) -> torch.Tensor:
    r"""Warp image patches or tensors by normalized 3D homographies.

    Args:
        patch_src: The image or tensor to warp. Should be from source of shape :math:`(N, C, D, H, W)`.
        src_homo_dst: The homography or stack of homographies from destination to source of shape
          :math:`(N, 4, 4)`.
        dsize: The height and width of the image to warp.
        mode: interpolation mode to calculate output values ``'bilinear'`` | ``'nearest'``.
        padding_mode: padding mode for outside grid values ``'zeros'`` | ``'border'`` | ``'reflection'``.
        align_corners: interpolation flag.
        normalized_coordinates: Whether the homography assumes [-1, 1] normalized coordinates or not.

    Return:
        Patch sampled at locations from source to destination.

    Example:
        >>> input = torch.rand(1, 3, 32, 32)
        >>> homography = torch.eye(3).view(1, 3, 3)
        >>> output = homography_warp(input, homography, (32, 32))
    """
    if not src_homo_dst.device == patch_src.device:
        raise TypeError(
            "Patch and homography must be on the same device. \
                         Got patch.device: {} src_H_dst.device: {}.".format(
                patch_src.device, src_homo_dst.device
            )
        )

    depth, height, width = dsize
    grid = create_meshgrid3d(
        depth, height, width, normalized_coordinates=normalized_coordinates, device=patch_src.device
    )
    warped_grid = warp_grid3d(grid, src_homo_dst)

    return F.grid_sample(patch_src, warped_grid, mode=mode, padding_mode=padding_mode, align_corners=align_corners)
