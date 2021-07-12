import warnings
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from kornia.geometry.conversions import (
    convert_affinematrix_to_homography,
    convert_affinematrix_to_homography3d,
    deg2rad,
    normalize_pixel_coordinates,
)
from kornia.geometry.linalg import transform_points
from kornia.geometry.transform.homography_warper import normalize_homography
from kornia.geometry.transform.projwarp import get_projective_transform
from kornia.utils import create_meshgrid
from kornia.utils.helpers import _torch_inverse_cast, _torch_solve_cast

__all__ = [
    "warp_perspective",
    "warp_affine",
    "get_perspective_transform",
    "get_rotation_matrix2d",
    "remap",
    "invert_affine_transform",
    "angle_to_rotation_matrix",
    "get_affine_matrix2d",
    "get_affine_matrix3d",
    "get_shear_matrix2d",
    "get_shear_matrix3d",
]


def warp_perspective(
    src: torch.Tensor,
    M: torch.Tensor,
    dsize: Tuple[int, int],
    mode: str = 'bilinear',
    padding_mode: str = 'zeros',
    align_corners: Optional[bool] = None,
) -> torch.Tensor:
    r"""Applies a perspective transformation to an image.

    .. image:: https://kornia-tutorials.readthedocs.io/en/latest/_images/warp_perspective_10_2.png

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
        padding_mode: padding mode for outside grid values ``'zeros'`` | ``'border'`` | ``'reflection'``.
        align_corners(bool, optional): interpolation flag.

    Returns:
        the warped input image :math:`(B, C, H, W)`.

    Example:
       >>> img = torch.rand(1, 4, 5, 6)
       >>> H = torch.eye(3)[None]
       >>> out = warp_perspective(img, H, (4, 2), align_corners=True)
       >>> print(out.shape)
       torch.Size([1, 4, 4, 2])

    .. note::
        This function is often used in conjuntion with :func:`get_perspective_transform`.

    .. note::
        See a working example `here <https://kornia-tutorials.readthedocs.io/en/
        latest/warp_perspective.html>`_.
    """
    if not isinstance(src, torch.Tensor):
        raise TypeError("Input src type is not a torch.Tensor. Got {}".format(type(src)))

    if not isinstance(M, torch.Tensor):
        raise TypeError("Input M type is not a torch.Tensor. Got {}".format(type(M)))

    if not len(src.shape) == 4:
        raise ValueError("Input src must be a BxCxHxW tensor. Got {}".format(src.shape))

    if not (len(M.shape) == 3 and M.shape[-2:] == (3, 3)):
        raise ValueError("Input M must be a Bx3x3 tensor. Got {}".format(M.shape))

    # TODO: remove the statement below in kornia v0.6
    if align_corners is None:
        message: str = (
            "The align_corners default value has been changed. By default now is set True "
            "in order to match cv2.warpPerspective. In case you want to keep your previous "
            "behaviour set it to False. This warning will disappear in kornia > v0.6."
        )
        warnings.warn(message)
        # set default value for align corners
        align_corners = True

    B, C, H, W = src.size()
    h_out, w_out = dsize

    # we normalize the 3x3 transformation matrix and convert to 3x4
    dst_norm_trans_src_norm: torch.Tensor = normalize_homography(M, (H, W), (h_out, w_out))  # Bx3x3

    src_norm_trans_dst_norm = _torch_inverse_cast(dst_norm_trans_src_norm)  # Bx3x3

    # this piece of code substitutes F.affine_grid since it does not support 3x3
    grid = (
        create_meshgrid(h_out, w_out, normalized_coordinates=True, device=src.device).to(src.dtype).repeat(B, 1, 1, 1)
    )
    grid = transform_points(src_norm_trans_dst_norm[:, None, None], grid)

    return F.grid_sample(src, grid, align_corners=align_corners, mode=mode, padding_mode=padding_mode)


def warp_affine(
    src: torch.Tensor,
    M: torch.Tensor,
    dsize: Tuple[int, int],
    mode: str = 'bilinear',
    padding_mode: str = 'zeros',
    align_corners: Optional[bool] = None,
) -> torch.Tensor:
    r"""Applies an affine transformation to a tensor.

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
        padding_mode (str): padding mode for outside grid values ``'zeros'`` | ``'border'`` | ``'reflection'``.
        align_corners : mode for grid_generation.

    Returns:
        the warped tensor with shape :math:`(B, C, H, W)`.

    .. note::
        This function is often used in conjuntion with :func:`get_rotation_matrix2d`,
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
        raise TypeError("Input src type is not a torch.Tensor. Got {}".format(type(src)))

    if not isinstance(M, torch.Tensor):
        raise TypeError("Input M type is not a torch.Tensor. Got {}".format(type(M)))

    if not len(src.shape) == 4:
        raise ValueError("Input src must be a BxCxHxW tensor. Got {}".format(src.shape))

    if not (len(M.shape) == 3 or M.shape[-2:] == (2, 3)):
        raise ValueError("Input M must be a Bx2x3 tensor. Got {}".format(M.shape))

    # TODO: remove the statement below in kornia v0.6
    if align_corners is None:
        message: str = (
            "The align_corners default value has been changed. By default now is set True "
            "in order to match cv2.warpAffine. In case you want to keep your previous "
            "behaviour set it to False. This warning will disappear in kornia > v0.6."
        )
        warnings.warn(message)
        # set default value for align corners
        align_corners = True

    B, C, H, W = src.size()

    # we generate a 3x3 transformation matrix from 2x3 affine
    M_3x3: torch.Tensor = convert_affinematrix_to_homography(M)
    dst_norm_trans_src_norm: torch.Tensor = normalize_homography(M_3x3, (H, W), dsize)

    # src_norm_trans_dst_norm = torch.inverse(dst_norm_trans_src_norm)
    src_norm_trans_dst_norm = _torch_inverse_cast(dst_norm_trans_src_norm)

    grid = F.affine_grid(src_norm_trans_dst_norm[:, :2, :], [B, C, dsize[0], dsize[1]], align_corners=align_corners)

    return F.grid_sample(src, grid, align_corners=align_corners, mode=mode, padding_mode=padding_mode)


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
        src: coordinates of quadrangle vertices in the source image with shape :math:`(B, 4, 2)`.
        dst: coordinates of the corresponding quadrangle vertices in
            the destination image with shape :math:`(B, 4, 2)`.

    Returns:
        the perspective transformation with shape :math:`(B, 3, 3)`.

    .. note::
        This function is often used in conjuntion with :func:`warp_perspective`.
    """
    if not isinstance(src, torch.Tensor):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(type(src)))

    if not isinstance(dst, torch.Tensor):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(type(dst)))

    if not src.shape[-2:] == (4, 2):
        raise ValueError("Inputs must be a Bx4x2 tensor. Got {}".format(src.shape))

    if not src.shape == dst.shape:
        raise ValueError("Inputs must have the same shape. Got {}".format(dst.shape))

    if not (src.shape[0] == dst.shape[0]):
        raise ValueError(
            "Inputs must have same batch size dimension. Expect {} but got {}".format(src.shape, dst.shape)
        )

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
    X, LU = _torch_solve_cast(b, A)

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


def angle_to_rotation_matrix(angle: torch.Tensor) -> torch.Tensor:
    r"""Create a rotation matrix out of angles in degrees.

    Args:
        angle: tensor of angles in degrees, any shape.

    Returns:
        tensor of *x2x2 rotation matrices.

    Shape:
        - Input: :math:`(*)`
        - Output: :math:`(*, 2, 2)`

    Example:
        >>> input = torch.rand(1, 3)  # Nx3
        >>> output = angle_to_rotation_matrix(input)  # Nx3x2x2
    """
    ang_rad = deg2rad(angle)
    cos_a: torch.Tensor = torch.cos(ang_rad)
    sin_a: torch.Tensor = torch.sin(ang_rad)
    return torch.stack([cos_a, sin_a, -sin_a, cos_a], dim=-1).view(*angle.shape, 2, 2)


def get_rotation_matrix2d(center: torch.Tensor, angle: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
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
        This function is often used in conjuntion with :func:`warp_affine`.
    """
    if not isinstance(center, torch.Tensor):
        raise TypeError("Input center type is not a torch.Tensor. Got {}".format(type(center)))

    if not isinstance(angle, torch.Tensor):
        raise TypeError("Input angle type is not a torch.Tensor. Got {}".format(type(angle)))

    if not isinstance(scale, torch.Tensor):
        raise TypeError("Input scale type is not a torch.Tensor. Got {}".format(type(scale)))

    if not (len(center.shape) == 2 and center.shape[1] == 2):
        raise ValueError("Input center must be a Bx2 tensor. Got {}".format(center.shape))

    if not len(angle.shape) == 1:
        raise ValueError("Input angle must be a B tensor. Got {}".format(angle.shape))

    if not (len(scale.shape) == 2 and scale.shape[1] == 2):
        raise ValueError("Input scale must be a Bx2 tensor. Got {}".format(scale.shape))

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

    # convert angle and apply scale
    rotation_matrix: torch.Tensor = angle_to_rotation_matrix(angle)
    scaling_matrix: torch.Tensor = (
        torch.zeros((2, 2), device=rotation_matrix.device, dtype=rotation_matrix.dtype)
        .fill_diagonal_(1)
        .repeat(rotation_matrix.size(0), 1, 1)
    )

    scaling_matrix = scaling_matrix * scale.unsqueeze(dim=2).repeat(1, 1, 2)
    scaled_rotation: torch.Tensor = rotation_matrix @ scaling_matrix
    alpha: torch.Tensor = scaled_rotation[:, 0, 0]
    beta: torch.Tensor = scaled_rotation[:, 0, 1]

    # unpack the center to x, y coordinates
    x: torch.Tensor = center[..., 0]
    y: torch.Tensor = center[..., 1]

    # create output tensor
    batch_size: int = center.shape[0]
    one = torch.tensor(1.0, device=center.device, dtype=center.dtype)
    M: torch.Tensor = torch.zeros(batch_size, 2, 3, device=center.device, dtype=center.dtype)

    M[..., 0:2, 0:2] = scaled_rotation
    M[..., 0, 2] = (one - alpha) * x - beta * y
    M[..., 1, 2] = beta * x + (one - alpha) * y
    return M


def remap(
    tensor: torch.Tensor,
    map_x: torch.Tensor,
    map_y: torch.Tensor,
    mode: str = 'bilinear',
    padding_mode: str = 'zeros',
    align_corners: Optional[bool] = None,
    normalized_coordinates: bool = False,
) -> torch.Tensor:
    r"""Applies a generic geometrical transformation to a tensor.

    .. image:: _static/img/remap.png

    The function remap transforms the source tensor using the specified map:

    .. math::
        \text{dst}(x, y) = \text{src}(map_x(x, y), map_y(x, y))

    Args:
        tensor: the tensor to remap with shape (B, D, H, W).
          Where D is the number of channels.
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
           normalised in the range of [-1, 1].

    Returns:
        the warped tensor with same shape as the input grid maps.

    Example:
        >>> from kornia.utils import create_meshgrid
        >>> grid = create_meshgrid(2, 2, False)  # 1x2x2x2
        >>> grid += 1  # apply offset in both directions
        >>> input = torch.ones(1, 1, 2, 2)
        >>> remap(input, grid[..., 0], grid[..., 1], align_corners=True)   # 1x1x2x2
        tensor([[[[1., 0.],
                  [0., 0.]]]])

    .. note::
        This function is often used in conjuntion with :func:`kornia.utils.create_meshgrid`.
    """
    if not isinstance(tensor, torch.Tensor):
        raise TypeError("Input tensor type is not a torch.Tensor. Got {}".format(type(tensor)))

    if not isinstance(map_x, torch.Tensor):
        raise TypeError("Input map_x type is not a torch.Tensor. Got {}".format(type(map_x)))

    if not isinstance(map_y, torch.Tensor):
        raise TypeError("Input map_y type is not a torch.Tensor. Got {}".format(type(map_y)))

    if not tensor.shape[-2:] == map_x.shape[-2:] == map_y.shape[-2:]:
        raise ValueError("Inputs last two dimensions must match.")

    batch_size, _, height, width = tensor.shape

    # grid_sample need the grid between -1/1
    map_xy: torch.Tensor = torch.stack([map_x, map_y], dim=-1)

    # normalize coordinates if not already normalized
    if not normalized_coordinates:
        map_xy = normalize_pixel_coordinates(map_xy, height, width)

    # simulate broadcasting since grid_sample does not support it
    map_xy_norm: torch.Tensor = map_xy.expand(batch_size, -1, -1, -1)

    # warp ans return
    tensor_warped: torch.Tensor = F.grid_sample(
        tensor, map_xy_norm, mode=mode, padding_mode=padding_mode, align_corners=align_corners
    )
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
        matrix: original affine transform. The tensor must be
          in the shape of :math:`(B, 2, 3)`.

    Return:
        the reverse affine transform with shape :math:`(B, 2, 3)`.

    .. note::
        This function is often used in conjuntion with :func:`warp_affine`.
    """
    if not isinstance(matrix, torch.Tensor):
        raise TypeError("Input matrix type is not a torch.Tensor. Got {}".format(type(matrix)))

    if not (len(matrix.shape) == 3 and matrix.shape[-2:] == (2, 3)):
        raise ValueError("Input matrix must be a Bx2x3 tensor. Got {}".format(matrix.shape))

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
    r"""Composes affine matrix from the components.

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
        This function is often used in conjuntion with :func:`warp_affine`, :func:`warp_perspective`.
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
    r"""Composes shear matrix Bx4x4 from the components.

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
        This function is often used in conjuntion with :func:`warp_affine`, :func:`warp_perspective`.
    """
    sx = torch.tensor([0.0]).repeat(center.size(0)) if sx is None else sx
    sy = torch.tensor([0.0]).repeat(center.size(0)) if sy is None else sy

    x, y = torch.split(center, 1, dim=-1)
    x, y = x.view(-1), y.view(-1)

    sx_tan = torch.tan(sx)  # type: ignore
    sy_tan = torch.tan(sy)  # type: ignore
    ones = torch.ones_like(sx)  # type: ignore
    shear_mat = torch.stack(
        [
            ones,
            -sx_tan,
            sx_tan * y,  # type: ignore   # noqa: E241
            -sy_tan,
            ones + sx_tan * sy_tan,
            sy_tan * (sx_tan * y + x),  # noqa: E241
        ],
        dim=-1,
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
    r"""Composes 3d affine matrix from the components.

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
        This function is often used in conjuntion with :func:`warp_perspective`.
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
    r"""Composes shear matrix Bx4x4 from the components.
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
        This function is often used in conjuntion with :func:`warp_perspective3d`.
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
    zeros = torch.zeros_like(sxy_tan)  # type: ignore
    ones = torch.ones_like(sxy_tan)  # type: ignore

    m00, m10, m20 = ones, sxy_tan, sxz_tan
    m01, m11, m21 = syx_tan, sxy_tan * syx_tan + ones, sxz_tan * syx_tan + syz_tan
    m02 = syx_tan * szy_tan + szx_tan
    m12 = sxy_tan * szx_tan + szy_tan * m11
    m22 = sxz_tan * szx_tan + szy_tan * m21 + ones
    return m00, m10, m20, m01, m11, m21, m02, m12, m22
