"""Module to perform projective transformations to tensors."""
import warnings
from typing import List, Optional, Tuple

import torch

import kornia as K
from kornia.geometry.conversions import convert_affinematrix_to_homography3d
from kornia.geometry.transform.homography_warper import homography_warp3d, normalize_homography3d
from kornia.testing import check_is_tensor
from kornia.utils.helpers import _torch_inverse_cast, _torch_solve_cast

__all__ = [
    "warp_affine3d",
    "get_projective_transform",
    "projection_from_Rt",
    "get_perspective_transform3d",
    "warp_perspective3d",
]


def warp_affine3d(
    src: torch.Tensor,
    M: torch.Tensor,
    dsize: Tuple[int, int, int],
    flags: str = 'bilinear',
    padding_mode: str = 'zeros',
    align_corners: Optional[bool] = None,
) -> torch.Tensor:
    r"""Applies a projective transformation a to 3d tensor.

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
        This function is often used in conjuntion with :func:`get_perspective_transform3d`.
    """
    assert len(src.shape) == 5, src.shape
    assert len(M.shape) == 3 and M.shape[-2:] == (3, 4), M.shape
    assert len(dsize) == 3, dsize
    B, C, D, H, W = src.size()

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
    assert len(rmat.shape) >= 2 and rmat.shape[-2:] == (3, 3), rmat.shape
    assert len(tvec.shape) >= 2 and tvec.shape[-2:] == (3, 1), tvec.shape

    return torch.cat([rmat, tvec], dim=-1)  # Bx3x4


def get_projective_transform(center: torch.Tensor, angles: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    r"""Calculates the projection matrix for a 3D rotation.

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
        This function is often used in conjuntion with :func:`warp_affine3d`.
    """
    assert len(center.shape) == 2 and center.shape[-1] == 3, center.shape
    assert len(angles.shape) == 2 and angles.shape[-1] == 3, angles.shape
    assert center.device == angles.device, (center.device, angles.device)
    assert center.dtype == angles.dtype, (center.dtype, angles.dtype)

    # create rotation matrix
    angle_axis_rad: torch.Tensor = K.deg2rad(angles)
    rmat: torch.Tensor = K.angle_axis_to_rotation_matrix(angle_axis_rad)  # Bx3x3
    scaling_matrix: torch.Tensor = K.eye_like(3, rmat)
    scaling_matrix = scaling_matrix * scales.unsqueeze(dim=1)
    rmat = rmat @ scaling_matrix.to(rmat)

    # define matrix to move forth and back to origin
    from_origin_mat = torch.eye(4)[None].repeat(rmat.shape[0], 1, 1).type_as(center)  # Bx4x4
    from_origin_mat[..., :3, -1] += center

    to_origin_mat = from_origin_mat.clone()
    to_origin_mat = _torch_inverse_cast(from_origin_mat)

    # append tranlation with zeros
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
        This function is often used in conjuntion with :func:`warp_perspective3d`.
    """
    if not isinstance(src, (torch.Tensor)):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(type(src)))

    if not isinstance(dst, (torch.Tensor)):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(type(dst)))

    if not src.shape[-2:] == (8, 3):
        raise ValueError("Inputs must be a Bx8x3 tensor. Got {}".format(src.shape))

    if not src.shape == dst.shape:
        raise ValueError("Inputs must have the same shape. Got {}".format(dst.shape))

    if not (src.shape[0] == dst.shape[0]):
        raise ValueError(
            "Inputs must have same batch size dimension. Expect {} but got {}".format(src.shape, dst.shape)
        )

    assert src.device == dst.device and src.dtype == dst.dtype, (
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
    X, LU = _torch_solve_cast(b, A)

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
    r"""Applies a perspective transformation to an image.

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
        This function is often used in conjuntion with :func:`get_perspective_transform3d`.
    """
    check_is_tensor(src)
    check_is_tensor(M)

    if not len(src.shape) == 5:
        raise ValueError("Input src must be a BxCxDxHxW tensor. Got {}".format(src.shape))

    if not (len(M.shape) == 3 or M.shape[-2:] == (4, 4)):
        raise ValueError("Input M must be a Bx4x4 tensor. Got {}".format(M.shape))

    # launches the warper
    d, h, w = src.shape[-3:]
    return transform_warp_impl3d(src, M, (d, h, w), dsize, flags, border_mode, align_corners)


def transform_warp_impl3d(
    src: torch.Tensor,
    dst_pix_trans_src_pix: torch.Tensor,
    dsize_src: Tuple[int, int, int],
    dsize_dst: Tuple[int, int, int],
    grid_mode: str,
    padding_mode: str,
    align_corners: bool,
) -> torch.Tensor:
    """Compute the transform in normalized cooridnates and perform the warping."""
    dst_norm_trans_src_norm: torch.Tensor = normalize_homography3d(dst_pix_trans_src_pix, dsize_src, dsize_dst)

    src_norm_trans_dst_norm = torch.inverse(dst_norm_trans_src_norm)
    return homography_warp3d(src, src_norm_trans_dst_norm, dsize_dst, grid_mode, padding_mode, align_corners, True)
