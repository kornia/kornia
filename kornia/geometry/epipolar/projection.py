"""Module for image projections."""
from typing import Union

import torch

from kornia.geometry.epipolar import numeric


def intrinsics_like(focal: float, input: torch.Tensor) -> torch.Tensor:
    r"""Returns a 3x3 instrinsics matrix, with same size as the input.

    The center of projection will be based in the input image size.

    Args:
        focal (float): the focal length for tha camera matrix.
        input (torch.Tensor): image tensor that will determine the batch size and image height
          and width. It is assumed to be a tensor in the shape of :math:`(B, C, H, W)`.

    Returns:
        torch.Tensor: The camera matrix with the shape of :math:`(B, 3, 3)`.

    """
    assert len(input.shape) == 4, input.shape
    assert focal > 0, focal

    B, _, H, W = input.shape

    intrinsics = numeric.eye_like(3, input)
    intrinsics[..., 0, 0] *= focal
    intrinsics[..., 1, 1] *= focal
    intrinsics[..., 0, 2] += 1. * W / 2
    intrinsics[..., 1, 2] += 1. * H / 2
    return intrinsics


def random_intrinsics(low: Union[float, torch.Tensor],
                      high: Union[float, torch.Tensor]) -> torch.Tensor:
    r"""Generates a random camera matrix based on a given uniform distribution.

    Args:
        low (Union[float, torch.Tensor]): lower range (inclusive).
        high (Union[float, torch.Tensor]): upper range (exclusive).

    Returns:
        torch.Tensor: The random camera matrix with the shape of :math:`(1, 3, 3)`.

    """
    sampler = torch.distributions.Uniform(low, high)
    fx, fy, cx, cy = [sampler.sample((1,)) for _ in range(4)]
    zeros, ones = torch.zeros_like(fx), torch.ones_like(fx)
    camera_matrix: torch.Tensor = torch.cat([
        fx, zeros, cx,
        zeros, fy, cy,
        zeros, zeros, ones,
    ])
    return camera_matrix.view(1, 3, 3)


def scale_intrinsics(
        camera_matrix: torch.Tensor, scale_factor: Union[float, torch.Tensor]) -> torch.Tensor:
    r"""Scale a camera matrix containing the intrinsics.

    Applies the scaling factor to the focal length and center of projection.

    Args:
        camera_matrix (torch.Tensor): the camera calibration matrix containing the intrinsic
          parameters. The expected shape for the tensor is :math:`(B, 3, 3)`.
        scale_factor (Union[float, torch.Tensor]): the scaling factor to be applied.

    Returns:
        torch.Tensor: The scaled camera matrix with shame shape as input :math:`(B, 3, 3)`.

    """
    K_scale = camera_matrix.clone()
    K_scale[..., 0, 0] *= scale_factor
    K_scale[..., 1, 1] *= scale_factor
    K_scale[..., 0, 2] *= scale_factor
    K_scale[..., 1, 2] *= scale_factor
    return K_scale


def projection_from_KRt(K: torch.Tensor, R: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    r"""Get the projection matrix P from K, R and t.

    This function estimate the projection matrix by solving the following equation: :math:`P = K * [R|t]`.

    Args:
       K (torch.Tensor): the camera matrix with the instrinsics with shape :math:`(B, 3, 3)`.
       R (torch.Tensor): The rotation matrix with shape :math:`(B, 3, 3)`.
       t (torch.Tensor): The translation vector with shape :math:`(B, 3, 1)`.

    Returns:
       torch.Tensor: The projection matrix P with shape :math:`(B, 4, 4)`.

    """
    assert K.shape[-2:] == (3, 3), K.shape
    assert R.shape[-2:] == (3, 3), R.shape
    assert t.shape[-2:] == (3, 1), t.shape
    assert len(K.shape) == len(R.shape) == len(t.shape)

    Rt: torch.Tensor = torch.cat([R, t], dim=-1)  # 3x4
    Rt_h = torch.nn.functional.pad(Rt, [0, 0, 0, 1], "constant", 0.)  # 4x4
    Rt_h[..., -1, -1] += 1.

    K_h: torch.Tensor = torch.nn.functional.pad(K, [0, 1, 0, 1], "constant", 0.)  # 4x4
    K_h[..., -1, -1] += 1.

    return K @ Rt


def depth(R: torch.Tensor, t: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
    r"""Returns the depth of a point transformed by a rigid transform.

    Args:
       R (torch.Tensor): The rotation matrix with shape :math:`(*, 3, 3)`.
       t (torch.Tensor): The translation vector with shape :math:`(*, 3, 1)`.
       X (torch.Tensor): The 3d points with shape :math:`(*, 3)`.

    Returns:
       torch.Tensor: The depth value per point with shape :math:`(*, 1)`.

    """
    X_tmp = R @ X.transpose(-2, -1)
    X_out = X_tmp[..., 2, :] + t[..., 2, :]
    return X_out


# adapted from:
# https://github.com/opencv/opencv_contrib/blob/master/modules/sfm/src/fundamental.cpp#L61
# https://github.com/mapillary/OpenSfM/blob/master/opensfm/multiview.py#L14


def _nullspace(A):
    '''Compute the null space of A.
    Return the smallest singular value and the corresponding vector.
    '''
    u, s, vh = torch.svd(A)
    return s[..., -1], vh[..., -1]


def projections_from_fundamental(F_mat: torch.Tensor) -> torch.Tensor:
    r"""Get the projection matrices from the Fundamenal Matrix.

    Args:
       F_mat (torch.Tensor): the fundamenal matrix with the shape :math:`(*, 3, 3)`.

    Returns:
        torch.Tensor: The projection matrices with shape :math:`(*, 4, 4, 2)`.

    """
    assert len(F_mat.shape) >= 2, F_mat.shape
    assert F_mat.shape[-2:] == (3, 3), F_mat.shape

    R1 = numeric.eye_like(3, F_mat)  # Bx3x3
    t1 = numeric.vec_like(3, F_mat)  # Bx3

    Ft_mat = F_mat.transpose(-2, -1)

    _, e2 = _nullspace(Ft_mat)

    R2 = numeric.cross_product_matrix(e2) @ F_mat  # Bx3x3
    t2 = e2[..., :, None]  # Bx3x1

    P1 = torch.cat([R1, t1], dim=-1)  # Bx3x4
    P2 = torch.cat([R2, t2], dim=-1)  # Bx3x4

    return torch.stack([P1, P2], dim=-1)
