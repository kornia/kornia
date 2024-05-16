from typing import Optional

import torch

from kornia.core import cos, ones_like, sin, stack, zeros_like


# Based on https://github.com/opencv/opencv/blob/master/modules/calib3d/src/distortion_model.hpp#L75
def tilt_projection(taux: torch.Tensor, tauy: torch.Tensor, return_inverse: bool = False) -> torch.Tensor:
    r"""Estimate the tilt projection matrix or the inverse tilt projection matrix.

    Args:
        taux: Rotation angle in radians around the :math:`x`-axis with shape :math:`(*, 1)`.
        tauy: Rotation angle in radians around the :math:`y`-axis with shape :math:`(*, 1)`.
        return_inverse: False to obtain the tilt projection matrix. True for the inverse matrix.

    Returns:
        torch.Tensor: Inverse tilt projection matrix with shape :math:`(*, 3, 3)`.
    """
    if taux.shape != tauy.shape:
        raise ValueError(f"Shape of taux {taux.shape} and tauy {tauy.shape} do not match.")

    ndim: int = taux.dim()
    taux = taux.reshape(-1)
    tauy = tauy.reshape(-1)

    cTx = cos(taux)
    sTx = sin(taux)
    cTy = cos(tauy)
    sTy = sin(tauy)
    zero = zeros_like(cTx)
    one = ones_like(cTx)

    Rx = stack([one, zero, zero, zero, cTx, sTx, zero, -sTx, cTx], -1).reshape(-1, 3, 3)
    Ry = stack([cTy, zero, -sTy, zero, one, zero, sTy, zero, cTy], -1).reshape(-1, 3, 3)
    R = Ry @ Rx

    if return_inverse:
        invR22 = 1 / R[..., 2, 2]
        invPz = stack(
            [invR22, zero, R[..., 0, 2] * invR22, zero, invR22, R[..., 1, 2] * invR22, zero, zero, one], -1
        ).reshape(-1, 3, 3)

        inv_tilt = R.transpose(-1, -2) @ invPz
        if ndim == 0:
            inv_tilt = torch.squeeze(inv_tilt)

        return inv_tilt

    Pz = stack([R[..., 2, 2], zero, -R[..., 0, 2], zero, R[..., 2, 2], -R[..., 1, 2], zero, zero, one], -1).reshape(
        -1, 3, 3
    )

    tilt = Pz @ R.transpose(-1, -2)
    if ndim == 0:
        tilt = torch.squeeze(tilt)

    return tilt


def distort_points(
    points: torch.Tensor, K: torch.Tensor, dist: torch.Tensor, new_K: Optional[torch.Tensor] = None
) -> torch.Tensor:
    r"""Distortion of a set of 2D points based on the lens distortion model.

    Radial :math:`(k_1, k_2, k_3, k_4, k_4, k_6)`,
    tangential :math:`(p_1, p_2)`, thin prism :math:`(s_1, s_2, s_3, s_4)`, and tilt :math:`(\tau_x, \tau_y)`
    distortion models are considered in this function.

    Args:
        points: Input image points with shape :math:`(*, N, 2)`.
        K: Intrinsic camera matrix with shape :math:`(*, 3, 3)`.
        dist: Distortion coefficients
            :math:`(k_1,k_2,p_1,p_2[,k_3[,k_4,k_5,k_6[,s_1,s_2,s_3,s_4[,\tau_x,\tau_y]]]])`. This is
            a vector with 4, 5, 8, 12 or 14 elements with shape :math:`(*, n)`.
        new_K: Intrinsic camera matrix of the distorted image. By default, it is the same as K but you may additionally
            scale and shift the result by using a different matrix. Shape: :math:`(*, 3, 3)`. Default: None.

    Returns:
        Undistorted 2D points with shape :math:`(*, N, 2)`.

    Example:
        >>> points = torch.rand(1, 1, 2)
        >>> K = torch.eye(3)[None]
        >>> dist_coeff = torch.rand(1, 4)
        >>> points_dist = distort_points(points, K, dist_coeff)
    """
    if points.dim() < 2 and points.shape[-1] != 2:
        raise ValueError(f"points shape is invalid. Got {points.shape}.")

    if K.shape[-2:] != (3, 3):
        raise ValueError(f"K matrix shape is invalid. Got {K.shape}.")

    if new_K is None:
        new_K = K
    elif new_K.shape[-2:] != (3, 3):
        raise ValueError(f"new_K matrix shape is invalid. Got {new_K.shape}.")

    if dist.shape[-1] not in [4, 5, 8, 12, 14]:
        raise ValueError(f"Invalid number of distortion coefficients. Got {dist.shape[-1]}")

    # Adding zeros to obtain vector with 14 coeffs.
    if dist.shape[-1] < 14:
        dist = torch.nn.functional.pad(dist, [0, 14 - dist.shape[-1]])

    # Convert 2D points from pixels to normalized camera coordinates
    new_cx: torch.Tensor = new_K[..., 0:1, 2]  # princial point in x (Bx1)
    new_cy: torch.Tensor = new_K[..., 1:2, 2]  # princial point in y (Bx1)
    new_fx: torch.Tensor = new_K[..., 0:1, 0]  # focal in x (Bx1)
    new_fy: torch.Tensor = new_K[..., 1:2, 1]  # focal in y (Bx1)

    # This is equivalent to K^-1 [u,v,1]^T
    x: torch.Tensor = (points[..., 0] - new_cx) / new_fx  # (BxN - Bx1)/Bx1 -> BxN or (N,)
    y: torch.Tensor = (points[..., 1] - new_cy) / new_fy  # (BxN - Bx1)/Bx1 -> BxN or (N,)

    # Distort points
    r2 = x * x + y * y

    rad_poly = (1 + dist[..., 0:1] * r2 + dist[..., 1:2] * r2 * r2 + dist[..., 4:5] * r2**3) / (
        1 + dist[..., 5:6] * r2 + dist[..., 6:7] * r2 * r2 + dist[..., 7:8] * r2**3
    )
    xd = (
        x * rad_poly
        + 2 * dist[..., 2:3] * x * y
        + dist[..., 3:4] * (r2 + 2 * x * x)
        + dist[..., 8:9] * r2
        + dist[..., 9:10] * r2 * r2
    )
    yd = (
        y * rad_poly
        + dist[..., 2:3] * (r2 + 2 * y * y)
        + 2 * dist[..., 3:4] * x * y
        + dist[..., 10:11] * r2
        + dist[..., 11:12] * r2 * r2
    )

    # Compensate for tilt distortion
    if torch.any(dist[..., 12] != 0) or torch.any(dist[..., 13] != 0):
        tilt = tilt_projection(dist[..., 12], dist[..., 13])

        # Transposed untilt points (instead of [x,y,1]^T, we obtain [x,y,1])
        points_untilt = stack([xd, yd, ones_like(xd)], -1) @ tilt.transpose(-2, -1)
        xd = points_untilt[..., 0] / points_untilt[..., 2]
        yd = points_untilt[..., 1] / points_untilt[..., 2]

    # Convert points from normalized camera coordinates to pixel coordinates
    cx: torch.Tensor = K[..., 0:1, 2]  # princial point in x (Bx1)
    cy: torch.Tensor = K[..., 1:2, 2]  # princial point in y (Bx1)
    fx: torch.Tensor = K[..., 0:1, 0]  # focal in x (Bx1)
    fy: torch.Tensor = K[..., 1:2, 1]  # focal in y (Bx1)

    x = fx * xd + cx
    y = fy * yd + cy

    return stack([x, y], -1)
