import torch


# Based on https://github.com/opencv/opencv/blob/master/modules/calib3d/src/distortion_model.hpp#L75
def tiltProjection(taux: torch.Tensor, tauy: torch.Tensor, inv: bool = False) -> torch.Tensor:
    r"""Estimate the tilt projection matrix or the inverse tilt projection matrix

    Args:
        taux (torch.Tensor): Rotation angle in radians around the :math:`x`-axis with shape :math:`(*, 1)`.
        tauy (torch.Tensor): Rotation angle in radians around the :math:`y`-axis with shape :math:`(*, 1)`.
        inv (bool): False to obtain the the tilt projection matrix. False for the inverse matrix

    Returns:
        torch.Tensor: Inverse tilt projection matrix with shape :math:`(*, 3, 3)`.
    """
    assert taux.dim() == tauy.dim()
    assert taux.numel() == tauy.numel()

    ndim = taux.dim()
    taux = taux.reshape(-1)
    tauy = tauy.reshape(-1)

    cTx = torch.cos(taux)
    sTx = torch.sin(taux)
    cTy = torch.cos(tauy)
    sTy = torch.sin(tauy)
    zero = torch.zeros_like(cTx)
    one = torch.ones_like(cTx)

    Rx = torch.stack([one, zero, zero, zero, cTx, sTx, zero, -sTx, cTx], -1).reshape(-1, 3, 3)
    Ry = torch.stack([cTy, zero, -sTy, zero, one, zero, sTy, zero, cTy], -1).reshape(-1, 3, 3)
    R = Ry @ Rx

    if inv:
        invR22 = 1 / R[..., 2, 2]
        invPz = torch.stack(
            [invR22, zero, R[..., 0, 2] * invR22,
            zero, invR22, R[..., 1, 2] * invR22,
            zero, zero, one], -1
        ).reshape(-1, 3, 3)

        invTilt = R.transpose(-1, -2) @ invPz
        if ndim == 0:
            invTilt = torch.squeeze(invTilt)

        return invTilt

    else:
        Pz = torch.stack(
            [R[..., 2, 2], zero, -R[..., 0, 2],
            zero, R[..., 2, 2], -R[..., 1, 2],
            zero, zero, one], -1
        ).reshape(-1, 3, 3)

        tilt = Pz @ R.transpose(-1, -2)
        if ndim == 0:
            tilt = torch.squeeze(tilt)

        return tilt


def distort_points(points: torch.Tensor, K: torch.Tensor, dist: torch.Tensor) -> torch.Tensor:
    r"""Distortion of a set of 2D points based on the lens distortion model.

    Radial :math:`(k_1, k_2, k_3, k_4, k_4, k_6)`,
    tangential :math:`(p_1, p_2)`, thin prism :math:`(s_1, s_2, s_3, s_4)`, and tilt :math:`(\tau_x, \tau_y)` distortion models are considered in this function.

    Args:
        points: Input image points with shape :math:`(*, N, 2)`.
        K: Intrinsic camera matrix with shape :math:`(*, 3, 3)`.
        dist: Distortion coefficients
            :math:`(k_1,k_2,p_1,p_2[,k_3[,k_4,k_5,k_6[,s_1,s_2,s_3,s_4[,\tau_x,\tau_y]]]])`. This is
            a vector with 4, 5, 8, 12 or 14 elements with shape :math:`(*, n)`

    Returns:
        Undistorted 2D points with shape :math:`(*, N, 2)`.
    """
    assert points.dim() >= 2 and points.shape[-1] == 2
    assert K.shape[-2:] == (3, 3)
    assert dist.shape[-1] in [4, 5, 8, 12, 14]

    if dist.shape[-1] < 14:
        dist = torch.nn.functional.pad(dist, [0, 14 - dist.shape[-1]])

    # Convert 2D points from pixels to normalized camera coordinates
    cx: torch.Tensor = K[..., 0:1, 2]  # princial point in x (Bx1)
    cy: torch.Tensor = K[..., 1:2, 2]  # princial point in y (Bx1)
    fx: torch.Tensor = K[..., 0:1, 0]  # focal in x (Bx1)
    fy: torch.Tensor = K[..., 1:2, 1]  # focal in y (Bx1)
    # This is equivalent to K^-1 [u,v,1]^T
    x: torch.Tensor = (points[..., 0] - cx) / fx  # (BxN - Bx1)/Bx1 -> BxN or (N,)
    y: torch.Tensor = (points[..., 1] - cy) / fy  # (BxN - Bx1)/Bx1 -> BxN or (N,)

    # Distort points
    r2 = x * x + y * y

    rad_poly = (1 + dist[..., 0:1] * r2 + dist[..., 1:2] * r2 * r2 + dist[..., 4:5] * r2 ** 3) / (
        1 + dist[..., 5:6] * r2 + dist[..., 6:7] * r2 * r2 + dist[..., 7:8] * r2 ** 3
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
        tilt = tiltProjection(dist[..., 12], dist[..., 13])

        # Transposed untilt points (instead of [x,y,1]^T, we obtain [x,y,1])
        pointsUntilt = torch.stack([xd, yd, torch.ones(xd.shape, device=xd.device, dtype=xd.dtype)], -1) @ tilt.transpose(-2, -1)
        xd = pointsUntilt[..., 0] / pointsUntilt[..., 2]
        yd = pointsUntilt[..., 1] / pointsUntilt[..., 2]

    # Covert points from normalized camera coordinates to pixel coordinates
    x = fx * xd + cx
    y = fy * yd + cy

    return torch.stack([x, y], -1)
