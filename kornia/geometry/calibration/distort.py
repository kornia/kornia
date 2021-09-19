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
