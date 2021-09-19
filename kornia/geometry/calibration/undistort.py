from numpy import imag
import torch
from kornia.geometry.calibration.distort import tiltProjection, distort_points
from kornia.geometry.transform.imgwarp import remap


# Based on https://github.com/opencv/opencv/blob/master/modules/calib3d/src/undistort.dispatch.cpp#L384
def undistort_points(points: torch.Tensor, K: torch.Tensor, dist: torch.Tensor) -> torch.Tensor:
    r"""Compensate for lens distortion a set of 2D image points.

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
    x: torch.Tensor = (points[..., 0] - cx) / fx  # (BxN - Bx1)/Bx1 -> BxN
    y: torch.Tensor = (points[..., 1] - cy) / fy  # (BxN - Bx1)/Bx1 -> BxN

    # Compensate for tilt distortion
    if torch.any(dist[..., 12] != 0) or torch.any(dist[..., 13] != 0):
        invTilt = tiltProjection(dist[..., 12], dist[..., 13], True)

        # Transposed untilt points (instead of [x,y,1]^T, we obtain [x,y,1])
        pointsUntilt = torch.stack([x, y, torch.ones(x.shape, device=x.device, dtype=x.dtype)], -1) @ invTilt.transpose(-2, -1)
        x = pointsUntilt[..., 0] / pointsUntilt[..., 2]
        y = pointsUntilt[..., 1] / pointsUntilt[..., 2]

    # Iteratively undistort points
    x0, y0 = x, y
    for _ in range(5):
        r2 = x * x + y * y

        inv_rad_poly = (1 + dist[..., 5:6] * r2 + dist[..., 6:7] * r2 * r2 + dist[..., 7:8] * r2 ** 3) / (
            1 + dist[..., 0:1] * r2 + dist[..., 1:2] * r2 * r2 + dist[..., 4:5] * r2 ** 3
        )
        deltaX = (
            2 * dist[..., 2:3] * x * y
            + dist[..., 3:4] * (r2 + 2 * x * x)
            + dist[..., 8:9] * r2
            + dist[..., 9:10] * r2 * r2
        )
        deltaY = (
            dist[..., 2:3] * (r2 + 2 * y * y)
            + 2 * dist[..., 3:4] * x * y
            + dist[..., 10:11] * r2
            + dist[..., 11:12] * r2 * r2
        )

        x = (x0 - deltaX) * inv_rad_poly
        y = (y0 - deltaY) * inv_rad_poly

    # Convert points from normalized camera coordinates to pixel coordinates
    x = fx * x + cx
    y = fy * y + cy

    return torch.stack([x, y], -1)


# Based on https://github.com/opencv/opencv/blob/master/modules/calib3d/src/undistort.dispatch.cpp#L287
def undistort_image(image: torch.Tensor, K: torch.Tensor, dist: torch.Tensor) -> torch.Tensor:
    r"""Compensate an image for lens distortion.

    Radial :math:`(k_1, k_2, k_3, k_4, k_4, k_6)`,
    tangential :math:`(p_1, p_2)`, thin prism :math:`(s_1, s_2, s_3, s_4)`, and tilt :math:`(\tau_x, \tau_y)` distortion models are considered in this function.

    Args:
        image: Input image with shape :math:`(*, C, H, W)`.
        K: Intrinsic camera matrix with shape :math:`(*, 3, 3)`.
        dist: Distortion coefficients
            :math:`(k_1,k_2,p_1,p_2[,k_3[,k_4,k_5,k_6[,s_1,s_2,s_3,s_4[,\tau_x,\tau_y]]]])`. This is
            a vector with 4, 5, 8, 12 or 14 elements with shape :math:`(*, n)`

    Returns:
        Undistorted image with shape :math:`(*, C, H, W)`.
    """
    assert image.dim() >= 2
    assert K.shape[-2:] == (3, 3)
    assert dist.shape[-1] in [4, 5, 8, 12, 14]

    B, _, rows, cols = image.shape
    if image.dtype != torch.float:
        image = image.float()

    # Create point coordinates for each pixel of the image
    x, y = torch.meshgrid(torch.arange(cols), torch.arange(rows))
    pts: torch.Tensor = torch.cat([x.T.float().reshape(-1,1), y.T.reshape(-1,1)], 1) # (rows*cols)x2

    # Distort points and define maps
    ptsd: torch.Tensor = distort_points(pts, K, dist) # Bx(rows*cols)x2
    mapx: torch.Tensor = ptsd[..., 0].reshape(B, rows, cols) # B x rows x cols, float
    mapy: torch.Tensor = ptsd[..., 1].reshape(B, rows, cols) # B x rows x cols, float

    # Remap image to undistort
    out = remap(image, mapx, mapy, align_corners=True)
    out = torch.round(torch.clamp(out, 0, 255)).to(torch.uint8)

    return out
