from __future__ import annotations

from typing import Optional

import torch

from kornia.core import stack
from kornia.core.check import KORNIA_CHECK_SHAPE
from kornia.geometry.linalg import transform_points
from kornia.geometry.transform import remap
from kornia.utils import create_meshgrid

from .distort import distort_points, tilt_projection


# Based on https://github.com/opencv/opencv/blob/master/modules/calib3d/src/undistort.dispatch.cpp#L384
def undistort_points(
    points: torch.Tensor, K: torch.Tensor, dist: torch.Tensor, new_K: Optional[torch.Tensor] = None, num_iters: int = 5
) -> torch.Tensor:
    r"""Compensate for lens distortion a set of 2D image points.

    Radial :math:`(k_1, k_2, k_3, k_4, k_5, k_6)`,
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
        num_iters: Number of undistortion iterations. Default: 5.
    Returns:
        Undistorted 2D points with shape :math:`(*, N, 2)`.

    Example:
        >>> _ = torch.manual_seed(0)
        >>> x = torch.rand(1, 4, 2)
        >>> K = torch.eye(3)[None]
        >>> dist = torch.rand(1, 4)
        >>> undistort_points(x, K, dist)
        tensor([[[-0.1513, -0.1165],
                 [ 0.0711,  0.1100],
                 [-0.0697,  0.0228],
                 [-0.1843, -0.1606]]])
    """
    KORNIA_CHECK_SHAPE(points, ["*", "N", "2"])
    KORNIA_CHECK_SHAPE(K, ["*", "3", "3"])

    if points.dim() < 2 and points.shape[-1] != 2:
        raise ValueError(f"points shape is invalid. Got {points.shape}.")

    if new_K is None:
        new_K = K
    else:
        KORNIA_CHECK_SHAPE(new_K, ["*", "3", "3"])

    if dist.shape[-1] not in [4, 5, 8, 12, 14]:
        raise ValueError(f"Invalid number of distortion coefficients. Got {dist.shape[-1]}")

    # Adding zeros to obtain vector with 14 coeffs.
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
        inv_tilt = tilt_projection(dist[..., 12], dist[..., 13], True)

        # Transposed untilt points (instead of [x,y,1]^T, we obtain [x,y,1])
        x, y = transform_points(inv_tilt, stack([x, y], dim=-1)).unbind(-1)

    # Iteratively undistort points
    x0, y0 = x, y
    for _ in range(num_iters):
        r2 = x * x + y * y

        inv_rad_poly = (1 + dist[..., 5:6] * r2 + dist[..., 6:7] * r2 * r2 + dist[..., 7:8] * r2**3) / (
            1 + dist[..., 0:1] * r2 + dist[..., 1:2] * r2 * r2 + dist[..., 4:5] * r2**3
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
    new_cx: torch.Tensor = new_K[..., 0:1, 2]  # princial point in x (Bx1)
    new_cy: torch.Tensor = new_K[..., 1:2, 2]  # princial point in y (Bx1)
    new_fx: torch.Tensor = new_K[..., 0:1, 0]  # focal in x (Bx1)
    new_fy: torch.Tensor = new_K[..., 1:2, 1]  # focal in y (Bx1)
    x = new_fx * x + new_cx
    y = new_fy * y + new_cy
    return stack([x, y], -1)


# Based on https://github.com/opencv/opencv/blob/master/modules/calib3d/src/undistort.dispatch.cpp#L287
def undistort_image(image: torch.Tensor, K: torch.Tensor, dist: torch.Tensor) -> torch.Tensor:
    r"""Compensate an image for lens distortion.

    Radial :math:`(k_1, k_2, k_3, k_4, k_4, k_6)`,
    tangential :math:`(p_1, p_2)`, thin prism :math:`(s_1, s_2, s_3, s_4)`, and tilt :math:`(\tau_x, \tau_y)`
    distortion models are considered in this function.

    Args:
        image: Input image with shape :math:`(*, C, H, W)`.
        K: Intrinsic camera matrix with shape :math:`(*, 3, 3)`.
        dist: Distortion coefficients
            :math:`(k_1,k_2,p_1,p_2[,k_3[,k_4,k_5,k_6[,s_1,s_2,s_3,s_4[,\tau_x,\tau_y]]]])`. This is
            a vector with 4, 5, 8, 12 or 14 elements with shape :math:`(*, n)`.

    Returns:
        Undistorted image with shape :math:`(*, C, H, W)`.

    Example:
        >>> img = torch.rand(1, 3, 5, 5)
        >>> K = torch.eye(3)[None]
        >>> dist_coeff = torch.rand(1, 4)
        >>> out = undistort_image(img, K, dist_coeff)
        >>> out.shape
        torch.Size([1, 3, 5, 5])
    """
    if len(image.shape) < 3:
        raise ValueError(f"Image shape is invalid. Got: {image.shape}.")

    if K.shape[-2:] != (3, 3):
        raise ValueError(f"K matrix shape is invalid. Got {K.shape}.")

    if dist.shape[-1] not in [4, 5, 8, 12, 14]:
        raise ValueError(f"Invalid number of distortion coefficients. Got {dist.shape[-1]}.")

    if not image.is_floating_point():
        raise ValueError(f"Invalid input image data type. Input should be float. Got {image.dtype}.")

    if image.shape[:-3] != K.shape[:-2] or image.shape[:-3] != dist.shape[:-1]:
        # Input with image shape (1, C, H, W), K shape (3, 3), dist shape (4)
        # allowed to avoid a breaking change.
        if not all((image.shape[:-3] == (1,), K.shape[:-2] == (), dist.shape[:-1] == ())):
            raise ValueError(
                "Input shape is invalid. Input batch dimensions should match. "
                f"Got {image.shape[:-3]}, {K.shape[:-2]}, {dist.shape[:-1]}."
            )

    channels, rows, cols = image.shape[-3:]
    B = image.numel() // (channels * rows * cols)

    # Create point coordinates for each pixel of the image
    xy_grid: torch.Tensor = create_meshgrid(rows, cols, False, image.device, image.dtype)
    pts = xy_grid.reshape(-1, 2)  # (rows*cols)x2 matrix of pixel coordinates

    # Distort points and define maps
    ptsd: torch.Tensor = distort_points(pts, K, dist)  # Bx(rows*cols)x2
    mapx: torch.Tensor = ptsd[..., 0].reshape(B, rows, cols)  # B x rows x cols, float
    mapy: torch.Tensor = ptsd[..., 1].reshape(B, rows, cols)  # B x rows x cols, float

    # Remap image to undistort
    out = remap(image.reshape(B, channels, rows, cols), mapx, mapy, align_corners=True)

    return out.view_as(image)
