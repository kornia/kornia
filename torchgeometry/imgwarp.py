import torch

from .utils import inverse
from .homography_warper import homography_warp
from .conversions import deg2rad


__all__ = [
    "warp_perspective",
    "get_rotation_matrix2d",
]


def center_transform(transform, height, width):
    assert len(transform.shape) == 3, transform.shape
    # move points to origin
    center_mat_origin = torch.unsqueeze(
        torch.eye(
            3,
            device=transform.device,
            dtype=transform.dtype),
        dim=0)
    center_mat_origin[..., 0, 2] = float(width) / 2
    center_mat_origin[..., 1, 2] = float(height) / 2
    # move points from origin
    origin_mat_center = torch.unsqueeze(
        torch.eye(
            3,
            device=transform.device,
            dtype=transform.dtype),
        dim=0)
    origin_mat_center[..., 0, 2] = -float(width) / 2
    origin_mat_center[..., 1, 2] = -float(height) / 2
    return torch.matmul(center_mat_origin,
                        torch.matmul(transform, origin_mat_center))


def normalize_transform_to_pix(transform, height, width):
    assert len(transform.shape) == 3, transform.shape
    normal_trans_pix = torch.tensor([[
        [2. / (width - 1), 0., -1.],
        [0., 2. / (height - 1), -1.],
        [0., 0., 1.]]],
        device=transform.device, dtype=transform.dtype)  # 1x3x3
    pix_trans_normal = inverse(normal_trans_pix)         # 1x3x3
    return torch.matmul(normal_trans_pix,
                        torch.matmul(transform, pix_trans_normal))


def warp_perspective(src, M, dsize, flags='bilinear', border_mode=None,
                     border_value=0):
    r"""Applies a perspective transformation to an image.

    The function warpPerspective transforms the source image using
    the specified matrix:

    .. math::
        dst(x, y) = src \left(
        \frac{M_{11} x + M_{12} y + M_{33}}{M_{31} x + M_{32} y + M_{33}} ,
        \frac{M_{21} x + M_{22} y + M_{23}}{M_{31} x + M_{32} y + M_{33}}
        \right)

    Args:
        src (Tensor): input image.
        M (Tensor): transformation matrix.
        dsize (tuple): size of the output image (height, width).

    Returns:
        Tensor: the warped input image.

    Shape:
        - Input: :math:`(B, C, H, W)` and :math:`(B, 3, 3)`
        - Output: :math:`(B, C, H, W)`

    .. note::
       See a working example `here <https://github.com/arraiy/torchgeometry/
       blob/master/examples/warp_perspective.ipynb>`_.
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
    # center the transformation and normalize
    _, _, height, width = src.shape
    M_new = center_transform(M, height, width)
    M_new = normalize_transform_to_pix(M_new, height, width)
    # warp and return
    return homography_warp(src, M_new, dsize)


def get_rotation_matrix2d(center, angle, scale):
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
        - Input: :math:`(B, 2)`, :math:`(B, 1)` and :math:`(B, 1)`
        - Output: :math:`(B, 2, 3)`

    Example:
        >>> center = torch.zeros(1, 2)
        >>> scale = torch.ones(1, 1)
        >>> angle = 45. * torch.ones(1, 1)
        >>> M = tgm.get_rotation_matrix2d(center, angle, scale)
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
    if not (len(angle.shape) == 2 and angle.shape[1] == 1):
        raise ValueError("Input angle must be a Bx1 tensor. Got {}"
                         .format(angle.shape))
    if not (len(scale.shape) == 2 and scale.shape[1] == 1):
        raise ValueError("Input scale must be a Bx1 tensor. Got {}"
                         .format(scale.shape))
    if not (center.shape[0] == angle.shape[0] == scale.shape[0]):
        raise ValueError("Inputs must have same batch size dimension. Got {}"
                         .format(center.shape, angle.shape, scale.shape))
    # convert angle and apply scale
    angle_rad = deg2rad(angle)
    alpha = torch.cos(angle_rad) * scale
    beta = torch.sin(angle_rad) * scale

    # unpack the center to x, y coordinates
    x, y = torch.chunk(center, chunks=2, dim=1)

    # create output tensor
    batch_size, _ = center.shape
    M = torch.zeros(batch_size, 2, 3, device=center.device, dtype=center.dtype)
    M[..., 0, 0] = alpha
    M[..., 0, 1] = beta
    M[..., 0, 2] = (1. - alpha) * x - beta * y
    M[..., 1, 0] = -beta
    M[..., 1, 1] = alpha
    M[..., 1, 2] = beta * x + (1. - alpha) * y
    return M
