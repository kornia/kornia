import torch

from .utils import inverse
from .homography_warper import homography_warp


__all__ = [
    "warp_perspective",
    "get_perspective_transform",
]


def center_transform(transform, height, width):
    assert len(transform.shape) == 3, transform.shape
    # move points from origin
    center_mat_origin = torch.unsqueeze(
        torch.eye(
            3,
            device=transform.device,
            dtype=transform.dtype),
        dim=0)
    center_mat_origin[..., 0, 2] = float(width) / 2
    center_mat_origin[..., 1, 2] = float(height) / 2
    # move points to origin
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
    """Applies a perspective transformation to an image.

    The function warpPerspective transforms the source image using
    the specified matrix:

    .. math::
        dst(x, y) = src \left(
        \\frac{M_{11} x + M_{12} y + M_{33}}{M_{31} x + M_{32} y + M_{33}} ,
        \\frac{M_{21} x + M_{22} y + M_{23}}{M_{31} x + M_{32} y + M_{33}}
        \\right)

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


def get_perspective_transform(src, dst):
    """Calculates a perspective transform from four pairs of the corresponding
    points.

    The function calculates the matrix of a perspective transform so that:

    .. math ::

        todo

    Args:
        src (Tensor): coordinates of quadrangle vertices in the source image.
        dst (Tensor): coordinates of the corresponding quadrangle vertices in
        the destination image.

    Returns:
        Tensor: the perspective transformation.

    Shape:
        - Input: :math:`(B, 4, 2)` and :math:`(B, 4, 2)`
        - Output: :math:`(B, 3, 3)`
    """
    if not torch.is_tensor(src):
        raise TypeError("Input type is not a torch.Tensor. Got {}"
                        .format(type(src)))
    if not torch.is_tensor(dst):
        raise TypeError("Input type is not a torch.Tensor. Got {}"
                        .format(type(dst)))
    if not src.shape[-2:] == (4, 2):
        raise ValueError("Inputs must be a Bx4x2 tensor. Got {}"
                         .format(src.shape))
    if not src.shape == dst.shape:
        raise ValueError("Inputs must have the same shape. Got {}"
                         .format(dst.shape))
    def ax(p, q):
        ones = torch.ones_like(p)[..., 0:1]
        zeros = torch.zeros_like(p)[..., 0:1]
        return torch.cat(
            [ p[:, 0:1], p[:, 1:2], ones, zeros, zeros, zeros,
             -p[:, 0:1] * q[:, 0:1], -p[:, 1:2] * q[:, 0:1] ], dim=1)

    def ay(p, q):
        ones = torch.ones_like(p)[..., 0:1]
        zeros = torch.zeros_like(p)[..., 0:1]
        return torch.cat(
            [ zeros, zeros, zeros, p[:, 0:1], p[:, 1:2], ones,
              -p[:, 0:1] * q[:, 1:2], -p[:, 1:2] * q[:, 1:2] ], dim=1)
    # we build matrix A by using only 4 point correspondence. The linear
    # system is solved with the least square method, so here 
    # we could even pass more correspondence 
    p = []
    p.append(ax(src[:, 0], dst[:, 0]))
    p.append(ay(src[:, 0], dst[:, 0]))

    p.append(ax(src[:, 1], dst[:, 1]))
    p.append(ay(src[:, 1], dst[:, 1]))

    p.append(ax(src[:, 2], dst[:, 2]))
    p.append(ay(src[:, 2], dst[:, 2]))

    p.append(ax(src[:, 3], dst[:, 3]))
    p.append(ay(src[:, 3], dst[:, 3]))

    # A is Bx8x8
    A = torch.stack(p, dim=1)

    # b is a Bx8x1
    b = torch.stack([
        dst[:, 0:1, 0], dst[:, 0:1, 1],
        dst[:, 1:2, 0], dst[:, 1:2, 1],
        dst[:, 2:3, 0], dst[:, 2:3, 1],
        dst[:, 3:4, 0], dst[:, 3:4, 1],
    ], dim=1)

    # solve the system Ax = b
    X, LU = torch.gesv(b, A)

    # create variable to return
    batch_size = src.shape[0]
    M = torch.ones(batch_size, 9, device=src.device, dtype=src.dtype)
    M[..., :8] = torch.squeeze(X, dim=-1)
    return M.view(-1, 3, 3)  # Bx3x3

