import torch

from .utils import inverse
from .homography_warper import homography_warp


__all__ = [
    "warp_perspective",
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
