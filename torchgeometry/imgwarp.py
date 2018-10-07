import torch

from .homography_warper import homography_warp


__all__ = [
    "warp_perspective",
]


def warp_perspective(src, M, dsize, flags='bilinear', border_mode=None,
                     border_value=0):
    """Applies a perspective transformation to an image.

    The function warpPerspective transforms the source image using the specified matrix:

    .. math::
        dst(x, y) = src \left(\\frac{M_{11} x + M_{12} y + M_{33}}{M_{31} x + M_{32} y + M_{33}} , \\frac{M_{21} x + M_{22} y + M_{23}}{M_{31} x + M_{32} y + M_{33}}\\right)

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
       See a working example `here <../../../examples/warp_perspective.ipynb>`_.
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
    # warp and return
    return homography_warp(src, M, dsize)
