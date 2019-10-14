import torch
import torch.nn as nn


class TotalVariation(nn.Module):
    r"""Computes the Total Variation according to
    [1] https://en.wikipedia.org/wiki/Total_variation
    Shape:
        - Input: :math:`(N, C, H, W)` or :math:`(C, H, W)` where C = number of classes.
        - Target: :math:`(N,)` or :math:`()`
    Examples:
        >>> kornia.losses.total_variation(torch.ones(3,4,4)) # tensor(0.)
        >>> tv = kornia.losses.TotalVariation()
        >>> output = tv(torch.ones(2,3,4,4)) # tensor([0., 0.])
        >>> output.backward()
    """
    def __init__(self):
        super(TotalVariation, self).__init__()

    def forward(self, img):
        return total_variation(img)


def total_variation(img: torch.tensor) -> torch.tensor:
    r"""Function that computes Total Variation.

    See :class:`~kornia.losses.TotalVariation` for details.
    """
    if not torch.is_tensor(img):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(src1)}")
    img_shape = img.shape
    if len(img_shape) == 3:
        pixel_dif1 = img[:, 1:, :] - img[:, :-1, :]
        pixel_dif2 = img[:, :, 1:] - img[:, :, :-1]
        reduce_axes = [0,1,2]
    elif len(img_shape) == 4:
        pixel_dif1 = img[:, :, 1:, :] - img[:, :, :-1, :]
        pixel_dif2 = img[:, :, :, 1:] - img[:, :, :, :-1]
        reduce_axes = [1,2,3]
    else:
        raise ValueError("Expected input tensor to be of rank 3 or 4, but got " + str(len(img_shape)))

    return pixel_dif1.abs().sum(axis=reduce_axes) + pixel_dif2.abs().sum(axis=reduce_axes)
