import torch
import torch.nn as nn


class TotalVariation(nn.Module):
    r"""Computes the Total Variation according to [1].

    Shape:
        - Input: :math:`(N, C, H, W)` or :math:`(C, H, W)` where C = number of classes.
        - Output: :math:`(N,)` or :math:`()`

    Examples:
        >>> total_variation(torch.ones(3, 4, 4))
        tensor(0.)
        >>> tv = TotalVariation()
        >>> output = tv(torch.ones((2, 3, 4, 4), requires_grad=True))
        >>> output.data
        tensor([0., 0.])
        >>> output.sum().backward()  # grad can be implicitly created only for scalar outputs

    Reference:
        [1] https://en.wikipedia.org/wiki/Total_variation
    """

    def __init__(self) -> None:
        super(TotalVariation, self).__init__()

    def forward(  # type: ignore
            self, img) -> torch.Tensor:
        return total_variation(img)


def total_variation(img: torch.Tensor) -> torch.Tensor:
    r"""Function that computes Total Variation.

    See :class:`~kornia.losses.TotalVariation` for details.
    """
    if not torch.is_tensor(img):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(img)}")
    img_shape = img.shape
    if len(img_shape) == 3 or len(img_shape) == 4:
        pixel_dif1 = img[..., 1:, :] - img[..., :-1, :]
        pixel_dif2 = img[..., :, 1:] - img[..., :, :-1]
        reduce_axes = (-3, -2, -1)
    else:
        raise ValueError("Expected input tensor to be of ndim 3 or 4, but got " + str(len(img_shape)))

    return pixel_dif1.abs().sum(dim=reduce_axes) + pixel_dif2.abs().sum(dim=reduce_axes)
