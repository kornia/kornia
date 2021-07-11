import torch
import torch.nn as nn


def total_variation(img: torch.Tensor) -> torch.Tensor:
    r"""Function that computes Total Variation according to [1].

    Args:
        img: the input image with shape :math:`(N, C, H, W)` or :math:`(C, H, W)`.

    Return:
         a scalar with the computer loss.

    Examples:
        >>> total_variation(torch.ones(3, 4, 4))
        tensor(0.)

    .. note::
       See a working example `here <https://kornia-tutorials.readthedocs.io/en/latest/
       total_variation_denoising.html>`__.

    Reference:
        [1] https://en.wikipedia.org/wiki/Total_variation
    """
    if not isinstance(img, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(img)}")

    if len(img.shape) < 3 or len(img.shape) > 4:
        raise ValueError(f"Expected input tensor to be of ndim 3 or 4, but got {len(img.shape)}.")

    pixel_dif1 = img[..., 1:, :] - img[..., :-1, :]
    pixel_dif2 = img[..., :, 1:] - img[..., :, :-1]

    reduce_axes = (-3, -2, -1)
    res1 = pixel_dif1.abs().sum(dim=reduce_axes)
    res2 = pixel_dif2.abs().sum(dim=reduce_axes)

    return res1 + res2


class TotalVariation(nn.Module):
    r"""Computes the Total Variation according to [1].

    Shape:
        - Input: :math:`(N, C, H, W)` or :math:`(C, H, W)`.
        - Output: :math:`(N,)` or scalar.

    Examples:
        >>> tv = TotalVariation()
        >>> output = tv(torch.ones((2, 3, 4, 4), requires_grad=True))
        >>> output.data
        tensor([0., 0.])
        >>> output.sum().backward()  # grad can be implicitly created only for scalar outputs

    Reference:
        [1] https://en.wikipedia.org/wiki/Total_variation
    """

    def forward(self, img) -> torch.Tensor:
        return total_variation(img)
