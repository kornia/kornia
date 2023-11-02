from __future__ import annotations

from kornia.core import Module, Tensor
from kornia.core.check import KORNIA_CHECK, KORNIA_CHECK_SHAPE


def total_variation(img: Tensor, reduction: str = "sum") -> Tensor:
    r"""Function that computes Total Variation according to [1].

    Args:
        img: the input image with shape :math:`(*, H, W)`.
        reduction : Specifies the reduction to apply to the output: ``'mean'`` | ``'sum'``.
         ``'mean'``: the sum of the output will be divided by the number of elements
         in the output, ``'sum'``: the output will be summed.

    Return:
         a tensor with shape :math:`(*,)`.

    Examples:
        >>> total_variation(torch.ones(4, 4))
        tensor(0.)
        >>> total_variation(torch.ones(2, 5, 3, 4, 4)).shape
        torch.Size([2, 5, 3])

    .. note::
       See a working example `here <https://kornia.github.io/tutorials/nbs/total_variation_denoising.html>`__.
       Total Variation is formulated with summation, however this is not resolution invariant.
       Thus, `reduction='mean'` was added as an optional reduction method.

    Reference:
        [1] https://en.wikipedia.org/wiki/Total_variation
    """
    # TODO: here torchscript doesn't like KORNIA_CHECK_TYPE
    if not isinstance(img, Tensor):
        raise TypeError(f"Not a Tensor type. Got: {type(img)}")

    KORNIA_CHECK_SHAPE(img, ["*", "H", "W"])
    KORNIA_CHECK(reduction in ("mean", "sum"), f"Expected reduction to be one of 'mean'/'sum', but got '{reduction}'.")

    pixel_dif1 = img[..., 1:, :] - img[..., :-1, :]
    pixel_dif2 = img[..., :, 1:] - img[..., :, :-1]

    res1 = pixel_dif1.abs()
    res2 = pixel_dif2.abs()

    reduce_axes = (-2, -1)
    if reduction == "mean":
        if img.is_floating_point():
            res1 = res1.to(img).mean(dim=reduce_axes)
            res2 = res2.to(img).mean(dim=reduce_axes)
        else:
            res1 = res1.float().mean(dim=reduce_axes)
            res2 = res2.float().mean(dim=reduce_axes)
    elif reduction == "sum":
        res1 = res1.sum(dim=reduce_axes)
        res2 = res2.sum(dim=reduce_axes)
    else:
        raise NotImplementedError("Invalid reduction option.")

    return res1 + res2


class TotalVariation(Module):
    r"""Compute the Total Variation according to [1].

    Shape:
        - Input: :math:`(*, H, W)`.
        - Output: :math:`(*,)`.

    Examples:
        >>> tv = TotalVariation()
        >>> output = tv(torch.ones((2, 3, 4, 4), requires_grad=True))
        >>> output.data
        tensor([[0., 0., 0.],
                [0., 0., 0.]])
        >>> output.sum().backward()  # grad can be implicitly created only for scalar outputs

    Reference:
        [1] https://en.wikipedia.org/wiki/Total_variation
    """

    def forward(self, img: Tensor) -> Tensor:
        return total_variation(img)
