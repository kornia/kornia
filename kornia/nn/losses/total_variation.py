import torch
import torch.nn as nn

import kornia


class TotalVariation(nn.Module):
    r"""Computes the Total Variation according to [1].

    Shape:
        - Input: :math:`(N, C, H, W)` or :math:`(C, H, W)` where C = number of classes.
        - Output: :math:`(N,)` or :math:`()`

    Examples:
        >>> kornia.losses.total_variation(torch.ones(3,4,4)) # tensor(0.)
        >>> tv = kornia.losses.TotalVariation()
        >>> output = tv(torch.ones(2,3,4,4)) # tensor([0., 0.])
        >>> output.backward()

    Reference:
        [1] https://en.wikipedia.org/wiki/Total_variation
    """

    def __init__(self) -> None:
        super(TotalVariation, self).__init__()

    def forward(  # type: ignore
            self, img) -> torch.Tensor:
        return kornia.losses.total_variation(img)
