from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from kornia.feature.nms import non_maxima_suppression2d
from kornia.filters import spatial_gradient, gaussian_blur2d


class CornerHarris(nn.Module):
    r"""Computes the Harris corner detection.

    The response map is computed according the following formulation:

    .. math::
        R = det(M) - k \cdot trace(M)^2

    where:

    .. math::
        M = \sum_{(x,y) \in W}
        \begin{bmatrix}
            I^{2}_x & I_x I_y \\
            I_x I_y & I^{2}_y \\
        \end{bmatrix}

    and :math:`k` is an empirically determined constant
    :math:`k ∈ [ 0.04 , 0.06 ]`

    Args:
        k (torch.Tensor): the Harris detector free parameter.

    Return:
        torch.Tensor: the response map per channel.

    Shape:
      - Input: :math:`(B, C, H, W)`
      - Output: :math:`(B, C, H, W)`

    Examples:
        >>> input = torch.tensor([[[
            [0., 0., 0., 0., 0., 0., 0.],
            [0., 1., 1., 1., 1., 1., 0.],
            [0., 1., 1., 1., 1., 1., 0.],
            [0., 1., 1., 1., 1., 1., 0.],
            [0., 1., 1., 1., 1., 1., 0.],
            [0., 1., 1., 1., 1., 1., 0.],
            [0., 0., 0., 0., 0., 0., 0.],
        ]]])  # 1x1x7x7
        >>> # compute the response map
        >>> output = kornia.feature.CornerHarris()(input)
        tensor([[[[0., 0., 0., 0., 0., 0., 0.],
                  [0., 1., 0., 0., 0., 1., 0.],
                  [0., 0., 0., 0., 0., 0., 0.],
                  [0., 0., 0., 0., 0., 0., 0.],
                  [0., 0., 0., 0., 0., 0., 0.],
                  [0., 1., 0., 0., 0., 1., 0.],
                  [0., 0., 0., 0., 0., 0., 0.]]]])
    """

    def __init__(self, k: torch.Tensor) -> None:
        super(CornerHarris, self).__init__()
        self.k: torch.Tensor = k
        # TODO: add as signature parameter
        self.kernel_size: Tuple[int, int] = (3, 3)

    def forward(self, input: torch.Tensor) -> torch.Tensor:  # type: ignore
        if not torch.is_tensor(input):
            raise TypeError("Input type is not a torch.Tensor. Got {}"
                            .format(type(input)))
        if not len(input.shape) == 4:
            raise ValueError("Invalid input shape, we expect BxCxHxW. Got: {}"
                             .format(input.shape))
        # compute the first order gradients with sobel operator
        # TODO: implement support for kernel different than three
        gradients: torch.Tensor = spatial_gradient(input)
        dx: torch.Tensor = gradients[:, :, 0]
        dy: torch.Tensor = gradients[:, :, 1]

        # compute the structure tensor M elements
        def g(x):
            return gaussian_blur2d(x, (3, 3), (1., 1.))

        dx2: torch.Tensor = g(dx * dx)
        dy2: torch.Tensor = g(dy * dy)
        dxy: torch.Tensor = g(dx * dy)

        det_m: torch.Tensor = dx2 * dy2 - dxy * dxy
        trace_m: torch.Tensor = dx2 + dy2

        # compute the response map
        scores: torch.Tensor = det_m - self.k * trace_m ** 2

        # threshold
        # TODO: add as signature parameter ?
        scores = torch.clamp(scores, min=1e-6)

        # apply non maxima suppresion
        scores = non_maxima_suppression2d(scores, kernel_size=(3, 3))

        # normalize and return
        scores_max: torch.Tensor = F.adaptive_max_pool2d(scores, output_size=1)
        return scores / scores_max


# functiona api


def corner_harris(input: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    r"""Computes the Harris corner detection.

    See :class:`~kornia.feature.CornerHarris` for details.
    """
    return CornerHarris(k)(input)
