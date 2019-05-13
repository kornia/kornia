from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchgeometry.image import spatial_gradient_2nd_order, gaussian_blur
from torchgeometry.feature.nms import non_maxima_suppression2d 


class Hessian(nn.Module):
    r"""Computes the absolute of determinant of the Hessian matrix.

    The response map is computed according the following formulation:

    .. math::
        R = det(H) 

    where:

    .. math::
        M = \sum_{(x,y) \in W}
        \begin{bmatrix}
            I_{xx} & I_{xy} \\
            I_{xy} & I_yy \\
        \end{bmatrix}

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
        >>> output = tgm.feature.Hessian()(input)
		tensor([[[
			[0., 0., 0., 0., 0., 0., 0.],
        	[0., 0., 0., 0., 0., 0., 0.],
			[0., 0., 1., 0., 1., 0., 0.],
			[0., 0., 0., 0., 0., 0., 0.],
			[0., 0., 1., 0., 1., 0., 0.],
			[0., 0., 0., 0., 0., 0., 0.],
			[0., 0., 0., 0., 0., 0., 0.]]]])
    """

    def __init__(self) -> None:
        super(Hessian, self).__init__()

    def forward(self, input: torch.Tensor, scale: float = 1.0) -> torch.Tensor:  # type: ignore
        if not torch.is_tensor(input):
            raise TypeError("Input type is not a torch.Tensor. Got {}"
                            .format(type(input)))
        if not len(input.shape) == 4:
            raise ValueError("Invalid input shape, we expect BxCxHxW. Got: {}"
                             .format(input.shape))
        # compute the second order gradients with sobel operator
        # TODO: implement support for kernel different than three
        # TODO: implement direct 2nd order kernel instead of 2x 1st order
        gradients: torch.Tensor = spatial_gradient_2nd_order(input)
        gxx: torch.Tensor = gradients[:, :, 0]
        gxy: torch.Tensor = gradients[:, :, 1]
        gyy: torch.Tensor = gradients[:, :, 3]
        
        # compute the response map
        scores: torch.Tensor = torch.abs(gxx * gyy - gxy**2) * (scale**4)

        # threshold
        scores = torch.clamp(scores, min=1e-6)

        # apply non maxima suppresion
        scores = non_maxima_suppression2d(scores, kernel_size=(3, 3))

        # normalize and return
        scores_max: torch.Tensor = F.adaptive_max_pool2d(scores, output_size=1)
        return scores / scores_max


# functional api


def hessian(input: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    r"""Computes the first order image derivative in both x and y directions
        using a Sobel operator.

    See :class:`~torchgeometry.feature.CornerHarris` for details.
    """
    return Hessian()(input, scale)

