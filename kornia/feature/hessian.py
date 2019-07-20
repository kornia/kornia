from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from kornia.feature.nms import non_maxima_suppression2d
from kornia.filters import spatial_gradient, gaussian_blur2d


class HessianResp(nn.Module):
    r"""Computes the Hessian extrema detection.

    The response map is computed according the following formulation:

    .. math::
        R = det(I_2)

    where:

    .. math::
        I = \sum_{(x,y) \in W}
        \begin{bmatrix}
            I_{xx} & I_{xy} \\
            I_{yx} & I_{yy} \\
        \end{bmatrix}

   Args:
        - k (torch.Tensor): the Harris detector free parameter.
        - do_blur (bool): perform Gaussian smoothing. Set to True, if use alone, False if on scale pyramid
        - nms (bool): perform hard non maxima supression
        - normalize (bool): if True, responce map is divided by max value
        
    Return:
        torch.Tensor: the response map per channel.

    Shape:
      - Input: :math:`(B, C, H, W)`
      - Output: :math:`(B, C, H, W)`

    Examples:
        >>> input = torch.tensor([[[
            [0., 0., 0., 0., 0., 0., 0.],
            [0., 1., 1., 1., 0., 0., 0.],
            [0., 1., 1., 1., 0., 0., 0.],
            [0., 1., 1., 1., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0.],
        ]]])  # 1x1x7x7
        >>> # compute the response map
        >>> output = kornia.feature.HessianResp()(input)
        tensor([[[[0., 0., 0., 0., 0., 0., 0.],
                  [0., 0., 0., 0., 0., 0., 0.],
                  [0., 0., 1., 0., 0., 0., 0.],
                  [0., 0., 0., 0., 0., 0., 0.],
                  [0., 0., 0., 0., 0., 0., 0.],
                  [0., 0., 0., 0., 0., 0., 0.],
                  [0., 0., 0., 0., 0., 0., 0.]]]])
    """
    def __init__(self,
                 do_blur:bool = True,
                 nms:bool = True,
                 normalize:bool = True) -> None:
        super(HessianResp, self).__init__()
        self.nms = nms
        self.normalize = normalize
        self.do_blur = do_blur

        self.gx =  nn.Conv2d(1, 1, kernel_size=(1,3), bias = False)
        self.gx.weight.data = torch.tensor([[[[0.5, 0, -0.5]]]])

        self.gy =  nn.Conv2d(1, 1, kernel_size=(3,1), bias = False)
        self.gy.weight.data = torch.tensor([[[[0.5], [0], [-0.5]]]])

        self.gxx =  nn.Conv2d(1, 1, kernel_size=(1,3),bias = False)
        self.gxx.weight.data = torch.tensor([[[[1.0, -2.0, 1.0]]]])
        
        self.gyy =  nn.Conv2d(1, 1, kernel_size=(3,1), bias = False)
        self.gyy.weight.data = torch.tensor([[[[1.0], [-2.0], [1.0]]]])
        return
    def forward(self, input: torch.Tensor) -> torch.Tensor:  # type: ignore
        if not torch.is_tensor(input):
            raise TypeError("Input type is not a torch.Tensor. Got {}"
                            .format(type(input)))
        if not len(input.shape) == 4:
            raise ValueError("Invalid input shape, we expect BxCxHxW. Got: {}"
                             .format(input.shape))
        self.to(input.device).to(input.dtype)
        B,CH,H,W = input.size()
        x = input.view(B*CH,1,H,W)
        def blur(x):
            return gaussian_blur2d(x, (3, 3), (1., 1.))
        if self.do_blur: #if using alone
            xb  = blur(x)
        else:
            xb = x

        gxx = self.gxx(F.pad(xb, (1, 1, 0, 0), 'replicate'))
        gyy = self.gyy(F.pad(xb, (0, 0, 1, 1), 'replicate'))
        gxy = self.gy(F.pad(self.gx(F.pad(xb, (1, 1, 0, 0), 'replicate')),
                            (0, 0, 1, 1), 'replicate'))
        
        scores = torch.abs(gxx * gyy - gxy * gxy).view(B,CH,H,W)
        
        # apply non maxima suppresion
        if self.nms:
            scores = non_maxima_suppression2d(scores, kernel_size=(3, 3))

        # normalize and return
        if self.normalize:
            scores_max: torch.Tensor = F.adaptive_max_pool2d(scores, output_size=1)
            scores = scores / scores_max.clamp(1e-6)
        return scores
    def update_response(self, resp, sigmas):
        """
        Scale pyramid blurs the intensity -> responces of the higher levers
        are smaller. For scale nms, one needs to fix that. 
        """
        if not torch.is_tensor(resp):
            raise TypeError("Input type is not a torch.Tensor. Got {}"
                            .format(type(input)))
        if not len(resp.shape) == 4:
            raise ValueError("Invalid input shape, we expect BxCxHxW. Got: {}"
                             .format(input.shape))
        
        B,CH,H,W = resp.size()
        new_resp = resp * sigmas[0:1,:].view(1,CH,1,1).pow(4)
        return new_resp


# functiona api


def hessian(input: torch.Tensor,
                  do_blur:bool = True,
                  nms:bool = True,
                  normalize:bool = True) -> torch.Tensor:
    r"""Computes the hessian responce.

    See :class:`~kornia.feature.HessianResp` for details.
    """
    return HessianResp(do_blur, nms, normalize)(input)
