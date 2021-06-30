from typing import Optional, Union

import torch
import torch.nn as nn

from kornia.filters import gaussian_blur2d, spatial_gradient


def harris_response(
    input: torch.Tensor,
    k: Union[torch.Tensor, float] = 0.04,
    grads_mode: str = 'sobel',
    sigmas: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    r"""Computes the Harris cornerness function.

    Function does not do any normalization or nms. The response map is computed according the following formulation:

    .. math::
        R = max(0, det(M) - k \cdot trace(M)^2)

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
        input: input image with shape :math:`(B, C, H, W)`.
        k: the Harris detector free parameter.
        grads_mode: can be ``'sobel'`` for standalone use or ``'diff'`` for use on Gaussian pyramid.
        sigmas: coefficients to be multiplied by multichannel response. Should be shape of :math:`(B)`
          It is necessary for performing non-maxima-suppression across different scale pyramid levels.
          See `vlfeat <https://github.com/vlfeat/vlfeat/blob/master/vl/covdet.c#L874>`_.

    Return:
        the response map per channel with shape :math:`(B, C, H, W)`.

    Example:
        >>> input = torch.tensor([[[
        ...    [0., 0., 0., 0., 0., 0., 0.],
        ...    [0., 1., 1., 1., 1., 1., 0.],
        ...    [0., 1., 1., 1., 1., 1., 0.],
        ...    [0., 1., 1., 1., 1., 1., 0.],
        ...    [0., 1., 1., 1., 1., 1., 0.],
        ...    [0., 1., 1., 1., 1., 1., 0.],
        ...    [0., 0., 0., 0., 0., 0., 0.],
        ... ]]])  # 1x1x7x7
        >>> # compute the response map
        harris_response(input, 0.04)
        tensor([[[[0.0012, 0.0039, 0.0020, 0.0000, 0.0020, 0.0039, 0.0012],
                  [0.0039, 0.0065, 0.0040, 0.0000, 0.0040, 0.0065, 0.0039],
                  [0.0020, 0.0040, 0.0029, 0.0000, 0.0029, 0.0040, 0.0020],
                  [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                  [0.0020, 0.0040, 0.0029, 0.0000, 0.0029, 0.0040, 0.0020],
                  [0.0039, 0.0065, 0.0040, 0.0000, 0.0040, 0.0065, 0.0039],
                  [0.0012, 0.0039, 0.0020, 0.0000, 0.0020, 0.0039, 0.0012]]]])
    """
    # TODO: Recompute doctest
    if not isinstance(input, torch.Tensor):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(type(input)))

    if not len(input.shape) == 4:
        raise ValueError("Invalid input shape, we expect BxCxHxW. Got: {}".format(input.shape))

    if sigmas is not None:
        if not isinstance(sigmas, torch.Tensor):
            raise TypeError("sigmas type is not a torch.Tensor. Got {}".format(type(sigmas)))
        if (not len(sigmas.shape) == 1) or (sigmas.size(0) != input.size(0)):
            raise ValueError("Invalid sigmas shape, we expect B == input.size(0). Got: {}".format(sigmas.shape))

    gradients: torch.Tensor = spatial_gradient(input, grads_mode)
    dx: torch.Tensor = gradients[:, :, 0]
    dy: torch.Tensor = gradients[:, :, 1]

    # compute the structure tensor M elements

    dx2: torch.Tensor = gaussian_blur2d(dx ** 2, (7, 7), (1.0, 1.0))
    dy2: torch.Tensor = gaussian_blur2d(dy ** 2, (7, 7), (1.0, 1.0))
    dxy: torch.Tensor = gaussian_blur2d(dx * dy, (7, 7), (1.0, 1.0))

    det_m: torch.Tensor = dx2 * dy2 - dxy * dxy
    trace_m: torch.Tensor = dx2 + dy2

    # compute the response map
    scores: torch.Tensor = det_m - k * (trace_m ** 2)

    if sigmas is not None:
        scores = scores * sigmas.pow(4).view(-1, 1, 1, 1)

    return scores


def gftt_response(
    input: torch.Tensor, grads_mode: str = 'sobel', sigmas: Optional[torch.Tensor] = None
) -> torch.Tensor:
    r"""Computes the Shi-Tomasi cornerness function.

    Function does not do any normalization or nms. The response map is computed according the following formulation:

    .. math::
        R = min(eig(M))

    where:

    .. math::
        M = \sum_{(x,y) \in W}
        \begin{bmatrix}
            I^{2}_x & I_x I_y \\
            I_x I_y & I^{2}_y \\
        \end{bmatrix}

    Args:
        input: input image with shape :math:`(B, C, H, W)`.
        grads_mode: can be ``'sobel'`` for standalone use or ``'diff'`` for use on Gaussian pyramid.
        sigmas: coefficients to be multiplied by multichannel response. Should be shape of :math:`(B)`
          It is necessary for performing non-maxima-suppression across different scale pyramid levels.
          See `vlfeat <https://github.com/vlfeat/vlfeat/blob/master/vl/covdet.c#L874>`_.

    Return:
        the response map per channel with shape :math:`(B, C, H, W)`.

    Example:
        >>> input = torch.tensor([[[
        ...    [0., 0., 0., 0., 0., 0., 0.],
        ...    [0., 1., 1., 1., 1., 1., 0.],
        ...    [0., 1., 1., 1., 1., 1., 0.],
        ...    [0., 1., 1., 1., 1., 1., 0.],
        ...    [0., 1., 1., 1., 1., 1., 0.],
        ...    [0., 1., 1., 1., 1., 1., 0.],
        ...    [0., 0., 0., 0., 0., 0., 0.],
        ... ]]])  # 1x1x7x7
        >>> # compute the response map
        gftt_response(input)
        tensor([[[[0.0155, 0.0334, 0.0194, 0.0000, 0.0194, 0.0334, 0.0155],
                  [0.0334, 0.0575, 0.0339, 0.0000, 0.0339, 0.0575, 0.0334],
                  [0.0194, 0.0339, 0.0497, 0.0000, 0.0497, 0.0339, 0.0194],
                  [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                  [0.0194, 0.0339, 0.0497, 0.0000, 0.0497, 0.0339, 0.0194],
                  [0.0334, 0.0575, 0.0339, 0.0000, 0.0339, 0.0575, 0.0334],
                  [0.0155, 0.0334, 0.0194, 0.0000, 0.0194, 0.0334, 0.0155]]]])
    """
    # TODO: Recompute doctest
    if not isinstance(input, torch.Tensor):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(type(input)))

    if not len(input.shape) == 4:
        raise ValueError("Invalid input shape, we expect BxCxHxW. Got: {}".format(input.shape))

    gradients: torch.Tensor = spatial_gradient(input, grads_mode)
    dx: torch.Tensor = gradients[:, :, 0]
    dy: torch.Tensor = gradients[:, :, 1]

    dx2: torch.Tensor = gaussian_blur2d(dx ** 2, (7, 7), (1.0, 1.0))
    dy2: torch.Tensor = gaussian_blur2d(dy ** 2, (7, 7), (1.0, 1.0))
    dxy: torch.Tensor = gaussian_blur2d(dx * dy, (7, 7), (1.0, 1.0))

    det_m: torch.Tensor = dx2 * dy2 - dxy * dxy
    trace_m: torch.Tensor = dx2 + dy2

    e1: torch.Tensor = 0.5 * (trace_m + torch.sqrt((trace_m ** 2 - 4 * det_m).abs()))
    e2: torch.Tensor = 0.5 * (trace_m - torch.sqrt((trace_m ** 2 - 4 * det_m).abs()))

    scores: torch.Tensor = torch.min(e1, e2)

    if sigmas is not None:
        scores = scores * sigmas.pow(4).view(-1, 1, 1, 1)

    return scores


def hessian_response(
    input: torch.Tensor, grads_mode: str = 'sobel', sigmas: Optional[torch.Tensor] = None
) -> torch.Tensor:
    r"""Computes the absolute of determinant of the Hessian matrix.

    Function does not do any normalization or nms. The response map is computed according the following formulation:

    .. math::
        R = det(H)

    where:

    .. math::
        M = \sum_{(x,y) \in W}
        \begin{bmatrix}
            I_{xx} & I_{xy} \\
            I_{xy} & I_{yy} \\
        \end{bmatrix}

    Args:
        input: input image with shape :math:`(B, C, H, W)`.
        grads_mode: can be ``'sobel'`` for standalone use or ``'diff'`` for use on Gaussian pyramid.
        sigmas: coefficients to be multiplied by multichannel response. Should be shape of :math:`(B)`
          It is necessary for performing non-maxima-suppression across different scale pyramid levels.
          See `vlfeat <https://github.com/vlfeat/vlfeat/blob/master/vl/covdet.c#L874>`_.

    Return:
        the response map per channel with shape :math:`(B, C, H, W)`.

    Shape:
       - Input: :math:`(B, C, H, W)`
       - Output: :math:`(B, C, H, W)`

    Examples:
        >>> input = torch.tensor([[[
        ...    [0., 0., 0., 0., 0., 0., 0.],
        ...    [0., 1., 1., 1., 1., 1., 0.],
        ...    [0., 1., 1., 1., 1., 1., 0.],
        ...    [0., 1., 1., 1., 1., 1., 0.],
        ...    [0., 1., 1., 1., 1., 1., 0.],
        ...    [0., 1., 1., 1., 1., 1., 0.],
        ...    [0., 0., 0., 0., 0., 0., 0.],
        ... ]]])  # 1x1x7x7
        >>> # compute the response map
        hessian_response(input)
        tensor([[[[0.0155, 0.0334, 0.0194, 0.0000, 0.0194, 0.0334, 0.0155],
                  [0.0334, 0.0575, 0.0339, 0.0000, 0.0339, 0.0575, 0.0334],
                  [0.0194, 0.0339, 0.0497, 0.0000, 0.0497, 0.0339, 0.0194],
                  [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                  [0.0194, 0.0339, 0.0497, 0.0000, 0.0497, 0.0339, 0.0194],
                  [0.0334, 0.0575, 0.0339, 0.0000, 0.0339, 0.0575, 0.0334],
                  [0.0155, 0.0334, 0.0194, 0.0000, 0.0194, 0.0334, 0.0155]]]])
    """
    # TODO: Recompute doctest
    if not isinstance(input, torch.Tensor):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(type(input)))

    if not len(input.shape) == 4:
        raise ValueError("Invalid input shape, we expect BxCxHxW. Got: {}".format(input.shape))

    if sigmas is not None:
        if not isinstance(sigmas, torch.Tensor):
            raise TypeError("sigmas type is not a torch.Tensor. Got {}".format(type(sigmas)))

        if (not len(sigmas.shape) == 1) or (sigmas.size(0) != input.size(0)):
            raise ValueError("Invalid sigmas shape, we expect B == input.size(0). Got: {}".format(sigmas.shape))

    gradients: torch.Tensor = spatial_gradient(input, grads_mode, 2)
    dxx: torch.Tensor = gradients[:, :, 0]
    dxy: torch.Tensor = gradients[:, :, 1]
    dyy: torch.Tensor = gradients[:, :, 2]

    scores: torch.Tensor = dxx * dyy - dxy ** 2

    if sigmas is not None:
        scores = scores * sigmas.pow(4).view(-1, 1, 1, 1)

    return scores


def dog_response(input: torch.Tensor) -> torch.Tensor:
    r"""Computes the Difference-of-Gaussian response.

    Args:
        input: a given the gaussian 5d tensor :math:`(B, C, D, H, W)`.

    Return:
        the response map per channel with shape :math:`(B, C, D-1, H, W)`.

    """
    if not isinstance(input, torch.Tensor):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(type(input)))

    if not len(input.shape) == 5:
        raise ValueError("Invalid input shape, we expect BxCxDxHxW. Got: {}".format(input.shape))

    return input[:, :, 1:] - input[:, :, :-1]


class BlobDoG(nn.Module):
    r"""Module that calculates Difference-of-Gaussians blobs.

    See :func:`~kornia.feature.dog_response` for details.
    """

    def __init__(self) -> None:
        super(BlobDoG, self).__init__()
        return

    def __repr__(self) -> str:
        return self.__class__.__name__

    def forward(self, input: torch.Tensor, sigmas: Optional[torch.Tensor] = None) -> torch.Tensor:  # type: ignore
        return dog_response(input)  # type: ignore


class CornerHarris(nn.Module):
    r"""Module that calculates Harris corners.

    See :func:`~kornia.feature.harris_response` for details.
    """

    def __init__(self, k: Union[float, torch.Tensor], grads_mode='sobel') -> None:
        super(CornerHarris, self).__init__()
        if type(k) is float:
            self.register_buffer('k', torch.tensor(k))
        else:
            self.register_buffer('k', k)  # type: ignore
        self.grads_mode: str = grads_mode
        return

    def __repr__(self) -> str:
        return self.__class__.__name__ + '(k=' + str(self.k) + ', ' + 'grads_mode=' + self.grads_mode + ')'

    def forward(self, input: torch.Tensor, sigmas: Optional[torch.Tensor] = None) -> torch.Tensor:  # type: ignore
        return harris_response(input, self.k, self.grads_mode, sigmas)  # type: ignore


class CornerGFTT(nn.Module):
    r"""Module that calculates Shi-Tomasi corners.

    See :func:`~kornia.feature.gfft_response` for details.
    """

    def __init__(self, grads_mode='sobel') -> None:
        super(CornerGFTT, self).__init__()
        self.grads_mode: str = grads_mode
        return

    def __repr__(self) -> str:
        return self.__class__.__name__ + 'grads_mode=' + self.grads_mode + ')'

    def forward(self, input: torch.Tensor, sigmas: Optional[torch.Tensor] = None) -> torch.Tensor:  # type: ignore
        return gftt_response(input, self.grads_mode, sigmas)


class BlobHessian(nn.Module):
    r"""Module that calculates Hessian blobs.

    See :func:`~kornia.feature.hessian_response` for details.
    """

    def __init__(self, grads_mode='sobel') -> None:
        super(BlobHessian, self).__init__()
        self.grads_mode: str = grads_mode
        return

    def __repr__(self) -> str:
        return self.__class__.__name__ + 'grads_mode=' + self.grads_mode + ')'

    def forward(self, input: torch.Tensor, sigmas: Optional[torch.Tensor] = None) -> torch.Tensor:  # type: ignore
        return hessian_response(input, self.grads_mode, sigmas)
