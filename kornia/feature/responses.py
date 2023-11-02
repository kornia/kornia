from typing import Optional, Union

import torch

from kornia.core import Module, Tensor, tensor
from kornia.core.check import KORNIA_CHECK_SHAPE
from kornia.filters import gaussian_blur2d, spatial_gradient


def _get_kernel_size(sigma: float) -> int:
    ksize = int(2.0 * 4.0 * sigma + 1.0)

    #  matches OpenCV, but may cause padding problem for small images
    #  PyTorch does not allow to pad more than original size.
    #  Therefore there is a hack in forward function

    if ksize % 2 == 0:
        ksize += 1
    return ksize


def harris_response(
    input: Tensor, k: Union[Tensor, float] = 0.04, grads_mode: str = "sobel", sigmas: Optional[Tensor] = None
) -> Tensor:
    r"""Compute the Harris cornerness function.

    .. image:: _static/img/harris_response.png

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
    KORNIA_CHECK_SHAPE(input, ["B", "C", "H", "W"])

    if sigmas is not None:
        if not isinstance(sigmas, Tensor):
            raise TypeError(f"sigmas type is not a Tensor. Got {type(sigmas)}")
        if (not len(sigmas.shape) == 1) or (sigmas.size(0) != input.size(0)):
            raise ValueError(f"Invalid sigmas shape, we expect B == input.size(0). Got: {sigmas.shape}")

    gradients: Tensor = spatial_gradient(input, grads_mode)
    dx: Tensor = gradients[:, :, 0]
    dy: Tensor = gradients[:, :, 1]

    # compute the structure tensor M elements

    dx2: Tensor = gaussian_blur2d(dx**2, (7, 7), (1.0, 1.0))
    dy2: Tensor = gaussian_blur2d(dy**2, (7, 7), (1.0, 1.0))
    dxy: Tensor = gaussian_blur2d(dx * dy, (7, 7), (1.0, 1.0))

    det_m: Tensor = dx2 * dy2 - dxy * dxy
    trace_m: Tensor = dx2 + dy2

    # compute the response map
    scores: Tensor = det_m - k * (trace_m**2)

    if sigmas is not None:
        scores = scores * sigmas.pow(4).view(-1, 1, 1, 1)

    return scores


def gftt_response(input: Tensor, grads_mode: str = "sobel", sigmas: Optional[Tensor] = None) -> Tensor:
    r"""Compute the Shi-Tomasi cornerness function.

    .. image:: _static/img/gftt_response.png

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
    KORNIA_CHECK_SHAPE(input, ["B", "C", "H", "W"])

    gradients: Tensor = spatial_gradient(input, grads_mode)
    dx: Tensor = gradients[:, :, 0]
    dy: Tensor = gradients[:, :, 1]

    dx2: Tensor = gaussian_blur2d(dx**2, (7, 7), (1.0, 1.0))
    dy2: Tensor = gaussian_blur2d(dy**2, (7, 7), (1.0, 1.0))
    dxy: Tensor = gaussian_blur2d(dx * dy, (7, 7), (1.0, 1.0))

    det_m: Tensor = dx2 * dy2 - dxy * dxy
    trace_m: Tensor = dx2 + dy2

    e1: Tensor = 0.5 * (trace_m + torch.sqrt((trace_m**2 - 4 * det_m).abs()))
    e2: Tensor = 0.5 * (trace_m - torch.sqrt((trace_m**2 - 4 * det_m).abs()))

    scores: Tensor = torch.min(e1, e2)

    if sigmas is not None:
        scores = scores * sigmas.pow(4).view(-1, 1, 1, 1)

    return scores


def hessian_response(input: Tensor, grads_mode: str = "sobel", sigmas: Optional[Tensor] = None) -> Tensor:
    r"""Compute the absolute of determinant of the Hessian matrix.

    .. image:: _static/img/hessian_response.png

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
    KORNIA_CHECK_SHAPE(input, ["B", "C", "H", "W"])

    if sigmas is not None:
        if not isinstance(sigmas, Tensor):
            raise TypeError(f"sigmas type is not a Tensor. Got {type(sigmas)}")

        if (not len(sigmas.shape) == 1) or (sigmas.size(0) != input.size(0)):
            raise ValueError(f"Invalid sigmas shape, we expect B == input.size(0). Got: {sigmas.shape}")

    gradients: Tensor = spatial_gradient(input, grads_mode, 2)
    dxx: Tensor = gradients[:, :, 0]
    dxy: Tensor = gradients[:, :, 1]
    dyy: Tensor = gradients[:, :, 2]

    scores: Tensor = dxx * dyy - dxy**2

    if sigmas is not None:
        scores = scores * sigmas.pow(4).view(-1, 1, 1, 1)

    return scores


def dog_response(input: Tensor) -> Tensor:
    r"""Compute the Difference-of-Gaussian response.

    Args:
        input: a given the gaussian 5d tensor :math:`(B, C, D, H, W)`.

    Return:
        the response map per channel with shape :math:`(B, C, D-1, H, W)`.
    """
    KORNIA_CHECK_SHAPE(input, ["B", "C", "L", "H", "W"])

    return input[:, :, 1:] - input[:, :, :-1]


def dog_response_single(input: Tensor, sigma1: float = 1.0, sigma2: float = 1.6) -> Tensor:
    r"""Compute the Difference-of-Gaussian response.

    .. image:: _static/img/dog_response_single.png

    Args:
        input: a given the gaussian 4d tensor :math:`(B, C, H, W)`.
        sigma1: lower gaussian sigma
        sigma2: bigger gaussian sigma

    Return:
        the response map per channel with shape :math:`(B, C, H, W)`.
    """
    KORNIA_CHECK_SHAPE(input, ["B", "C", "H", "W"])
    ks1 = _get_kernel_size(sigma1)
    ks2 = _get_kernel_size(sigma1)
    g1 = gaussian_blur2d(input, (ks1, ks1), (sigma1, sigma1))
    g2 = gaussian_blur2d(input, (ks2, ks2), (sigma2, sigma2))
    return g2 - g1


class BlobDoG(Module):
    r"""Module that calculates Difference-of-Gaussians blobs.

    See
    :func: `~kornia.feature.dog_response` for details.
    """

    def __init__(self) -> None:
        super().__init__()

    def __repr__(self) -> str:
        return self.__class__.__name__

    def forward(self, input: Tensor, sigmas: Optional[Tensor] = None) -> Tensor:
        return dog_response(input)


class BlobDoGSingle(Module):
    r"""Module that calculates Difference-of-Gaussians blobs.

    .. image:: _static/img/dog_response_single.png

    See :func:`~kornia.feature.dog_response_single` for details.
    """

    def __init__(self, sigma1: float = 1.0, sigma2: float = 1.6) -> None:
        super().__init__()
        self.sigma1 = sigma1
        self.sigma2 = sigma2

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}, sigma1={self.sigma1}, sigma2={self.sigma2})"

    def forward(self, input: Tensor, sigmas: Optional[Tensor] = None) -> Tensor:
        return dog_response_single(input, self.sigma1, self.sigma2)


class CornerHarris(Module):
    r"""Module that calculates Harris corners.

    .. image:: _static/img/harris_response.png

    See :func:`~kornia.feature.harris_response` for details.
    """

    k: Tensor

    def __init__(self, k: Union[float, Tensor], grads_mode: str = "sobel") -> None:
        super().__init__()
        if isinstance(k, float):
            self.register_buffer("k", tensor(k))
        else:
            self.register_buffer("k", k)
        self.grads_mode: str = grads_mode

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(k={self.k}, grads_mode={self.grads_mode})"

    def forward(self, input: Tensor, sigmas: Optional[Tensor] = None) -> Tensor:
        return harris_response(input, self.k, self.grads_mode, sigmas)


class CornerGFTT(Module):
    r"""Module that calculates Shi-Tomasi corners.

    .. image:: _static/img/gftt_response.png

    See :func:`~kornia.feature.gftt_response` for details.
    """

    def __init__(self, grads_mode: str = "sobel") -> None:
        super().__init__()
        self.grads_mode: str = grads_mode

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(grads_mode={self.grads_mode})"

    def forward(self, input: Tensor, sigmas: Optional[Tensor] = None) -> Tensor:
        return gftt_response(input, self.grads_mode, sigmas)


class BlobHessian(Module):
    r"""Module that calculates Hessian blobs.

    .. image:: _static/img/hessian_response.png

    See :func:`~kornia.feature.hessian_response` for details.
    """

    def __init__(self, grads_mode: str = "sobel") -> None:
        super().__init__()
        self.grads_mode: str = grads_mode

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(grads_mode={self.grads_mode})"

    def forward(self, input: Tensor, sigmas: Optional[Tensor] = None) -> Tensor:
        return hessian_response(input, self.grads_mode, sigmas)
