# LICENSE HEADER MANAGED BY add-license-header
#
# Copyright 2018 Kornia Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from typing import Optional, Union

import torch
from torch import nn

from kornia.core.check import KORNIA_CHECK_SHAPE
from kornia.filters import gaussian_blur2d, spatial_gradient


def _get_kernel_size(sigma: float) -> int:
    ksize = int(2.0 * 4.0 * sigma + 1.0)

    #  matches OpenCV, but may cause padding problem for small images
    #  PyTorch does not allow to F.pad more than original size.
    #  Therefore there is a hack in forward function

    if ksize % 2 == 0:
        ksize += 1
    return ksize


def harris_response(
    input: torch.Tensor,
    k: Union[torch.Tensor, float] = 0.04,
    grads_mode: str = "sobel",
    sigmas: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    r"""Compute the Harris cornerness function.

    .. image:: _static/img/harris_response.png

    The function does no non-maxima suppression. The response map is computed according to the following
    formulation:

    .. math::
        R = det(M) - k \cdot trace(M)^2

    where :math:`w` are the weights of a window :math:`W`:

    .. math::
        M = \sum_{(x,y) \in W} w(x, y)
        \begin{bmatrix}
            I^{2}_x & I_x I_y \\
            I_x I_y & I^{2}_y \\
        \end{bmatrix}

    and :math:`k` is an empirically determined constant, usually in :math:`[0.04, 0.06]`.

    Convention:
        - :math:`R` is signed and not clamped: positive at a corner, negative along an edge.
        - :math:`I_x` and :math:`I_y` come from :func:`~kornia.filters.spatial_gradient` with ``mode=grads_mode``,
          in intensity per pixel for both modes, and :math:`W` is a :math:`7 \times 7` Gaussian window of
          :math:`\sigma = 1` whose weights sum to one. A linear ramp of gradient :math:`g` therefore gives
          :math:`R = -k \|g\|^4` away from the border.
        - Apart from ``sigmas``, the response depends on the image's intensity scale: multiplying the image by ``c``
          multiplies the response by ``c ** 4``, and an offset does not change it, so a threshold tuned on images in
          ``[0, 1]`` does not carry over to ``[0, 255]``.
        - ``k`` is not validated.

    Args:
        input: input image with shape :math:`(B, C, H, W)`.
        k: the Harris detector free parameter.
        grads_mode: can be ``'sobel'`` for standalone use or ``'diff'`` for use on Gaussian pyramid.
        sigmas: per-image scales with shape :math:`(B)`. The response of image ``b`` is multiplied by
          ``sigmas[b] ** 4``, so that non-maxima suppression can compare the levels of a scale pyramid, as in
          `vlfeat <https://github.com/vlfeat/vlfeat/blob/master/vl/covdet.c>`_.

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
        >>> # compute the response map; rounded for a platform-stable display
        >>> harris_response(input, 0.04).round(decimals=4) + 0.0
        tensor([[[[0.0042, 0.0054, 0.0035, 0.0006, 0.0035, 0.0054, 0.0042],
                  [0.0054, 0.0068, 0.0046, 0.0014, 0.0046, 0.0068, 0.0054],
                  [0.0035, 0.0046, 0.0034, 0.0014, 0.0034, 0.0046, 0.0035],
                  [0.0006, 0.0014, 0.0014, 0.0006, 0.0014, 0.0014, 0.0006],
                  [0.0035, 0.0046, 0.0034, 0.0014, 0.0034, 0.0046, 0.0035],
                  [0.0054, 0.0068, 0.0046, 0.0014, 0.0046, 0.0068, 0.0054],
                  [0.0042, 0.0054, 0.0035, 0.0006, 0.0035, 0.0054, 0.0042]]]])

    """
    KORNIA_CHECK_SHAPE(input, ["B", "C", "H", "W"])

    if sigmas is not None:
        if not isinstance(sigmas, torch.Tensor):
            raise TypeError(f"sigmas type is not a torch.Tensor. Got {type(sigmas)}")
        if (not len(sigmas.shape) == 1) or (sigmas.size(0) != input.size(0)):
            raise ValueError(f"Invalid sigmas shape, we expect B == input.size(0). Got: {sigmas.shape}")

    gradients: torch.Tensor = spatial_gradient(input, grads_mode)
    dx: torch.Tensor = gradients[:, :, 0]
    dy: torch.Tensor = gradients[:, :, 1]

    # compute the structure torch.Tensor M elements

    dx2: torch.Tensor = gaussian_blur2d(dx**2, (7, 7), (1.0, 1.0))
    dy2: torch.Tensor = gaussian_blur2d(dy**2, (7, 7), (1.0, 1.0))
    dxy: torch.Tensor = gaussian_blur2d(dx * dy, (7, 7), (1.0, 1.0))

    det_m: torch.Tensor = dx2 * dy2 - dxy * dxy
    trace_m: torch.Tensor = dx2 + dy2

    # compute the response map
    scores: torch.Tensor = det_m - k * (trace_m**2)

    if sigmas is not None:
        scores = scores * sigmas.pow(4).view(-1, 1, 1, 1)

    return scores


def gftt_response(
    input: torch.Tensor, grads_mode: str = "sobel", sigmas: Optional[torch.Tensor] = None
) -> torch.Tensor:
    r"""Compute the Shi-Tomasi cornerness function.

    .. image:: _static/img/gftt_response.png

    The function does no non-maxima suppression. The response map is computed according to the following
    formulation:

    .. math::
        R = min(eig(M))

    where :math:`M` is the windowed structure tensor of :func:`~kornia.feature.harris_response`, with the same
    gradients and window.

    Convention:
        - :math:`R` is not negative, up to roundoff. It is large at a corner and near zero along an edge, where the
          Harris response is negative.
        - Multiplying the image by ``c`` multiplies the response by ``c ** 2``, and an offset does not change it.
        - ``sigmas`` multiplies the response by ``sigmas ** 4``, the power of :func:`~kornia.feature.harris_response`,
          not the ``sigmas ** 2`` of the smaller eigenvalue's degree in the gradients. That degree holds for a window
          that grows with the blur :math:`\sigma`; :math:`W` keeps its size, so at a corner the smaller eigenvalue
          falls faster than :math:`\sigma^{-3}` over the blurs of a :class:`~kornia.geometry.ScalePyramid` octave
          (1.6 to 4 pixels), and ``sigmas ** 4`` is the closer correction.

    Args:
        input: input image with shape :math:`(B, C, H, W)`.
        grads_mode: can be ``'sobel'`` for standalone use or ``'diff'`` for use on Gaussian pyramid.
        sigmas: per-image scales with shape :math:`(B)`. The response of image ``b`` is multiplied by
          ``sigmas[b] ** 4``, so that non-maxima suppression can compare the levels of a scale pyramid.

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
        >>> # compute the response map; rounded for a platform-stable display
        >>> gftt_response(input).round(decimals=4) + 0.0
        tensor([[[[0.0379, 0.0456, 0.0283, 0.0121, 0.0283, 0.0456, 0.0379],
                  [0.0456, 0.0598, 0.0402, 0.0168, 0.0402, 0.0598, 0.0456],
                  [0.0283, 0.0402, 0.0545, 0.0245, 0.0545, 0.0402, 0.0283],
                  [0.0121, 0.0168, 0.0245, 0.0276, 0.0245, 0.0168, 0.0121],
                  [0.0283, 0.0402, 0.0545, 0.0245, 0.0545, 0.0402, 0.0283],
                  [0.0456, 0.0598, 0.0402, 0.0168, 0.0402, 0.0598, 0.0456],
                  [0.0379, 0.0456, 0.0283, 0.0121, 0.0283, 0.0456, 0.0379]]]])

    """
    KORNIA_CHECK_SHAPE(input, ["B", "C", "H", "W"])

    if sigmas is not None:
        if not isinstance(sigmas, torch.Tensor):
            raise TypeError(f"sigmas type is not a torch.Tensor. Got {type(sigmas)}")
        if (not len(sigmas.shape) == 1) or (sigmas.size(0) != input.size(0)):
            raise ValueError(f"Invalid sigmas shape, we expect B == input.size(0). Got: {sigmas.shape}")

    gradients: torch.Tensor = spatial_gradient(input, grads_mode)
    dx: torch.Tensor = gradients[:, :, 0]
    dy: torch.Tensor = gradients[:, :, 1]

    dx2: torch.Tensor = gaussian_blur2d(dx**2, (7, 7), (1.0, 1.0))
    dy2: torch.Tensor = gaussian_blur2d(dy**2, (7, 7), (1.0, 1.0))
    dxy: torch.Tensor = gaussian_blur2d(dx * dy, (7, 7), (1.0, 1.0))

    trace_m: torch.Tensor = dx2 + dy2

    # trace^2 - 4 det as a sum of squares: it is never negative, and the sqrt below gets a zero gradient instead
    # of 0 / 0 = nan where the structure tensor vanishes, which is every flat 9 x 9 patch of the image.
    disc: torch.Tensor = (dx2 - dy2) ** 2 + 4.0 * dxy * dxy
    positive = disc > 0
    root: torch.Tensor = torch.where(positive, torch.where(positive, disc, torch.ones_like(disc)).sqrt(), 0.0)

    scores: torch.Tensor = 0.5 * (trace_m - root)  # the smaller eigenvalue

    if sigmas is not None:
        scores = scores * sigmas.pow(4).view(-1, 1, 1, 1)

    return scores


def hessian_response(
    input: torch.Tensor, grads_mode: str = "sobel", sigmas: Optional[torch.Tensor] = None
) -> torch.Tensor:
    r"""Compute the determinant of the Hessian matrix.

    .. image:: _static/img/hessian_response.png

    The function does no non-maxima suppression. The response map is computed according to the following
    formulation:

    .. math::
        R = det(H)

    where:

    .. math::
        H =
        \begin{bmatrix}
            I_{xx} & I_{xy} \\
            I_{xy} & I_{yy} \\
        \end{bmatrix}

    Convention:
        - :math:`R = I_{xx} I_{yy} - I_{xy}^2` is signed and pointwise, without a window: positive at bright and dark
          blobs alike, negative at a saddle. The second derivatives come from
          :func:`~kornia.filters.spatial_gradient` with ``mode=grads_mode`` and ``order=2``.
        - Multiplying the image by ``c`` multiplies the response by ``c ** 2``, and an offset does not change it.

    Args:
        input: input image with shape :math:`(B, C, H, W)`.
        grads_mode: can be ``'sobel'`` for standalone use or ``'diff'`` for use on Gaussian pyramid.
        sigmas: per-image scales with shape :math:`(B)`. The response of image ``b`` is multiplied by
          ``sigmas[b] ** 4``, so that non-maxima suppression can compare the levels of a scale pyramid.

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
        >>> # compute the response map; rounded for a platform-stable display
        >>> hessian_response(input).round(decimals=4) + 0.0
        tensor([[[[-0.0137, -0.0332, -0.0205,  0.0000, -0.0205, -0.0332, -0.0137],
                  [-0.0332,  0.0098,  0.0381,  0.0000,  0.0381,  0.0098, -0.0332],
                  [-0.0205,  0.0381,  0.0547,  0.0000,  0.0547,  0.0381, -0.0205],
                  [ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000],
                  [-0.0205,  0.0381,  0.0547,  0.0000,  0.0547,  0.0381, -0.0205],
                  [-0.0332,  0.0098,  0.0381,  0.0000,  0.0381,  0.0098, -0.0332],
                  [-0.0137, -0.0332, -0.0205,  0.0000, -0.0205, -0.0332, -0.0137]]]])

    """
    KORNIA_CHECK_SHAPE(input, ["B", "C", "H", "W"])

    if sigmas is not None:
        if not isinstance(sigmas, torch.Tensor):
            raise TypeError(f"sigmas type is not a torch.Tensor. Got {type(sigmas)}")

        if (not len(sigmas.shape) == 1) or (sigmas.size(0) != input.size(0)):
            raise ValueError(f"Invalid sigmas shape, we expect B == input.size(0). Got: {sigmas.shape}")

    gradients: torch.Tensor = spatial_gradient(input, grads_mode, 2)
    dxx: torch.Tensor = gradients[:, :, 0]
    dxy: torch.Tensor = gradients[:, :, 1]
    dyy: torch.Tensor = gradients[:, :, 2]

    scores: torch.Tensor = dxx * dyy - dxy**2

    if sigmas is not None:
        scores = scores * sigmas.pow(4).view(-1, 1, 1, 1)

    return scores


def dog_response(input: torch.Tensor) -> torch.Tensor:
    r"""Compute the Difference-of-Gaussian response.

    Convention:
        - Output level ``i`` is ``input[:, :, i + 1] - input[:, :, i]`` along the level axis ``D`` (dim 2), so ``D``
          levels give ``D - 1``. For a stack blurred from fine to coarse, this is Lowe's
          :math:`L(k\sigma) - L(\sigma)`: negative at a bright blob, positive at a dark one.

    Args:
        input: a given the gaussian 5d torch.Tensor :math:`(B, C, D, H, W)`.

    Return:
        the response map per channel with shape :math:`(B, C, D-1, H, W)`.

    """
    KORNIA_CHECK_SHAPE(input, ["B", "C", "L", "H", "W"])

    return input[:, :, 1:] - input[:, :, :-1]


def dog_response_single(input: torch.Tensor, sigma1: float = 1.0, sigma2: float = 1.6) -> torch.Tensor:
    r"""Compute the Difference-of-Gaussian response.

    .. image:: _static/img/dog_response_single.png

    Convention:
        - Returns the Gaussian blur of ``sigma2`` minus that of ``sigma1``, the order of
          :func:`~kornia.feature.dog_response`; see its Convention block. Each kernel has the size
          ``int(8 * sigma + 1)``, rounded up to an odd number. The order of ``sigma1`` and ``sigma2`` is not checked:
          swapping them flips the sign.

    Args:
        input: a given the gaussian 4d torch.Tensor :math:`(B, C, H, W)`.
        sigma1: lower gaussian sigma
        sigma2: bigger gaussian sigma

    Return:
        the response map per channel with shape :math:`(B, C, H, W)`.

    """
    KORNIA_CHECK_SHAPE(input, ["B", "C", "H", "W"])
    ks1 = _get_kernel_size(sigma1)
    ks2 = _get_kernel_size(sigma2)
    g1 = gaussian_blur2d(input, (ks1, ks1), (sigma1, sigma1))
    g2 = gaussian_blur2d(input, (ks2, ks2), (sigma2, sigma2))
    return g2 - g1


class BlobDoG(nn.Module):
    r"""nn.Module that calculates Difference-of-Gaussians blobs.

    Convention:
        Returns :func:`~kornia.feature.dog_response` of the :math:`(B, C, D, H, W)` scale stack; see its Convention
        block. ``sigmas`` is ignored.
    """

    def __init__(self) -> None:
        super().__init__()

    def __repr__(self) -> str:
        return self.__class__.__name__

    def forward(self, input: torch.Tensor, sigmas: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute Difference-of-Gaussians response maps.

        Args:
            input: Input scale stack of shape :math:`(B, C, D, H, W)`.
            sigmas: Unused in this wrapper; kept for API compatibility.

        Returns:
            DoG response tensor from :func:`dog_response`.
        """
        return dog_response(input)


class BlobDoGSingle(nn.Module):
    r"""nn.Module that calculates Difference-of-Gaussians blobs.

    .. image:: _static/img/dog_response_single.png

    Convention:
        Returns :func:`~kornia.feature.dog_response_single` with the constructor's ``sigma1`` and ``sigma2``; see its
        Convention block. ``sigmas`` is ignored.
    """

    def __init__(self, sigma1: float = 1.0, sigma2: float = 1.6) -> None:
        super().__init__()
        self.sigma1 = sigma1
        self.sigma2 = sigma2

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}, sigma1={self.sigma1}, sigma2={self.sigma2})"

    def forward(self, input: torch.Tensor, sigmas: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute single-scale Difference-of-Gaussians response.

        Args:
            input: Input tensor of shape :math:`(B, C, H, W)`.
            sigmas: Unused in this wrapper; kept for API compatibility.

        Returns:
            Response tensor from :func:`dog_response_single` using stored
            ``sigma1`` and ``sigma2``.
        """
        return dog_response_single(input, self.sigma1, self.sigma2)


class CornerHarris(nn.Module):
    r"""nn.Module that calculates Harris corners.

    .. image:: _static/img/harris_response.png

    Convention:
        Returns :func:`~kornia.feature.harris_response` with the constructor's ``k`` and ``grads_mode``; see its
        Convention block. Unlike the function's, this ``k`` has no default.
    """

    k: torch.Tensor

    def __init__(self, k: Union[float, torch.Tensor], grads_mode: str = "sobel") -> None:
        super().__init__()
        if isinstance(k, float):
            self.register_buffer("k", torch.tensor(k))
        else:
            self.register_buffer("k", k)
        self.grads_mode: str = grads_mode

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(k={self.k}, grads_mode={self.grads_mode})"

    def forward(self, input: torch.Tensor, sigmas: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute Harris corner response for each input channel.

        Args:
            input: Input tensor of shape :math:`(B, C, H, W)`.
            sigmas: Optional per-image scales forwarded to :func:`harris_response`, which multiplies the
                response by ``sigmas ** 4``.

        Returns:
            Harris response map tensor.
        """
        return harris_response(input, self.k, self.grads_mode, sigmas)


class CornerGFTT(nn.Module):
    r"""nn.Module that calculates Shi-Tomasi corners.

    .. image:: _static/img/gftt_response.png

    Convention:
        Returns :func:`~kornia.feature.gftt_response` with the constructor's ``grads_mode``; see its Convention block.
    """

    def __init__(self, grads_mode: str = "sobel") -> None:
        super().__init__()
        self.grads_mode: str = grads_mode

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(grads_mode={self.grads_mode})"

    def forward(self, input: torch.Tensor, sigmas: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute Shi-Tomasi (GFTT) corner response maps.

        Args:
            input: Input tensor of shape :math:`(B, C, H, W)`.
            sigmas: Optional per-image scales forwarded to :func:`gftt_response`, which multiplies the
                response by ``sigmas ** 4``.

        Returns:
            GFTT response map tensor from :func:`gftt_response`.
        """
        return gftt_response(input, self.grads_mode, sigmas)


class BlobHessian(nn.Module):
    r"""nn.Module that calculates Hessian blobs.

    .. image:: _static/img/hessian_response.png

    Convention:
        Returns :func:`~kornia.feature.hessian_response` with the constructor's ``grads_mode``; see its Convention
        block.
    """

    def __init__(self, grads_mode: str = "sobel") -> None:
        super().__init__()
        self.grads_mode: str = grads_mode

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(grads_mode={self.grads_mode})"

    def forward(self, input: torch.Tensor, sigmas: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute Hessian determinant-like blob responses.

        Args:
            input: Input tensor of shape :math:`(B, C, H, W)`.
            sigmas: Optional per-image scales forwarded to :func:`hessian_response`, which multiplies the
                response by ``sigmas ** 4``.

        Returns:
            Hessian response map tensor.
        """
        return hessian_response(input, self.grads_mode, sigmas)
