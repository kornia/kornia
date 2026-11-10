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

import numbers
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
    r"""Compute the Harris corner response det(M) - k trace(M)^2 of the windowed structure tensor M.

    ``R = harris_response(img, k)`` takes images ``(B, C, H, W)`` and returns the Harris response of every pixel of
    every channel, ``(B, C, H, W)``, without non-maxima suppression: positive at a corner, negative along an edge and
    zero on a constant image. ``k`` defaults to ``0.04``. To pass it to a detector as ``resp_module``, use its module
    form :class:`CornerHarris`; for a corner response with no ``k`` to tune and no negative values up to roundoff,
    use :func:`gftt_response`; for blobs, :func:`hessian_response`.

    .. image:: _static/img/harris_response.png

    The response map is computed according to the following formulation:

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
        - Each channel is processed independently. ``H`` and ``W`` must be at least 4: a smaller image raises
          ``RuntimeError`` from the reflect padding of the window.

    Args:
        input: input image with shape ``(B, C, H, W)``.
        k: the Harris detector free parameter, a number or a tensor that broadcasts against ``(B, C, H, W)``, such as
          ``(B, 1, 1, 1)`` for one value per image.
        grads_mode: can be ``'sobel'`` for standalone use or ``'diff'`` for use on Gaussian pyramid.
        sigmas: per-image scales with shape ``(B,)``. The response of image ``b`` is multiplied by
          ``sigmas[b] ** 4``, so that non-maxima suppression can compare the levels of a scale pyramid, as in
          `vlfeat <https://github.com/vlfeat/vlfeat/blob/master/vl/covdet.c>`_. Another type raises ``TypeError``
          and another shape ``ValueError``.

    Return:
        the response map per channel with shape ``(B, C, H, W)``.

    See Also:
        - :class:`CornerHarris`: the module form, with ``k`` and ``grads_mode`` fixed at construction.
        - :func:`gftt_response`: the smaller eigenvalue of the same structure tensor, with no ``k`` and not negative
          up to roundoff.
        - :func:`hessian_response`: a blob response from second derivatives, positive at blobs of both polarities.
        - :class:`ScaleSpaceDetector`: detects LAFs at the scale-space maxima of a response module such as
          :class:`CornerHarris`.

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
    r"""Compute the Shi-Tomasi corner response, the smaller eigenvalue of the windowed structure tensor.

    ``R = gftt_response(img)`` takes images ``(B, C, H, W)`` and returns the Shi-Tomasi ("good features to track")
    response of every pixel of every channel, ``(B, C, H, W)``, without non-maxima suppression: large at a corner, near
    zero along an edge and zero on a constant image. Unlike :func:`harris_response`, from the same structure tensor, it
    has no ``k`` to tune. To pass it to a detector as ``resp_module``, use its module form :class:`CornerGFTT`; for
    blobs, use :func:`hessian_response`.

    .. image:: _static/img/gftt_response.png

    The response map is computed according to the following formulation:

    .. math::
        R = min(eig(M))

    where :math:`M` is the windowed structure tensor of :func:`~kornia.feature.harris_response`, with the same
    gradients and window.

    Convention:
        - :math:`R` is not negative, up to roundoff. Along an edge, where it is near zero, the Harris response is
          negative.
        - Multiplying the image by ``c`` multiplies the response by ``c ** 2``, and an offset does not change it.
        - ``sigmas`` multiplies the response by ``sigmas ** 4``, the same power used by
          :func:`~kornia.feature.harris_response`. This is the function's scale-weighting convention, not a guarantee
          of scale invariance. The integration window :math:`W` stays fixed as the image blur changes; the response's
          decay with blur depends on the image structure, including the corner geometry.
        - Each channel is processed independently. ``H`` and ``W`` must be at least 4: a smaller image raises
          ``RuntimeError`` from the reflect padding of the window.

    Args:
        input: input image with shape ``(B, C, H, W)``.
        grads_mode: can be ``'sobel'`` for standalone use or ``'diff'`` for use on Gaussian pyramid.
        sigmas: per-image scales with shape ``(B,)``. The response of image ``b`` is multiplied by
          ``sigmas[b] ** 4``, so that non-maxima suppression can compare the levels of a scale pyramid. Another type
          raises ``TypeError`` and another shape ``ValueError``.

    Return:
        the response map per channel with shape ``(B, C, H, W)``.

    See Also:
        - :class:`CornerGFTT`: the module form, with ``grads_mode`` fixed at construction.
        - :func:`harris_response`: the signed Harris measure of the same structure tensor, with a free ``k``.
        - :func:`hessian_response`: a blob response from second derivatives, without a window.
        - :class:`GFTTAffNetHardNet`: a :class:`ScaleSpaceDetector` on this response, with AffNet shapes and HardNet
          descriptors.

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
    r"""Compute the blob response det(H) of the per-pixel Hessian, positive at bright and dark blobs alike.

    ``R = hessian_response(img)`` takes images ``(B, C, H, W)`` and returns the determinant of the Hessian of every
    pixel of every channel, ``(B, C, H, W)``, without non-maxima suppression. Its module form :class:`BlobHessian` is
    the default ``resp_module`` of :class:`ScaleSpaceDetector`. For a Difference-of-Gaussian response, which tells a
    bright blob (a minimum) from a dark one (a maximum), use :func:`dog_response`; for corners,
    :func:`harris_response` or :func:`gftt_response`.

    .. image:: _static/img/hessian_response.png

    The response map is computed according to the following formulation:

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
        - Each channel is processed independently.

    Args:
        input: input image with shape ``(B, C, H, W)``.
        grads_mode: can be ``'sobel'`` for standalone use or ``'diff'`` for use on Gaussian pyramid.
        sigmas: per-image scales with shape ``(B,)``. The response of image ``b`` is multiplied by
          ``sigmas[b] ** 4``, so that non-maxima suppression can compare the levels of a scale pyramid. Another type
          raises ``TypeError`` and another shape ``ValueError``.

    Return:
        the response map per channel with shape ``(B, C, H, W)``.

    Shape:
       - Input: ``(B, C, H, W)``
       - Output: ``(B, C, H, W)``

    See Also:
        - :class:`BlobHessian`: the module form, with ``grads_mode`` fixed at construction.
        - :func:`dog_response`: a signed blob response of a scale stack, a minimum at a bright blob.
        - :func:`harris_response`: a corner response from first derivatives in a window.
        - :class:`ScaleSpaceDetector`: detects LAFs at the scale-space maxima of this response, its default.

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
    r"""Compute Difference-of-Gaussian responses between adjacent levels of a (B, C, D, H, W) scale stack.

    ``R = dog_response(stack)`` takes a Gaussian scale stack ``(B, C, D, H, W)``, with its ``D`` levels along dim 2, and
    returns the differences of adjacent levels, ``(B, C, D - 1, H, W)``. An octave of
    :class:`~kornia.geometry.transform.ScalePyramid` is such a stack; to detect at the extrema, pass the module form
    :class:`BlobDoG` to :class:`ScaleSpaceDetector` with ``scale_space_response=True`` and
    ``minima_are_also_good=True``. For the DoG of one image at two given blurs, use :func:`dog_response_single`.

    Convention:
        - Output level ``i`` is ``input[:, :, i + 1] - input[:, :, i]`` along the level axis ``D`` (dim 2), so ``D``
          levels give ``D - 1``. For a stack blurred from fine to coarse, this is Lowe's
          :math:`L(k\sigma) - L(\sigma)`: negative at a bright blob, positive at a dark one.

    Args:
        input: a Gaussian scale stack ``(B, C, D, H, W)``, its levels along ``D``.

    Return:
        the response map per channel with shape ``(B, C, D - 1, H, W)``.

    See Also:
        - :class:`BlobDoG`: the module form, to pass to :class:`ScaleSpaceDetector` as ``resp_module``.
        - :func:`dog_response_single`: the DoG of a ``(B, C, H, W)`` image, blurred at two given sigmas.
        - :func:`hessian_response`: a blob response of one image, positive at blobs of both polarities.
        - :class:`~kornia.geometry.transform.ScalePyramid`: builds the scale stacks, one ``(B, C, L, H_o, W_o)`` per
          octave.

    """
    KORNIA_CHECK_SHAPE(input, ["B", "C", "L", "H", "W"])

    return input[:, :, 1:] - input[:, :, :-1]


def dog_response_single(input: torch.Tensor, sigma1: float = 1.0, sigma2: float = 1.6) -> torch.Tensor:
    r"""Compute the Difference-of-Gaussian response of a (B, C, H, W) image, its sigma2 blur minus its sigma1 blur.

    ``R = dog_response_single(img, sigma1, sigma2)`` takes images ``(B, C, H, W)`` and returns, per channel, their
    Gaussian blur of ``sigma2`` minus their blur of ``sigma1``, ``(B, C, H, W)``: for ``sigma2 > sigma1``, negative at
    a bright blob and positive at a dark one. It blurs the image itself, so it needs no scale stack; for the
    differences of adjacent levels of a precomputed stack, use :func:`dog_response`. Its module form is
    :class:`BlobDoGSingle`.

    .. image:: _static/img/dog_response_single.png

    Convention:
        - Returns the Gaussian blur of ``sigma2`` minus that of ``sigma1``, the order of
          :func:`~kornia.feature.dog_response`; see its Convention block. Each kernel has the size
          ``int(8 * sigma + 1)``, rounded up to an odd number. The order of ``sigma1`` and ``sigma2`` is not checked:
          swapping them flips the sign.
        - ``H`` and ``W`` must be larger than half the larger kernel, at least 7 for the default ``sigma2=1.6``: a
          smaller image raises ``RuntimeError`` from the reflect padding of the blur.

    Args:
        input: images ``(B, C, H, W)``.
        sigma1: lower gaussian sigma
        sigma2: bigger gaussian sigma

    Return:
        the response map per channel with shape ``(B, C, H, W)``.

    See Also:
        - :class:`BlobDoGSingle`: the module form, with ``sigma1`` and ``sigma2`` fixed at construction.
        - :func:`dog_response`: the differences of adjacent levels of a ``(B, C, D, H, W)`` scale stack.
        - :func:`hessian_response`: a blob response that is positive at blobs of both polarities.
        - :class:`SIFTFeature`: detects at the maxima of this response's magnitude on each level of an image pyramid.

    """
    KORNIA_CHECK_SHAPE(input, ["B", "C", "H", "W"])
    ks1 = _get_kernel_size(sigma1)
    ks2 = _get_kernel_size(sigma2)
    g1 = gaussian_blur2d(input, (ks1, ks1), (sigma1, sigma1))
    g2 = gaussian_blur2d(input, (ks2, ks2), (sigma2, sigma2))
    return g2 - g1


class BlobDoG(nn.Module):
    r"""Compute Difference-of-Gaussian responses of a (B, C, D, H, W) scale stack; the module form of dog_response.

    ``R = BlobDoG()(stack, sigmas)`` returns ``dog_response(stack)``, the ``(B, C, D - 1, H, W)`` differences of
    adjacent levels, and ignores ``sigmas``. It is the ``resp_module`` that :class:`ScaleSpaceDetector` takes with
    ``scale_space_response=True``; since a bright blob is a DoG minimum, pass ``minima_are_also_good=True`` too, as
    :class:`DoGHardNet` and the ``"patch"`` backend of :class:`SIFTFeatureScaleSpace` do. For the DoG of one image at
    two given blurs, use :class:`BlobDoGSingle` instead.

    Convention:
        Returns :func:`~kornia.feature.dog_response` of the ``(B, C, D, H, W)`` scale stack; see its Convention
        block.

    See Also:
        - :func:`dog_response`: the function form.
        - :class:`BlobDoGSingle`: the DoG of a ``(B, C, H, W)`` image at two sigmas fixed at construction.
        - :class:`BlobHessian`: a per-level blob response, positive at blobs of both polarities.
        - :class:`ScaleSpaceDetector`: detects LAFs at the extrema of this response over its scale pyramid.
        - :class:`SIFTFeatureScaleSpace`: a SIFT preset whose ``"patch"`` backend is that detector on this response.

    """

    def __init__(self) -> None:
        super().__init__()

    def __repr__(self) -> str:
        return self.__class__.__name__

    def forward(self, input: torch.Tensor, sigmas: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute Difference-of-Gaussians response maps.

        Args:
            input: Input scale stack of shape ``(B, C, D, H, W)``.
            sigmas: Unused in this wrapper; kept for API compatibility.

        Returns:
            DoG response tensor from :func:`dog_response`.
        """
        return dog_response(input)


class BlobDoGSingle(nn.Module):
    r"""Compute the Difference-of-Gaussian response of a (B, C, H, W) image; the module form of dog_response_single.

    ``R = BlobDoGSingle(sigma1, sigma2)(img, sigmas)`` returns ``dog_response_single(img, sigma1, sigma2)``, the
    ``(B, C, H, W)`` blur of ``sigma2`` minus the blur of ``sigma1``, and ignores ``sigmas``. Passed to
    :class:`MultiResolutionDetector`, which keeps maxima only, it misses the centres of bright blobs, the DoG minima:
    :class:`SIFTFeature` passes its magnitude instead. For the DoG of a whole scale stack, use :class:`BlobDoG`.

    .. image:: _static/img/dog_response_single.png

    Convention:
        Returns :func:`~kornia.feature.dog_response_single` with the constructor's ``sigma1`` and ``sigma2``; see its
        Convention block.

    See Also:
        - :func:`dog_response_single`: the function form, with the sigmas passed on each call.
        - :class:`BlobDoG`: the DoG between adjacent levels of a ``(B, C, D, H, W)`` scale stack.
        - :class:`MultiResolutionDetector`: detects LAFs at the maxima of a per-image response on an image pyramid.

    """

    def __init__(self, sigma1: float = 1.0, sigma2: float = 1.6) -> None:
        super().__init__()
        self.sigma1 = sigma1
        self.sigma2 = sigma2

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(sigma1={self.sigma1}, sigma2={self.sigma2})"

    def forward(self, input: torch.Tensor, sigmas: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute single-scale Difference-of-Gaussians response.

        Args:
            input: Input tensor of shape ``(B, C, H, W)``.
            sigmas: Unused in this wrapper; kept for API compatibility.

        Returns:
            Response tensor from :func:`dog_response_single` using stored
            ``sigma1`` and ``sigma2``.
        """
        return dog_response_single(input, self.sigma1, self.sigma2)


class CornerHarris(nn.Module):
    r"""Compute the Harris corner response with a fixed k; the module form of harris_response.

    ``R = CornerHarris(k)(img, sigmas)`` returns ``harris_response(img, k, grads_mode, sigmas)``, the ``(B, C, H, W)``
    Harris response of ``(B, C, H, W)`` images, positive at a corner. ``k`` is stored as a buffer, so it moves with
    :meth:`torch.nn.Module.to`. Pass it to :class:`ScaleSpaceDetector` as ``resp_module`` to detect corners over
    scale; for a response with no ``k`` to choose, use :class:`CornerGFTT` instead.

    .. image:: _static/img/harris_response.png

    Convention:
        Returns :func:`~kornia.feature.harris_response` with the constructor's ``k`` and ``grads_mode``; see its
        Convention block. Unlike the function's, this ``k`` has no default. A ``float``, ``int`` or NumPy scalar
        ``k`` becomes a 0-dimensional float32 tensor, and a float64 image uses the number itself while the buffer
        still holds it; a tensor is stored as it is.

    See Also:
        - :func:`harris_response`: the function form, with ``k`` defaulting to ``0.04``.
        - :class:`CornerGFTT`: the Shi-Tomasi corner response of the same structure tensor, without ``k``.
        - :class:`BlobHessian`: the Hessian-determinant blob response.
        - :class:`ScaleSpaceDetector`: detects LAFs at the scale-space maxima of a response module such as this one.

    """

    k: torch.Tensor

    def __init__(self, k: Union[float, torch.Tensor], grads_mode: str = "sobel") -> None:
        super().__init__()
        # The number itself, for a float64 image: the buffer holds it rounded to float32.
        self._k_number: Optional[float] = None
        if isinstance(k, numbers.Real) and not isinstance(k, bool):
            # A float, an int or a NumPy scalar, which harris_response accepts: register_buffer takes only tensors.
            self._k_number = float(k)
            self.register_buffer("k", torch.tensor(float(k)))
        else:
            self.register_buffer("k", k)
        self.grads_mode: str = grads_mode

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(k={self.k}, grads_mode={self.grads_mode})"

    def forward(self, input: torch.Tensor, sigmas: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute Harris corner response for each input channel.

        Args:
            input: Input tensor of shape ``(B, C, H, W)``.
            sigmas: Optional per-image scales forwarded to :func:`harris_response`, which multiplies the
                response by ``sigmas ** 4``.

        Returns:
            Harris response map tensor.
        """
        k = self.k
        number = self.__dict__.get("_k_number")  # absent from a module pickled before it existed
        if number is not None and input.dtype == torch.float64 and k.is_floating_point():
            # While the buffer still holds the number as built (not loaded or assigned another k), apply it exactly.
            built = torch.tensor(number).to(device=k.device, dtype=k.dtype)
            # One-dimensional: the TorchScript ONNX exporter casts a 0-dimensional constant to the buffer's float32.
            exact = torch.tensor([number], dtype=torch.float64, device=k.device)
            k64 = k.to(torch.float64)
            # ``k64 - k64.detach()`` is zero, so the value stays exact while a trainable ``k`` keeps its gradient.
            k = torch.where(k == built, exact + (k64 - k64.detach()), k64).reshape(k.shape)
        return harris_response(input, k, self.grads_mode, sigmas)


class CornerGFTT(nn.Module):
    r"""Compute the Shi-Tomasi corner response; the module form of gftt_response.

    ``R = CornerGFTT()(img, sigmas)`` returns ``gftt_response(img, grads_mode, sigmas)``, the ``(B, C, H, W)``
    smaller structure-tensor eigenvalue of ``(B, C, H, W)`` images, large at a corner. Pass it to
    :class:`ScaleSpaceDetector` as ``resp_module`` to detect corners over scale, as :class:`GFTTAffNetHardNet` does;
    for the signed Harris measure, use :class:`CornerHarris` instead.

    .. image:: _static/img/gftt_response.png

    Convention:
        Returns :func:`~kornia.feature.gftt_response` with the constructor's ``grads_mode``; see its Convention block.

    See Also:
        - :func:`gftt_response`: the function form.
        - :class:`CornerHarris`: the Harris corner response of the same structure tensor, with a free ``k``.
        - :class:`BlobHessian`: the Hessian-determinant blob response.
        - :class:`GFTTAffNetHardNet`: a :class:`ScaleSpaceDetector` on this response, with AffNet shapes and HardNet
          descriptors.

    """

    def __init__(self, grads_mode: str = "sobel") -> None:
        super().__init__()
        self.grads_mode: str = grads_mode

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(grads_mode={self.grads_mode})"

    def forward(self, input: torch.Tensor, sigmas: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute Shi-Tomasi (GFTT) corner response maps.

        Args:
            input: Input tensor of shape ``(B, C, H, W)``.
            sigmas: Optional per-image scales forwarded to :func:`gftt_response`, which multiplies the
                response by ``sigmas ** 4``.

        Returns:
            GFTT response map tensor from :func:`gftt_response`.
        """
        return gftt_response(input, self.grads_mode, sigmas)


class BlobHessian(nn.Module):
    r"""Compute the Hessian-determinant blob response; the module form of hessian_response.

    ``R = BlobHessian()(img, sigmas)`` returns ``hessian_response(img, grads_mode, sigmas)``, the ``(B, C, H, W)``
    Hessian determinant of ``(B, C, H, W)`` images, positive at bright and dark blobs alike. It is the default
    ``resp_module`` of :class:`ScaleSpaceDetector`, which passes each level's sigma so that the levels compare. For a
    DoG response over a scale stack, use :class:`BlobDoG`; for corners, :class:`CornerGFTT` or :class:`CornerHarris`.

    .. image:: _static/img/hessian_response.png

    Convention:
        Returns :func:`~kornia.feature.hessian_response` with the constructor's ``grads_mode``; see its Convention
        block.

    See Also:
        - :func:`hessian_response`: the function form.
        - :class:`BlobDoG`: the DoG blob response of a scale stack, a minimum at a bright blob.
        - :class:`CornerGFTT`: a corner response, the smaller structure-tensor eigenvalue.
        - :class:`ScaleSpaceDetector`: detects LAFs at the scale-space maxima of this response, its default.
        - :class:`HesAffNetHardNet`: that detector on this response, with AffNet shapes and HardNet descriptors.

    """

    def __init__(self, grads_mode: str = "sobel") -> None:
        super().__init__()
        self.grads_mode: str = grads_mode

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(grads_mode={self.grads_mode})"

    def forward(self, input: torch.Tensor, sigmas: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute Hessian determinant-like blob responses.

        Args:
            input: Input tensor of shape ``(B, C, H, W)``.
            sigmas: Optional per-image scales forwarded to :func:`hessian_response`, which multiplies the
                response by ``sigmas ** 4``.

        Returns:
            Hessian response map tensor.
        """
        return hessian_response(input, self.grads_mode, sigmas)
