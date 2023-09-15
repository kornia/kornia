from __future__ import annotations

from kornia.core import Module, IntegratedTensor, pad
from kornia.core.check import KORNIA_CHECK_IS_TENSOR, KORNIA_CHECK_SHAPE

from .kernels import get_spatial_gradient_kernel2d, get_spatial_gradient_kernel3d, normalize_kernel2d

import keras_core as keras

def spatial_gradient(input: IntegratedTensor, mode: str = 'sobel', order: int = 1, normalized: bool = True) -> IntegratedTensor:
    r"""Compute the first order image derivative in both x and y using a Sobel operator.

    .. image:: _static/img/spatial_gradient.png

    Args:
        input: input image tensor with shape :math:`(B, C, H, W)`.
        mode: derivatives modality, can be: `sobel` or `diff`.
        order: the order of the derivatives.
        normalized: whether the output is normalized.

    Return:
        the derivatives of the input feature map. with shape :math:`(B, C, 2, H, W)`.

    .. note::
       See a working example `here <https://kornia.github.io/tutorials/nbs/filtering_edges.html>`__.

    Examples:
        >>> input = torch.rand(1, 3, 4, 4)
        >>> output = spatial_gradient(input)  # 1x3x2x4x4
        >>> output.shape
        torch.Size([1, 3, 2, 4, 4])
    """
    KORNIA_CHECK_IS_TENSOR(input)
    KORNIA_CHECK_SHAPE(input, ['B', 'C', 'H', 'W'])

    # allocate kernel
    kernel = get_spatial_gradient_kernel2d(mode, order, dtype=input.dtype)
    if normalized:
        kernel = normalize_kernel2d(kernel)

    # prepare kernel
    b, c, h, w = input.shape
    tmp_kernel = kernel[:, None, ...]

    # Pad with "replicate for spatial dims, but with zeros for channel
    spatial_pad = [keras.ops.shape(kernel)[1] // 2, keras.ops.shape(kernel)[1] // 2, keras.ops.shape(kernel)[2] // 2, keras.ops.shape(kernel)[2] // 2]
    out_channels: int = 3 if order == 2 else 2
    padded_inp: IntegratedTensor = pad(keras.ops.reshape(input,(b * c, 1, h, w)), spatial_pad, 'replicate')
    out = keras.ops.conv(inputs=padded_inp, kernel=tmp_kernel, groups=1, padding=0, strides=1)
    return keras.ops.reshape(out,(b, c, out_channels, h, w))


def spatial_gradient3d(input: IntegratedTensor, mode: str = 'diff', order: int = 1) -> IntegratedTensor:
    r"""Compute the first and second order volume derivative in x, y and d using a diff operator.

    Args:
        input: input features tensor with shape :math:`(B, C, D, H, W)`.
        mode: derivatives modality, can be: `sobel` or `diff`.
        order: the order of the derivatives.

    Return:
        the spatial gradients of the input feature map with shape math:`(B, C, 3, D, H, W)`
        or :math:`(B, C, 6, D, H, W)`.

    Examples:
        >>> input = torch.rand(1, 4, 2, 4, 4)
        >>> output = spatial_gradient3d(input)
        >>> output.shape
        torch.Size([1, 4, 3, 2, 4, 4])
    """
    KORNIA_CHECK_IS_TENSOR(input)
    KORNIA_CHECK_SHAPE(input, ['B', 'C', 'D', 'H', 'W'])

    b, c, d, h, w = input.shape
    dtype = input.dtype
    if (mode == 'diff') and (order == 1):
        # we go for the special case implementation due to conv3d bad speed
        x: IntegratedTensor = pad(input, 6 * [1], 'replicate')
        center = slice(1, -1)
        left = slice(0, -2)
        right = slice(2, None)
        out = keras.ops.empty(b, c, 3, d, h, w, dtype=dtype)
        out[..., 0, :, :, :] = x[..., center, center, right] - x[..., center, center, left]
        out[..., 1, :, :, :] = x[..., center, right, center] - x[..., center, left, center]
        out[..., 2, :, :, :] = x[..., right, center, center] - x[..., left, center, center]
        out = 0.5 * out
    else:
        # prepare kernel
        # allocate kernel
        kernel = get_spatial_gradient_kernel3d(mode, order, dtype=dtype)

        tmp_kernel = kernel.repeat(c, 1, 1, 1, 1)

        # convolve input tensor with grad kernel
        kernel_flip = tmp_kernel.flip(-3)

        # Pad with "replicate for spatial dims, but with zeros for channel
        spatial_pad = [
            kernel.size(2) // 2,
            kernel.size(2) // 2,
            kernel.size(3) // 2,
            kernel.size(3) // 2,
            kernel.size(4) // 2,
            kernel.size(4) // 2,
        ]
        out_ch: int = 6 if order == 2 else 3
        out = keras.ops.conv(inputs=pad(input, spatial_pad, 'replicate'), kernel=kernel_flip, padding=0, groups=c)
        out = keras.ops.reshape(out,(b, c, out_ch, d, h, w))
    return out


def sobel(input: IntegratedTensor, normalized: bool = True, eps: float = 1e-6) -> IntegratedTensor:
    r"""Compute the Sobel operator and returns the magnitude per channel.

    .. image:: _static/img/sobel.png

    Args:
        input: the input image with shape :math:`(B,C,H,W)`.
        normalized: if True, L1 norm of the kernel is set to 1.
        eps: regularization number to avoid NaN during backprop.

    Return:
        the sobel edge gradient magnitudes map with shape :math:`(B,C,H,W)`.

    .. note::
       See a working example `here <https://kornia.github.io/tutorials/nbs/filtering_edges.html>`__.

    Example:
        >>> input = torch.rand(1, 3, 4, 4)
        >>> output = sobel(input)  # 1x3x4x4
        >>> output.shape
        torch.Size([1, 3, 4, 4])
    """
    KORNIA_CHECK_IS_TENSOR(input)
    KORNIA_CHECK_SHAPE(input, ['B', 'C', 'H', 'W'])

    # comput the x/y gradients
    edges: IntegratedTensor = spatial_gradient(input, normalized=normalized)

    # unpack the edges
    gx: IntegratedTensor = edges[:, :, 0]
    gy: IntegratedTensor = edges[:, :, 1]

    # compute gradient maginitude
    magnitude: IntegratedTensor = keras.ops.sqrt(gx * gx + gy * gy + eps)

    return magnitude


class SpatialGradient(Module):
    r"""Compute the first order image derivative in both x and y using a Sobel operator.

    Args:
        mode: derivatives modality, can be: `sobel` or `diff`.
        order: the order of the derivatives.
        normalized: whether the output is normalized.

    Return:
        the sobel edges of the input feature map.

    Shape:
        - Input: :math:`(B, C, H, W)`
        - Output: :math:`(B, C, 2, H, W)`

    Examples:
        >>> input = torch.rand(1, 3, 4, 4)
        >>> output = SpatialGradient()(input)  # 1x3x2x4x4
    """

    def __init__(self, mode: str = 'sobel', order: int = 1, normalized: bool = True) -> None:
        super().__init__()
        self.normalized: bool = normalized
        self.order: int = order
        self.mode: str = mode

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(order={self.order}, normalized={self.normalized}, mode={self.mode})"

    def call(self, input: IntegratedTensor) -> IntegratedTensor:
        return spatial_gradient(input, self.mode, self.order, self.normalized)


class SpatialGradient3d(Module):
    r"""Compute the first and second order volume derivative in x, y and d using a diff operator.

    Args:
        mode: derivatives modality, can be: `sobel` or `diff`.
        order: the order of the derivatives.

    Return:
        the spatial gradients of the input feature map.

    Shape:
        - Input: :math:`(B, C, D, H, W)`. D, H, W are spatial dimensions, gradient is calculated w.r.t to them.
        - Output: :math:`(B, C, 3, D, H, W)` or :math:`(B, C, 6, D, H, W)`

    Examples:
        >>> input = torch.rand(1, 4, 2, 4, 4)
        >>> output = SpatialGradient3d()(input)
        >>> output.shape
        torch.Size([1, 4, 3, 2, 4, 4])
    """

    def __init__(self, mode: str = 'diff', order: int = 1) -> None:
        super().__init__()
        self.order: int = order
        self.mode: str = mode
        self.kernel = get_spatial_gradient_kernel3d(mode, order)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(order={self.order}, mode={self.mode})"

    def call(self, input: IntegratedTensor) -> IntegratedTensor:
        return spatial_gradient3d(input, self.mode, self.order)


class Sobel(Module):
    r"""Compute the Sobel operator and returns the magnitude per channel.

    Args:
        normalized: if True, L1 norm of the kernel is set to 1.
        eps: regularization number to avoid NaN during backprop.

    Return:
        the sobel edge gradient magnitudes map.

    Shape:
        - Input: :math:`(B, C, H, W)`
        - Output: :math:`(B, C, H, W)`

    Examples:
        >>> input = torch.rand(1, 3, 4, 4)
        >>> output = Sobel()(input)  # 1x3x4x4
    """

    def __init__(self, normalized: bool = True, eps: float = 1e-6) -> None:
        super().__init__()
        self.normalized: bool = normalized
        self.eps: float = eps

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(normalized={self.normalized})"

    def call(self, input: IntegratedTensor) -> IntegratedTensor:
        return sobel(input, self.normalized, self.eps)
