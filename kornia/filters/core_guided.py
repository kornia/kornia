from __future__ import annotations

from kornia.core import Module, IntegratedTensor
from kornia.core.check import KORNIA_CHECK, KORNIA_CHECK_IS_TENSOR, KORNIA_CHECK_SHAPE

from .blur import box_blur
from .kernels import _unpack_2d_ks

import keras_core as keras

def _interpolate(input: IntegratedTensor, scale: float, mode: str):
    result = keras.ops.image.resize(input,((IntegratedTensor.height*scale),(IntegratedTensor.width*scale)),interpolation=mode)
    return result
def _preprocess_fast_guided_blur(
    guidance: IntegratedTensor, input: IntegratedTensor, kernel_size: tuple[int, int] | int, subsample: int = 1
) -> tuple[IntegratedTensor, IntegratedTensor, tuple[int, int]]:
    ky, kx = _unpack_2d_ks(kernel_size)
    if subsample > 1:
        s = 1 / subsample
        guidance_sub = _interpolate(guidance, scale_factor=s, mode="nearest")
        input_sub = guidance_sub if input is guidance else _interpolate(input, scale_factor=s, mode="nearest")
        ky, kx = ((k - 1) // subsample + 1 for k in (ky, kx))
    else:
        guidance_sub = guidance
        input_sub = input
    return guidance_sub, input_sub, (ky, kx)


def _guided_blur_grayscale_guidance(
    guidance: IntegratedTensor,
    input: IntegratedTensor,
    kernel_size: tuple[int, int] | int,
    eps: float | IntegratedTensor,
    border_type: str = 'reflect',
    subsample: int = 1,
) -> IntegratedTensor:
    guidance_sub, input_sub, kernel_size = _preprocess_fast_guided_blur(guidance, input, kernel_size, subsample)

    mean_I = box_blur(guidance_sub, kernel_size, border_type)
    corr_I = box_blur((guidance_sub*guidance_sub), kernel_size, border_type)
    var_I = corr_I - (mean_I*mean_I)

    if input is guidance:
        mean_p = mean_I
        cov_Ip = var_I

    else:
        mean_p = box_blur(input_sub, kernel_size, border_type)
        corr_Ip = box_blur(guidance_sub * input_sub, kernel_size, border_type)
        cov_Ip = corr_Ip - mean_I * mean_p

    if isinstance(eps, IntegratedTensor):
        eps = keras.ops.reshape(eps,(-1, 1, 1, 1))  # N -> NCHW      

    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I

    mean_a = box_blur(a, kernel_size, border_type)
    mean_b = box_blur(b, kernel_size, border_type)

    if subsample > 1:
        mean_a = _interpolate(mean_a, scale_factor=subsample, mode="bilinear")
        mean_b = _interpolate(mean_b, scale_factor=subsample, mode="bilinear")

    return mean_a * guidance + mean_b


def _guided_blur_multichannel_guidance(
    guidance: IntegratedTensor,
    input: IntegratedTensor,
    kernel_size: tuple[int, int] | int,
    eps: float | IntegratedTensor,
    border_type: str = 'reflect',
    subsample: int = 1,
) -> IntegratedTensor:
    guidance_sub, input_sub, kernel_size = _preprocess_fast_guided_blur(guidance, input, kernel_size, subsample)
    B, C, H, W = guidance_sub.shape

    mean_I = keras.ops.transpose(box_blur(guidance_sub, kernel_size, border_type),axes=(0, 2, 3, 1))
    II = (keras.ops.reshape(*guidance_sub.shape[:0],1,*guidance_sub.shape[1:]) * keras.ops.reshape(*guidance_sub.shape[:1],1,*guidance_sub.shape[2:]))
    II = keras.ops.reshape(*II.shape[:1], -1, *II.shape[2+1:]) #https://stackoverflow.com/questions/57234095/what-is-the-difference-between-flatten-and-view-1-in-pytorch
    corr_I = box_blur(II, kernel_size, border_type)
    corr_I = keras.ops.transpose(corr_I,axes=(0, 2, 3, 1))
    var_I = keras.ops.reshape(corr_I,(B, H, W, C, C)) - keras.ops.expand_dims(mean_I, -2) * keras.ops.expand_dims(mean_I, axis=-1)

    if guidance is input:
        mean_p = mean_I
        cov_Ip = var_I

    else:
        mean_p = keras.ops.transpose(box_blur(input_sub, kernel_size, border_type),axes=(0, 2, 3, 1))
        Ip = keras.ops.expand_dims(input_sub, 1) * input_sub
        Ip = keras.ops.reshape(*Ip.shape[:1], -1, *Ip.shape[2+1:])
        corr_Ip = box_blur(Ip, kernel_size, border_type)
        corr_Ip = keras.ops.transpose(corr_Ip,axes=(0, 2, 3, 1))
        cov_Ip = keras.ops.reshape(corr_Ip,(B, H, W, C, -1)) - keras.ops.expand_dims(mean_p, -2) * keras.ops.expand_dims(mean_I, -1)

    if isinstance(eps, IntegratedTensor):
        _eps = keras.ops.eye(C, dtype=guidance.dtype)                   
        _eps = keras.ops.reshape(_eps,(1, 1, 1, C, C)) * keras.ops.reshape(_eps,(-1, 1, 1, 1, 1)) 
    else:
        _eps = keras.ops.full((C,), eps)
        _eps = keras.ops.diag(_eps)
        _eps = keras.ops.reshape(_eps,(1, 1, 1, C, C))
    a = (var_I + _eps).linalg_solve(cov_Ip)  # B, H, W, C_guidance, C_input   
    b = mean_p - (keras.ops.expand_dims(mean_I, -2) @ a)
    b = keras.ops.reshape(*II.shape[:1], -1, *II.shape[2+1:])
    b = keras.ops.squeeze(b,axis=-2)  # B, H, W, C_input

    mean_a = box_blur(keras.ops.transpose(keras.ops.reshape(a, -2),axes=(0, 3, 1, 2)), kernel_size, border_type)
    mean_b = box_blur(keras.ops.transpose(b,axes=(0, 3, 1, 2)), kernel_size, border_type)

    if subsample > 1:
        mean_a = _interpolate(mean_a, scale_factor=subsample, mode="bilinear")
        mean_b = _interpolate(mean_b, scale_factor=subsample, mode="bilinear")
    mean_a = keras.ops.reshape(mean_a,(B, C, -1, H * subsample, W * subsample))

    # einsum might not be contiguous, thus mean_b is the first argument
    return mean_b + keras.ops.einsum("BCHW,BCcHW->BcHW", guidance, mean_a)


def guided_blur(
    guidance: IntegratedTensor,
    input: IntegratedTensor,
    kernel_size: tuple[int, int] | int,
    eps: float | IntegratedTensor,
    border_type: str = 'reflect',
    subsample: int = 1,
) -> IntegratedTensor:
    r"""Blur a tensor using a Guided filter.

    .. image:: _static/img/guided_blur.png

    The operator is an edge-preserving image smoothing filter. See :cite:`he2010guided`
    and :cite:`he2015fast` for details. Guidance and input can have different number of channels.

    Arguments:
        guidance: the guidance tensor with shape :math:`(B,C,H,W)`.
        input: the input tensor with shape :math:`(B,C,H,W)`.
        kernel_size: the size of the kernel.
        eps: regularization parameter. Smaller values preserve more edges.
        border_type: the padding mode to be applied before convolving.
          The expected modes are: ``'constant'``, ``'reflect'``,
          ``'replicate'`` or ``'circular'``. Default: ``'reflect'``.
        subsample: subsampling factor for Fast Guided filtering. Default: 1 (no subsampling)

    Returns:
        the blurred tensor with same shape as `input` :math:`(B, C, H, W)`.

    Examples:
        >>> guidance = torch.rand(2, 3, 5, 5)
        >>> input = torch.rand(2, 4, 5, 5)
        >>> output = guided_blur(guidance, input, 3, 0.1)
        >>> output.shape
        torch.Size([2, 4, 5, 5])
    """

    KORNIA_CHECK_IS_TENSOR(guidance)
    KORNIA_CHECK_SHAPE(guidance, ['B', 'C', 'H', 'W'])
    if input is not guidance:
        KORNIA_CHECK_IS_TENSOR(input)
        KORNIA_CHECK_SHAPE(input, ['B', 'C', 'H', 'W'])
        KORNIA_CHECK(
            (guidance.shape[0] == input.shape[0]) and (guidance.shape[-2:] == input.shape[-2:]),
            "guidance and input should have the same batch size and spatial dimensions",
        )

    if guidance.shape[1] == 1:
        return _guided_blur_grayscale_guidance(guidance, input, kernel_size, eps, border_type, subsample)
    else:
        return _guided_blur_multichannel_guidance(guidance, input, kernel_size, eps, border_type, subsample)


class GuidedBlur(Module):
    r"""Blur a tensor using a Guided filter.

    The operator is an edge-preserving image smoothing filter. See :cite:`he2010guided`
    and :cite:`he2015fast` for details. Guidance and input can have different number of channels.

    Arguments:
        kernel_size: the size of the kernel.
        eps: regularization parameter. Smaller values preserve more edges.
        border_type: the padding mode to be applied before convolving.
          The expected modes are: ``'constant'``, ``'reflect'``,
          ``'replicate'`` or ``'circular'``. Default: ``'reflect'``.
        subsample: subsampling factor for Fast Guided filtering. Default: 1 (no subsampling)

    Returns:
        the blurred input tensor.

    Shape:
        - Input: :math:`(B, C, H, W)`, :math:`(B, C, H, W)`
        - Output: :math:`(B, C, H, W)`

    Examples:
        >>> guidance = torch.rand(2, 3, 5, 5)
        >>> input = torch.rand(2, 4, 5, 5)
        >>> blur = GuidedBlur(3, 0.1)
        >>> output = blur(guidance, input)
        >>> output.shape
        torch.Size([2, 4, 5, 5])
    """

    def __init__(
        self, kernel_size: tuple[int, int] | int, eps: float, border_type: str = 'reflect', subsample: int = 1
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.eps = eps
        self.border_type = border_type
        self.subsample = subsample

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}"
            f"(kernel_size={self.kernel_size}, "
            f"eps={self.eps}, "
            f"border_type={self.border_type}, "
            f"subsample={self.subsample})"
        )

    def call(self, guidance: IntegratedTensor, input: IntegratedTensor) -> IntegratedTensor:
        return guided_blur(guidance, input, self.kernel_size, self.eps, self.border_type, self.subsample)
