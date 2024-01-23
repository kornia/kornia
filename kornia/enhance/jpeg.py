from typing import Optional, Tuple

import torch

from kornia.color import rgb_to_ycbcr, ycbcr_to_rgb
from kornia.core import Device, Dtype, Tensor, Module
from kornia.geometry import rescale
from kornia.core.check import (
    KORNIA_CHECK,
    KORNIA_CHECK_IS_TENSOR,
    KORNIA_CHECK_SHAPE,
)

QUANTIZATION_TABLE_Y: Tensor = torch.tensor(
    [
        [16, 11, 10, 16, 24, 40, 51, 61],
        [12, 12, 14, 19, 26, 58, 60, 55],
        [14, 13, 16, 24, 40, 57, 69, 56],
        [14, 17, 22, 29, 51, 87, 80, 62],
        [18, 22, 37, 56, 68, 109, 103, 77],
        [24, 35, 55, 64, 81, 104, 113, 92],
        [49, 64, 78, 87, 103, 121, 120, 101],
        [72, 92, 95, 98, 112, 100, 103, 99],
    ],
    dtype=torch.float,
)

QUANTIZATION_TABLE_C: Tensor = torch.tensor(
    [
        [17, 18, 24, 47, 99, 99, 99, 99],
        [18, 21, 26, 66, 99, 99, 99, 99],
        [24, 26, 56, 99, 99, 99, 99, 99],
        [47, 66, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
    ],
    dtype=torch.float,
)


def _patchify_8x8(input: Tensor) -> Tensor:
    """Function extracts non-overlapping 8 x 8 patches from the given input image.

    Args:
        input (Tensor): Input image of the shape :math:`(B, H, W)`.

    Returns:
        output (Tensor): Image patchify of the shape :math:`(B, N, 8, 8)`.
    """
    # Get input shape
    B, H, W = input.shape
    # Patchify to shape [B, N, H // 8, W // 8]
    output: Tensor = input.view(B, H // 8, 8, W // 8, 8).permute(0, 1, 3, 2, 4).reshape(B, -1, 8, 8)
    return output


def _unpatchify_8x8(input: Tensor, H: int, W: int) -> Tensor:
    """Function reverses non-overlapping 8 x 8 patching.

    Args:
        input (Tensor): Input image of the shape :math:`(B, N, 8, 8)`.

    Returns:
        output (Tensor): Image patchify of the shape :math:`(B, H, W)`.
    """
    # Get input shape
    B, N = input.shape[:2]
    # Unpatch to [B, H, W]
    output: Tensor = input.view(B, H // 8, W // 8, 8, 8).permute(0, 1, 3, 2, 4).reshape(B, H, W)
    return output


def _dct_8x8(input: Tensor) -> Tensor:
    """Performs an 8 x 8 discrete cosine transform.

    Args:
        input (Tensor): Patched input tensor of the shape :math:`(B, N, 8, 8)`.

    Returns:
        output (Tensor): DCT output tensor of the shape :math:`(B, N, 8, 8)`.
    """
    # Get dtype and device
    dtype: Dtype = input.dtype
    device: Device = input.device
    # Make DCT tensor and scaling
    index: Tensor = torch.arange(8, dtype=dtype, device=device)
    x, y, u, v = torch.meshgrid(index, index, index, index)
    dct_tensor: Tensor = ((2.0 * x + 1.0) * u * torch.pi / 16.0).cos() * ((2.0 * y + 1.0) * v * torch.pi / 16.0).cos()
    alpha: Tensor = torch.ones(8, dtype=dtype, device=device)
    alpha[0] = 1.0 / (2**0.5)
    dct_scale: Tensor = torch.einsum("i, j -> ij", alpha, alpha) * 0.25
    # Apply DCT
    output: Tensor = dct_scale[None, None] * torch.tensordot(input - 128.0, dct_tensor)
    return output


def _idct_8x8(input: Tensor) -> Tensor:
    """Performs an 8 x 8 discrete cosine transform.

    Args:
        input (Tensor): Patched input tensor of the shape :math:`(B, N, 8, 8)`.

    Returns:
        output (Tensor): DCT output tensor of the shape :math:`(B, N, 8, 8)`.
    """
    # Get dtype and device
    dtype: Dtype = input.dtype
    device: Device = input.device
    # Make and apply scaling
    alpha: Tensor = torch.ones(8, dtype=dtype, device=device)
    alpha[0] = 1.0 / (2**0.5)
    dct_scale: Tensor = torch.outer(alpha, alpha)
    input = input * dct_scale[None, None]
    # Make DCT tensor and scaling
    index: Tensor = torch.arange(8, dtype=dtype, device=device)
    x, y, u, v = torch.meshgrid(index, index, index, index)
    idct_tensor: Tensor = ((2.0 * u + 1.0) * x * torch.pi / 16.0).cos() * ((2.0 * v + 1.0) * y * torch.pi / 16.0).cos()
    # Apply DCT
    output: Tensor = 0.25 * torch.tensordot(input, idct_tensor, dims=2) + 128.0
    return output


def _differentiable_polynomial_rounding(input: Tensor) -> Tensor:
    """This function implements differentiable rounding.

    Args:
        input (Tensor): Input tensor of any shape to be rounded.

    Returns:
        output (Tensor): Pseudo rounded tensor of the same shape as input tensor.
    """
    # Perform differentiable rounding
    output: Tensor = torch.round(input) + (input - torch.round(input)) ** 3
    return output


def _differentiable_polynomial_floor(input: Tensor) -> Tensor:
    """This function implements differentiable floor.

    Args:
        input (Tensor): Input tensor of any shape to be floored.

    Returns:
        output (Tensor): Pseudo rounded tensor of the same shape as input tensor.
    """
    # Perform differentiable rounding
    output: Tensor = torch.floor(input) + (input - 0.5 - torch.floor(input)) ** 3
    return output


def _differentiable_clipping(
    input: Tensor,
    min: Optional[float] = None,
    max: Optional[float] = None,
    scale: float = 0.02,
) -> Tensor:
    """This function implements a differentiable and soft approximation of the clipping operation.

    Args:
        input (Tensor): Input tensor of any shape.
        min (Optional[float]): Minimum value.
        max (Optional[float]): Maximum value.
        scale (float): Scale value. Default 0.02.

    Returns:
        output (Tensor): Clipped output tensor of the same shape as the input tensor.
    """
    # Make a copy of the input tensor
    output: Tensor = input.clone()
    # Perform differentiable soft clipping
    if max is not None:
        output = torch.where(output > max, -scale * (torch.exp(-output + max) - 1.0) + max, output)
    if min is not None:
        output = torch.where(output < min, scale * (torch.exp(output - min) - 1.0) + min, output)
    return output


def _jpeg_quality_to_scale(
    compression_strength: Tensor,
) -> Tensor:
    """Converts a given JPEG quality to the scaling factor.

    Args:
        compression_strength (Tensor): Compression strength ranging from 0 to 100. Any shape is supported.

    Returns:
        scale (Tensor): Scaling factor to be applied to quantization matrix. Same shape as input.
    """
    # Get scale
    scale: Tensor = _differentiable_polynomial_floor(
        torch.where(compression_strength < 50, 5000.0 / compression_strength, 200.0 - 2.0 * compression_strength)
    )
    return scale


def _quantize(
    input: Tensor,
    jpeg_quality: Tensor,
    quantization_table: Tensor,
) -> Tensor:
    """Function performs quantization.

    Args:
        input (Tensor): Input tensor of the shape :math:`(B, N, 8, 8)`.
        jpeg_quality (Tensor): Compression strength to be applied, shape is :math:`(B)`.
        quantization_table (Tensor): Quantization table of the shape :math:`(1, 8, 8)` or :math:`(B, 8, 8)`.

    Returns:
        output (Tensor): Quantized output tensor of the shape :math:`(B, N, 8, 8)`.
    """
    # Scale quantization table
    quantization_table_scaled: Tensor = (
        quantization_table[:, None] * _jpeg_quality_to_scale(jpeg_quality)[:, None, None, None]
    )
    # Perform scaling
    quantization_table = _differentiable_polynomial_floor(
        _differentiable_clipping((quantization_table_scaled + 50.0) / 100.0, 1, 255)
    )
    output: Tensor = input / quantization_table
    # Perform rounding
    output = _differentiable_polynomial_rounding(output)
    return output


def _dequantize(
    input: Tensor,
    jpeg_quality: Tensor,
    quantization_table: Tensor,
) -> Tensor:
    """Function performs dequantization.

    Args:
        input (Tensor): Input tensor of the shape :math:`(B, N, 8, 8)`.
        jpeg_quality (Tensor): Compression strength to be applied, shape is :math:`(B)`.
        quantization_table (Tensor): Quantization table of the shape :math:`(1, 8, 8)` or :math:`(B, 8, 8)`.

    Returns:
        output (Tensor): Quantized output tensor of the shape :math:`(B, N, 8, 8)`.
    """
    # Scale quantization table
    quantization_table_scaled: Tensor = (
        quantization_table[:, None] * _jpeg_quality_to_scale(jpeg_quality)[:, None, None, None]
    )
    # Perform scaling
    output: Tensor = input * _differentiable_polynomial_floor(
        _differentiable_clipping((quantization_table_scaled + 50.0) / 100.0, 1, 255)
    )
    return output


def _chroma_subsampling(input_ycbcr: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
    """This function implements chroma subsampling.

    Args:
        input_ycbcr (Tensor): YCbCr input tensor of the shape :math:`(B, 3, H, W)`.

    Returns:
        output_y (Tensor): Y component (not-subsampled), shape is :math:`(B, H, W)`.
        output_cb (Tensor): Cb component (subsampled), shape is :math:`(B, H // 2, W // 2)`.
        output_cr (Tensor): Cr component (subsampled), shape is :math:`(B, H // 2, W // 2)`.
    """
    # Get components
    output_y: Tensor = input_ycbcr[:, 0]
    output_cb: Tensor = input_ycbcr[:, 1]
    output_cr: Tensor = input_ycbcr[:, 2]
    # Perform average pooling of Cb and Cr channels
    output_cb = rescale(output_cb[:, None], factor=0.5, interpolation="bilinear", align_corners=False, antialias=True)
    output_cr = rescale(output_cr[:, None], factor=0.5, interpolation="bilinear", align_corners=False, antialias=True)
    return output_y, output_cb[:, 0], output_cr[:, 0]


def _chroma_upsampling(input_c: Tensor) -> Tensor:
    """Function performs chroma upsampling.

    Args:
        input_c (Tensor): Cb or Cr component to be upsampled of the shape :math:`(B, H, W)`.

    Returns:
        output_c (Tensor): Upsampled C(b or r) component of the shape :math:`(B, H * 2, W * 2)`.
    """
    # Upsample component
    output_c: Tensor = rescale(
        input_c[:, None], factor=2.0, interpolation="bilinear", align_corners=False, antialias=False
    )
    return output_c[:, 0]


def _jpeg_encode(
    image_rgb: Tensor,
    jpeg_quality: Tensor,
    quantization_table_y: Tensor,
    quantization_table_c: Tensor,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Performs JPEG encoding.

    Args:
        image_rgb (Tensor): RGB input images of the shape :math:`(B, 3, H, W)`.
        jpeg_quality (Tensor): Compression strength of the shape :math:`(B)`.
        quantization_table_y (Tensor): Quantization table for Y channel.
        quantization_table_c (Tensor): Quantization table for C channels.

    Returns:
        y_encoded (Tensor): Encoded Y component of the shape :math:`(B, N, 8, 8)`.
        cb_encoded (Tensor): Encoded Cb component of the shape :math:`(B, N, 8, 8)`.
        cr_encoded (Tensor): Encoded Cr component of the shape :math:`(B, N, 8, 8)`.
    """
    # Convert RGB image to YCbCr.
    image_ycbcr: Tensor = rgb_to_ycbcr(image_rgb)
    # Scale pixel-range to [0, 255]
    image_ycbcr = 255.0 * image_ycbcr
    # Perform chroma subsampling
    input_y, input_cb, input_cr = _chroma_subsampling(image_ycbcr)
    # Patchify, DCT, and rounding
    input_y, input_cb, input_cr = _patchify_8x8(input_y), _patchify_8x8(input_cb), _patchify_8x8(input_cr)
    dct_y, dct_cb, dct_cr = _dct_8x8(input_y), _dct_8x8(input_cb), _dct_8x8(input_cr)
    y_encoded: Tensor = _quantize(
        dct_y,
        jpeg_quality,
        quantization_table_y,
    )
    cb_encoded: Tensor = _quantize(
        dct_cb,
        jpeg_quality,
        quantization_table_c,
    )
    cr_encoded: Tensor = _quantize(
        dct_cr,
        jpeg_quality,
        quantization_table_c,
    )
    return y_encoded, cb_encoded, cr_encoded


def _jpeg_decode(
    input_y: Tensor,
    input_cb: Tensor,
    input_cr: Tensor,
    jpeg_quality: Tensor,
    H: int,
    W: int,
    quantization_table_y: Tensor,
    quantization_table_c: Tensor,
) -> Tensor:
    """Performs JPEG decoding.

    Args:
        input_y (Tensor): Compressed Y component of the shape :math:`(B, N, 8, 8)`.
        input_cb (Tensor): Compressed Cb component of the shape :math:`(B, N, 8, 8)`.
        input_cr (Tensor): Compressed Cr component of the shape :math:`(B, N, 8, 8)`.
        jpeg_quality (Tensor): Compression strength of the shape :math:`(B)`.
        H (int): Original image height.
        W (int): Original image width.
        quantization_table_y (Tensor): Quantization table for Y channel.
        quantization_table_c (Tensor): Quantization table for C channels.

    Returns:
        rgb_decoded (Tensor): Decompressed RGB image of the shape :math:`(B, 3, H, W)`.
    """
    # Dequantize inputs
    input_y = _dequantize(
        input_y,
        jpeg_quality,
        quantization_table_y,
    )
    input_cb = _dequantize(
        input_cb,
        jpeg_quality,
        quantization_table_c,
    )
    input_cr = _dequantize(
        input_cr,
        jpeg_quality,
        quantization_table_c,
    )
    # Perform inverse DCT
    idct_y: Tensor = _idct_8x8(input_y)
    idct_cb: Tensor = _idct_8x8(input_cb)
    idct_cr: Tensor = _idct_8x8(input_cr)
    # Reverse patching
    image_y: Tensor = _unpatchify_8x8(idct_y, H, W)
    image_cb: Tensor = _unpatchify_8x8(idct_cb, H // 2, W // 2)
    image_cr: Tensor = _unpatchify_8x8(idct_cr, H // 2, W // 2)
    # Perform chroma upsampling
    image_cb = _chroma_upsampling(image_cb)
    image_cr = _chroma_upsampling(image_cr)
    # Back to [0, 1] pixel-range
    image_ycbcr: Tensor = torch.stack((image_y, image_cb, image_cr), dim=1) / 255.0
    # Convert back to RGB space.
    rgb_decoded: Tensor = ycbcr_to_rgb(image_ycbcr)
    return rgb_decoded


def diff_jpeg(
    image_rgb: Tensor,
    jpeg_quality: Tensor,
    quantization_table_y: Tensor = QUANTIZATION_TABLE_Y,
    quantization_table_c: Tensor = QUANTIZATION_TABLE_C,
) -> Tensor:
    r"""Differentiable JPEG encoding-decoding module.

    Based on [1, 2], we perform differentiable JPEG encoding-decoding as follows:

    .. math::

        \text{JPEG}_{\text{diff}}(I, q, QT_{y}, QT_{c}) = \hat{I}

    Where:
       - :math:`I` is the original image to be coded.
       - :math:`q` is the JPEG quality controling the compression strength.
       - :math:`QT_{y}` is the luma quantization table.
       - :math:`QT_{c}` is the chroma quantization table.
       - :math:`\hat{I}` is the resulting JPEG encoded-decoded image.

    Notes:
        The input (and output) pixel range is :math:`[0, 1]`. In case you want to handle normalized images you are
        required to first perform denormalization followed by normalizing the output images again.
        Note, this implementation models the encoding-decoding mapping of JPEG in a differentiable setting,
        howerver, does not allow to excess the JPEG-coded byte file itself. For more details please refer to [1].

    Reference:
        [1] https://arxiv.org/abs/2309.06978
        [2] https://machine-learning-and-security.github.io/papers/mlsec17_paper_54.pdf

    Args:
        image_rgb: the RGB image to be coded.
        jpeg_quality: JPEG quality in the range :math:`[0, 99]` controling the compression strength.
        quantization_table_y: quantization table for Y channel. Default: standard quantization table.
        quantization_table_c: quantization table for C channels. Default: standard quantization table.

    Shape:
        - image_rgb: :math:`(B, 3, H, W)`.
        - jpeg_quality: :math:`(1)` or :math:`(B)`.
        - quantization_table_y: :math:`(8, 8)` or :math:`(B, 8, 8)`.
        - quantization_table_c: :math:`(8, 8)` or :math:`(B, 8, 8)`.

    Return:
        JPEG coded image of the shape :math:`(B, 3, H, W)`

    Example:
        >>> img = torch.rand(3, 3, 64, 64, requires_grad=True, dtype=torch.float)
        >>> jpeg_quality = torch.tensor((99.0, 25.0, 1.0), requires_grad=True)
        >>> img_jpeg = diff_jpeg(img, jpeg_quality)
        >>> img_jpeg.sum().backward()
    """
    # Check that inputs are tensors
    KORNIA_CHECK_IS_TENSOR(image_rgb)
    KORNIA_CHECK_IS_TENSOR(jpeg_quality)
    KORNIA_CHECK_IS_TENSOR(quantization_table_y)
    KORNIA_CHECK_IS_TENSOR(quantization_table_c)
    # Check shape of inputs
    KORNIA_CHECK_SHAPE(image_rgb, ["B", "3", "H", "W"])
    KORNIA_CHECK(
        (image_rgb.shape[2] % 16 == 0) and (image_rgb.shape[3] % 16 == 0),
        f"image dimension must be divisible by 16. Got the shape {image_rgb.shape}.",
    )
    KORNIA_CHECK_SHAPE(jpeg_quality, ["B"])
    # Add batch dimension to quantization tables if needed
    if quantization_table_y.ndim == 2:
        quantization_table_y = quantization_table_y.unsqueeze(dim=0)
    if quantization_table_c.ndim == 2:
        quantization_table_c = quantization_table_c.unsqueeze(dim=0)
    # Check resulting shape of quantization tables
    KORNIA_CHECK_SHAPE(quantization_table_y, ["B", "8", "8"])
    KORNIA_CHECK_SHAPE(quantization_table_c, ["B", "8", "8"])
    # Check value range of JPEG quality
    KORNIA_CHECK(
        (jpeg_quality.amin().item() >= 0.0) and (jpeg_quality.amax().item() <= 99.0),
        f"JPEG quality is out of range. Expected range is [0, 99], "
        f"got [{jpeg_quality.amin().item()}, {jpeg_quality.amax().item()}]. Consider clipping jpeg_quality.",
    )
    # Get original shape
    H, W = image_rgb.shape[2:]
    # Get device and dtype
    dtype: Dtype = image_rgb.dtype
    device: Device = image_rgb.device
    # Quantization tables to same device and dtype as input image
    quantization_table_y = quantization_table_y.to(dtype).to(device)
    quantization_table_c = quantization_table_c.to(dtype).to(device)
    # Perform encoding
    y_encoded, cb_encoded, cr_encoded = _jpeg_encode(
        image_rgb=image_rgb,
        jpeg_quality=jpeg_quality,
        quantization_table_c=quantization_table_c,
        quantization_table_y=quantization_table_y,
    )
    image_rgb_jpeg: Tensor = _jpeg_decode(
        input_y=y_encoded,
        input_cb=cb_encoded,
        input_cr=cr_encoded,
        jpeg_quality=jpeg_quality,
        H=H,
        W=W,
        quantization_table_c=quantization_table_c,
        quantization_table_y=quantization_table_y,
    )
    # Clip coded image
    image_rgb_jpeg = _differentiable_clipping(input=image_rgb_jpeg, min=0.0, max=255.0)
    return image_rgb_jpeg


class DiffJPEG(Module):
    r"""Differentiable JPEG encoding-decoding module.

    Based on [1, 2], we perform differentiable JPEG encoding-decoding as follows:

    .. math::

        \text{JPEG}_{\text{diff}}(I, q, QT_{y}, QT_{c}) = \hat{I}

    Where:
       - :math:`I` is the original image to be coded.
       - :math:`q` is the JPEG quality controling the compression strength.
       - :math:`QT_{y}` is the luma quantization table.
       - :math:`QT_{c}` is the chroma quantization table.
       - :math:`\hat{I}` is the resulting JPEG encoded-decoded image.

    Notes:
        The input (and output) pixel range is :math:`[0, 1]`. In case you want to handle normalized images you are
        required to first perform denormalization followed by normalizing the output images again.
        Note, this implementation models the encoding-decoding mapping of JPEG in a differentiable setting,
        howerver, does not allow to excess the JPEG-coded byte file itself. For more details please refer to [1].

    Reference:
        [1] https://arxiv.org/abs/2309.06978
        [2] https://machine-learning-and-security.github.io/papers/mlsec17_paper_54.pdf

    Args:
        image_rgb: the RGB image to be coded.
        jpeg_quality: JPEG quality in the range :math:`[0, 99]` controling the compression strength.
        quantization_table_y: quantization table for Y channel. Default: standard quantization table.
        quantization_table_c: quantization table for C channels. Default: standard quantization table.

    Shape:
        - image_rgb: :math:`(B, 3, H, W)`.
        - jpeg_quality: :math:`(1)` or :math:`(B)`.
        - quantization_table_y: :math:`(8, 8)` or :math:`(B, 8, 8)`.
        - quantization_table_c: :math:`(8, 8)` or :math:`(B, 8, 8)`.

    Example:
        >>> diff_jpeg_module = DiffJPEG()
        >>> img = torch.rand(2, 3, 32, 32, requires_grad=True, dtype=torch.float)
        >>> jpeg_quality = torch.tensor((99.0, 1.0), requires_grad=True)
        >>> img_jpeg = diff_jpeg_module(img, jpeg_quality)
        >>> img_jpeg.sum().backward()
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(
        self,
        image_rgb: Tensor,
        jpeg_quality: Tensor,
        quantization_table_y: Tensor = QUANTIZATION_TABLE_Y,
        quantization_table_c: Tensor = QUANTIZATION_TABLE_C,
    ) -> Tensor:
        # Perform encoding-decoding
        image_rgb_jpeg: Tensor = diff_jpeg(
            image_rgb=image_rgb,
            jpeg_quality=jpeg_quality,
            quantization_table_c=quantization_table_c,
            quantization_table_y=quantization_table_y,
        )
        return image_rgb_jpeg
