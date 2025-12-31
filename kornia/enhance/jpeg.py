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

from __future__ import annotations

import math
from typing import Union

import torch
import torch.nn.functional as F
from torch import nn

from kornia.color import rgb_to_ycbcr, ycbcr_to_rgb
from kornia.constants import pi
from kornia.core.check import (
    KORNIA_CHECK,
    KORNIA_CHECK_IS_TENSOR,
    KORNIA_CHECK_SHAPE,
)
from kornia.geometry.transform.affwarp import rescale
from kornia.utils.image import perform_keep_shape_image
from kornia.utils.misc import (
    differentiable_clipping,
    differentiable_polynomial_floor,
    differentiable_polynomial_rounding,
)

_DCT8_CACHE: dict[
    tuple[Union[torch.dtype, None], Union[str, torch.device, None]], tuple[torch.Tensor, torch.Tensor]
] = {}

__all__ = ["JPEGCodecDifferentiable", "jpeg_codec_differentiable"]


def _get_default_qt_y(device: Union[str, torch.device, None], dtype: Union[torch.dtype, None]) -> torch.Tensor:
    """Generate default Quantization table of Y channel."""
    return torch.tensor(
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
        device=device,
        dtype=dtype,
    )


def _get_default_qt_c(device: Union[str, torch.device, None], dtype: Union[torch.dtype, None]) -> torch.Tensor:
    """Generate default Quantization table of C channels."""
    return torch.tensor(
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
        device=device,
        dtype=dtype,
    )


def _patchify_8x8(input: torch.Tensor) -> torch.Tensor:
    """Extract non-overlapping 8 x 8 patches from the given input image.

    Args:
        input (torch.Tensor): Input image of the shape :math:`(B, H, W)`.

    Returns:
        output (torch.Tensor): Image patchify of the shape :math:`(B, N, 8, 8)`.

    """
    # Get input shape
    B, H, W = input.shape
    # Patchify to shape [B, N, H // 8, W // 8]
    output: torch.Tensor = input.view(B, H // 8, 8, W // 8, 8).permute(0, 1, 3, 2, 4).reshape(B, -1, 8, 8)
    return output


def _unpatchify_8x8(input: torch.Tensor, H: int, W: int) -> torch.Tensor:
    """Reverse non-overlapping 8 x 8 patching.

    Args:
        input (torch.Tensor): Input image of the shape :math:`(B, N, 8, 8)`.
        H: height of resulting torch.tensor.
        W: width of resulting torch.tensor.

    Returns:
        output (torch.Tensor): Image patchify of the shape :math:`(B, H, W)`.

    """
    # Get input shape
    B, _N = input.shape[:2]
    # Unpatch to [B, H, W]
    output: torch.Tensor = input.view(B, H // 8, W // 8, 8, 8).permute(0, 1, 3, 2, 4).reshape(B, H, W)
    return output


def _dct_8x8(input: torch.Tensor) -> torch.Tensor:
    """Perform an 8 x 8 discrete cosine transform.

    Args:
        input (torch.Tensor): Patched input torch.tensor of the shape :math:`(B, N, 8, 8)`.

    Returns:
        output (torch.Tensor): DCT output torch.tensor of the shape :math:`(B, N, 8, 8)`.

    """
    # Get dtype and device
    dtype: Union[torch.dtype, None] = input.dtype
    device: Union[str, torch.device, None] = input.device
    dct_tensor, dct_scale = _get_dct8_basis_scale(dtype, device)
    # Apply DCT
    output: torch.Tensor = dct_scale[None, None] * torch.tensordot(input - 128.0, dct_tensor)
    return output


def _idct_8x8(input: torch.Tensor) -> torch.Tensor:
    """Perform an 8 x 8 inverse discrete cosine transform.

    Args:
        input (torch.Tensor): Patched input torch.tensor of the shape :math:`(B, N, 8, 8)`.

    Returns:
        output (torch.Tensor): IDCT output torch.tensor of the shape :math:`(B, N, 8, 8)`.

    """
    dtype = input.dtype
    device = input.device

    idx = torch.arange(8, dtype=dtype, device=device)
    spatial_idx = idx.unsqueeze(0)
    freq_idx = idx.unsqueeze(1)

    basis = torch.cos((2.0 * spatial_idx + 1.0) * freq_idx * pi / 16.0)
    alpha = torch.ones(8, dtype=dtype, device=device)
    alpha[0] = 1.0 / (2**0.5)
    dct_scale = torch.outer(alpha, alpha)
    input = input * dct_scale

    tmp = input @ basis
    output = (tmp.transpose(-1, -2) @ basis).transpose(-1, -2)

    output = output * 0.25 + 128.0
    return output


def _jpeg_quality_to_scale(
    compression_strength: torch.Tensor,
) -> torch.Tensor:
    """Convert a given JPEG quality to the scaling factor.

    Args:
        compression_strength (torch.Tensor): Compression strength ranging from 0 to 100. Any shape is supported.

    Returns:
        scale (torch.Tensor): Scaling factor to be applied to quantization matrix. Same shape as input.

    """
    # Get scale
    scale: torch.Tensor = differentiable_polynomial_floor(
        torch.where(
            compression_strength < 50,
            5000.0 / compression_strength,
            200.0 - 2.0 * compression_strength,
        )
    )
    return scale


def _quantize(
    input: torch.Tensor,
    jpeg_quality: torch.Tensor,
    quantization_table: torch.Tensor,
) -> torch.Tensor:
    """Perform quantization.

    Args:
        input (torch.Tensor): Input torch.tensor of the shape :math:`(B, N, 8, 8)`.
        jpeg_quality (torch.Tensor): Compression strength to be applied, shape is :math:`(B)`.
        quantization_table (torch.Tensor): Quantization table of the shape :math:`(1, 8, 8)` or :math:`(B, 8, 8)`.

    Returns:
        output (torch.Tensor): Quantized output torch.tensor of the shape :math:`(B, N, 8, 8)`.

    """
    # Scale quantization table
    quantization_table_scaled: torch.Tensor = (
        quantization_table[:, None] * _jpeg_quality_to_scale(jpeg_quality)[:, None, None, None]
    )
    # Perform scaling
    quantization_table = differentiable_polynomial_floor(
        differentiable_clipping((quantization_table_scaled + 50.0) / 100.0, 1, 255)
    )
    output: torch.Tensor = input / quantization_table
    # Perform rounding
    output = differentiable_polynomial_rounding(output)
    return output


def _dequantize(
    input: torch.Tensor,
    jpeg_quality: torch.Tensor,
    quantization_table: torch.Tensor,
) -> torch.Tensor:
    """Perform dequantization.

    Args:
        input (torch.Tensor): Input torch.tensor of the shape :math:`(B, N, 8, 8)`.
        jpeg_quality (torch.Tensor): Compression strength to be applied, shape is :math:`(B)`.
        quantization_table (torch.Tensor): Quantization table of the shape :math:`(1, 8, 8)` or :math:`(B, 8, 8)`.

    Returns:
        output (torch.Tensor): Quantized output torch.tensor of the shape :math:`(B, N, 8, 8)`.

    """
    # Scale quantization table
    quantization_table_scaled: torch.Tensor = (
        quantization_table[:, None] * _jpeg_quality_to_scale(jpeg_quality)[:, None, None, None]
    )
    # Perform scaling
    output: torch.Tensor = input * differentiable_polynomial_floor(
        differentiable_clipping((quantization_table_scaled + 50.0) / 100.0, 1, 255)
    )
    return output


def _chroma_subsampling(input_ycbcr: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Implement chroma subsampling.

    Args:
        input_ycbcr (torch.Tensor): YCbCr input torch.tensor of the shape :math:`(B, 3, H, W)`.

    Returns:
        output_y (torch.Tensor): Y component (not-subsampled), shape is :math:`(B, H, W)`.
        output_cb (torch.Tensor): Cb component (subsampled), shape is :math:`(B, H // 2, W // 2)`.
        output_cr (torch.Tensor): Cr component (subsampled), shape is :math:`(B, H // 2, W // 2)`.

    """
    # Get components
    output_y: torch.Tensor = input_ycbcr[:, 0]
    output_cb: torch.Tensor = input_ycbcr[:, 1]
    output_cr: torch.Tensor = input_ycbcr[:, 2]
    # Perform average pooling of Cb and Cr channels
    output_cb = rescale(
        output_cb[:, None],
        factor=0.5,
        interpolation="bilinear",
        align_corners=False,
        antialias=True,
    )
    output_cr = rescale(
        output_cr[:, None],
        factor=0.5,
        interpolation="bilinear",
        align_corners=False,
        antialias=True,
    )
    return output_y, output_cb[:, 0], output_cr[:, 0]


def _chroma_upsampling(input_c: torch.Tensor) -> torch.Tensor:
    """Perform chroma upsampling.

    Args:
        input_c (torch.Tensor): Cb or Cr component to be upsampled of the shape :math:`(B, H, W)`.

    Returns:
        output_c (torch.Tensor): Upsampled C(b or r) component of the shape :math:`(B, H * 2, W * 2)`.

    """
    # Upsample component
    output_c: torch.Tensor = rescale(
        input_c[:, None],
        factor=2.0,
        interpolation="bilinear",
        align_corners=False,
        antialias=False,
    )
    return output_c[:, 0]


def _jpeg_encode(
    image_rgb: torch.Tensor,
    jpeg_quality: torch.Tensor,
    quantization_table_y: torch.Tensor,
    quantization_table_c: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Perform JPEG encoding.

    Args:
        image_rgb (torch.Tensor): RGB input images of the shape :math:`(B, 3, H, W)`.
        jpeg_quality (torch.Tensor): Compression strength of the shape :math:`(B)`.
        quantization_table_y (torch.Tensor): Quantization table for Y channel.
        quantization_table_c (torch.Tensor): Quantization table for C channels.

    Returns:
        y_encoded (torch.Tensor): Encoded Y component of the shape :math:`(B, N, 8, 8)`.
        cb_encoded (torch.Tensor): Encoded Cb component of the shape :math:`(B, N, 8, 8)`.
        cr_encoded (torch.Tensor): Encoded Cr component of the shape :math:`(B, N, 8, 8)`.

    """
    # Convert RGB image to YCbCr.
    image_ycbcr: torch.Tensor = rgb_to_ycbcr(image_rgb)
    # Scale pixel-range to [0, 255]
    image_ycbcr = 255.0 * image_ycbcr
    # Perform chroma subsampling
    input_y, input_cb, input_cr = _chroma_subsampling(image_ycbcr)
    # Patchify, DCT, and rounding
    input_y, input_cb, input_cr = (
        _patchify_8x8(input_y),
        _patchify_8x8(input_cb),
        _patchify_8x8(input_cr),
    )
    dct_y = _dct_8x8(input_y)
    dct_cb_cr = _dct_8x8(torch.cat((input_cb, input_cr), dim=1))
    y_encoded: torch.Tensor = _quantize(
        dct_y,
        jpeg_quality,
        quantization_table_y,
    )
    cb_encoded, cr_encoded = _quantize(
        dct_cb_cr,
        jpeg_quality,
        quantization_table_c,
    ).chunk(2, dim=1)
    return y_encoded, cb_encoded, cr_encoded


def _jpeg_decode(
    input_y: torch.Tensor,
    input_cb: torch.Tensor,
    input_cr: torch.Tensor,
    jpeg_quality: torch.Tensor,
    H: int,
    W: int,
    quantization_table_y: torch.Tensor,
    quantization_table_c: torch.Tensor,
) -> torch.Tensor:
    """Perform JPEG decoding.

    Args:
        input_y (torch.Tensor): Compressed Y component of the shape :math:`(B, N, 8, 8)`.
        input_cb (torch.Tensor): Compressed Cb component of the shape :math:`(B, N, 8, 8)`.
        input_cr (torch.Tensor): Compressed Cr component of the shape :math:`(B, N, 8, 8)`.
        jpeg_quality (torch.Tensor): Compression strength of the shape :math:`(B)`.
        H (int): Original image height.
        W (int): Original image width.
        quantization_table_y (torch.Tensor): Quantization table for Y channel.
        quantization_table_c (torch.Tensor): Quantization table for C channels.

    Returns:
        rgb_decoded (torch.Tensor): Decompressed RGB image of the shape :math:`(B, 3, H, W)`.

    """
    # Dequantize inputs
    input_y = _dequantize(
        input_y,
        jpeg_quality,
        quantization_table_y,
    )
    input_cb_cr = _dequantize(
        torch.cat((input_cb, input_cr), dim=1),
        jpeg_quality,
        quantization_table_c,
    )
    # Perform inverse DCT
    idct_y: torch.Tensor = _idct_8x8(input_y)
    idct_cb, idct_cr = _idct_8x8(input_cb_cr).chunk(2, dim=1)
    # Reverse patching
    image_y: torch.Tensor = _unpatchify_8x8(idct_y, H, W)
    image_cb: torch.Tensor = _unpatchify_8x8(idct_cb, H // 2, W // 2)
    image_cr: torch.Tensor = _unpatchify_8x8(idct_cr, H // 2, W // 2)
    # Perform chroma upsampling
    image_cb = _chroma_upsampling(image_cb)
    image_cr = _chroma_upsampling(image_cr)
    # Back to [0, 1] pixel-range
    image_ycbcr: torch.Tensor = torch.stack((image_y, image_cb, image_cr), dim=1) / 255.0
    # Convert back to RGB space.
    rgb_decoded: torch.Tensor = ycbcr_to_rgb(image_ycbcr)
    return rgb_decoded


def _perform_padding(image: torch.Tensor) -> tuple[torch.Tensor, int, int]:
    """Pad a given image to be dividable by 16.

    Args:
        image: Image of the shape :math:`(*, 3, H, W)`.

    Returns:
        image_padded: Padded image of the shape :math:`(*, 3, H_{new}, W_{new})`.
        h_pad: Padded pixels along the horizontal axis.
        w_pad: Padded pixels along the vertical axis.

    """
    # Get spatial dimensions of the image
    H, W = image.shape[-2:]
    # Compute horizontal and vertical padding
    h_pad: int = math.ceil(H / 16) * 16 - H
    w_pad: int = math.ceil(W / 16) * 16 - W
    # Perform padding (we follow JPEG and F.pad only the bottom and right side of the image)
    image_padded: torch.Tensor = F.pad(image, (0, w_pad, 0, h_pad), "replicate")
    return image_padded, h_pad, w_pad


@perform_keep_shape_image
def jpeg_codec_differentiable(
    image_rgb: torch.Tensor,
    jpeg_quality: torch.Tensor,
    quantization_table_y: torch.Tensor | None = None,
    quantization_table_c: torch.Tensor | None = None,
) -> torch.Tensor:
    r"""Differentiable JPEG encoding-decoding module.

    Based on :cite:`reich2024` :cite:`shin2017`, we perform differentiable JPEG encoding-decoding as follows:

    .. image:: _static/img/jpeg_codec_differentiable.png

    .. math::

        \text{JPEG}_{\text{diff}}(I, q, QT_{y}, QT_{c}) = \hat{I}

    Where:
       - :math:`I` is the original image to be coded.
       - :math:`q` is the JPEG quality controlling the compression strength.
       - :math:`QT_{y}` is the luma quantization table.
       - :math:`QT_{c}` is the chroma quantization table.
       - :math:`\hat{I}` is the resulting JPEG encoded-decoded image.

    .. note:::
        The input (and output) pixel range is :math:`[0, 1]`. In case you want to handle normalized images you are
        required to first perform denormalization followed by normalizing the output images again.

        Note, that this implementation models the encoding-decoding mapping of JPEG in a differentiable setting,
        however, does not allow the excess of the JPEG-coded byte file itself.
        For more details please refer to :cite:`reich2024`.

        This implementation is not meant for data loading. For loading JPEG images please refer to `kornia.io`.
        There we provide an optimized Rust implementation for fast JPEG loading.

    Args:
        image_rgb: the RGB image to be coded.
        jpeg_quality: JPEG quality in the range :math:`[0, 100]` controlling the compression strength.
        quantization_table_y: quantization table for Y channel. Default: `None`, which will load the standard
          quantization table.
        quantization_table_c: quantization table for C channels. Default: `None`, which will load the standard
          quantization table.

    Shape:
        - image_rgb: :math:`(*, 3, H, W)`.
        - jpeg_quality: :math:`(1)` or :math:`(B)` (if used batch dim. needs to match w/ image_rgb).
        - quantization_table_y: :math:`(8, 8)` or :math:`(B, 8, 8)` (if used batch dim. needs to match w/ image_rgb).
        - quantization_table_c: :math:`(8, 8)` or :math:`(B, 8, 8)` (if used batch dim. needs to match w/ image_rgb).

    Return:
        JPEG coded image of the shape :math:`(B, 3, H, W)`

    Example:
        To perform JPEG coding with the standard quantization tables just provide a JPEG quality

        >>> img = torch.rand(3, 3, 64, 64, requires_grad=True, dtype=torch.float)
        >>> jpeg_quality = torch.tensor((99.0, 25.0, 1.0), requires_grad=True)
        >>> img_jpeg = jpeg_codec_differentiable(img, jpeg_quality)
        >>> img_jpeg.sum().backward()

        You also have the option to provide custom quantization tables

        >>> img = torch.rand(3, 3, 64, 64, requires_grad=True, dtype=torch.float)
        >>> jpeg_quality = torch.tensor((99.0, 25.0, 1.0), requires_grad=True)
        >>> quantization_table_y = torch.randint(1, 256, size=(3, 8, 8), dtype=torch.float)
        >>> quantization_table_c = torch.randint(1, 256, size=(3, 8, 8), dtype=torch.float)
        >>> img_jpeg = jpeg_codec_differentiable(img, jpeg_quality, quantization_table_y, quantization_table_c)
        >>> img_jpeg.sum().backward()

        In case you want to control the quantization purly base on the quantization tables use a JPEG quality of 99.5.
        Setting the JPEG quality to 99.5 leads to a QT scaling of 1, see Eq. 2 of :cite:`reich2024` for details.

        >>> img = torch.rand(3, 3, 64, 64, requires_grad=True, dtype=torch.float)
        >>> jpeg_quality = torch.ones(3) * 99.5
        >>> quantization_table_y = torch.randint(1, 256, size=(3, 8, 8), dtype=torch.float)
        >>> quantization_table_c = torch.randint(1, 256, size=(3, 8, 8), dtype=torch.float)
        >>> img_jpeg = jpeg_codec_differentiable(img, jpeg_quality, quantization_table_y, quantization_table_c)
        >>> img_jpeg.sum().backward()

    """
    # Check that inputs are tensors
    KORNIA_CHECK_IS_TENSOR(image_rgb)
    KORNIA_CHECK_IS_TENSOR(jpeg_quality)
    # Get device and dtype
    dtype: Union[torch.dtype, None] = image_rgb.dtype
    device: Union[str, torch.device, None] = image_rgb.device
    # Use default QT if QT is not given
    quantization_table_y = _get_default_qt_y(device, dtype) if quantization_table_y is None else quantization_table_y
    quantization_table_c = _get_default_qt_c(device, dtype) if quantization_table_c is None else quantization_table_c
    KORNIA_CHECK_IS_TENSOR(quantization_table_y)
    KORNIA_CHECK_IS_TENSOR(quantization_table_c)
    # Check shape of inputs
    KORNIA_CHECK_SHAPE(image_rgb, ["*", "3", "H", "W"])
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
        (jpeg_quality.amin().item() >= 0.0) and (jpeg_quality.amax().item() <= 100.0),
        f"JPEG quality is out of range. Expected range is [0, 100], "
        f"got [{jpeg_quality.amin().item()}, {jpeg_quality.amax().item()}]. Consider clipping jpeg_quality.",
    )
    # Pad the image to a shape dividable by 16
    image_rgb, h_pad, w_pad = _perform_padding(image_rgb)
    # Get height and shape
    H, W = image_rgb.shape[-2:]
    # Check matching batch dimensions
    if quantization_table_y.shape[0] != 1:
        KORNIA_CHECK(
            quantization_table_y.shape[0] == image_rgb.shape[0],
            f"Batch dimensions do not match. "
            f"Got {image_rgb.shape[0]} images and {quantization_table_y.shape[0]} quantization tables (Y).",
        )
    if quantization_table_c.shape[0] != 1:
        KORNIA_CHECK(
            quantization_table_c.shape[0] == image_rgb.shape[0],
            f"Batch dimensions do not match. "
            f"Got {image_rgb.shape[0]} images and {quantization_table_c.shape[0]} quantization tables (C).",
        )
    if jpeg_quality.shape[0] != 1:
        KORNIA_CHECK(
            jpeg_quality.shape[0] == image_rgb.shape[0],
            f"Batch dimensions do not match. "
            f"Got {image_rgb.shape[0]} images and {jpeg_quality.shape[0]} JPEG qualities.",
        )
    # keep jpeg_quality same device as input torch.tensor
    jpeg_quality = jpeg_quality.to(device, dtype)
    # Quantization tables to same device and dtype as input image
    quantization_table_y = quantization_table_y.to(device, dtype)
    quantization_table_c = quantization_table_c.to(device, dtype)
    # Perform encoding
    y_encoded, cb_encoded, cr_encoded = _jpeg_encode(
        image_rgb=image_rgb,
        jpeg_quality=jpeg_quality,
        quantization_table_c=quantization_table_c,
        quantization_table_y=quantization_table_y,
    )
    image_rgb_jpeg: torch.Tensor = _jpeg_decode(
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
    image_rgb_jpeg = differentiable_clipping(input=image_rgb_jpeg, min_val=0.0, max_val=255.0)
    # Crop the image again to the original shape
    image_rgb_jpeg = image_rgb_jpeg[..., : H - h_pad, : W - w_pad]
    return image_rgb_jpeg


def _get_dct8_basis_scale(
    dtype: Union[torch.dtype, None], device: Union[str, torch.device, None]
) -> tuple[torch.Tensor, torch.Tensor]:
    key = (dtype, device)
    if key not in _DCT8_CACHE:
        i = torch.arange(8, dtype=dtype, device=device)
        freq = (2.0 * i + 1.0)[:, None] * i[None, :] * (pi / 16.0)
        basis_1d = torch.cos(freq)
        dct_tensor = basis_1d[:, None, :, None] * basis_1d[None, :, None, :]
        alpha = torch.ones(8, dtype=dtype, device=device)
        alpha[0] = 1.0 / (2**0.5)
        dct_scale = torch.outer(alpha, alpha) * 0.25
        _DCT8_CACHE[key] = (dct_tensor, dct_scale)
    return _DCT8_CACHE[key]


class JPEGCodecDifferentiable(nn.Module):
    r"""Differentiable JPEG encoding-decoding module.

    Based on :cite:`reich2024` :cite:`shin2017`, we perform differentiable JPEG encoding-decoding as follows:

    .. math::

        \text{JPEG}_{\text{diff}}(I, q, QT_{y}, QT_{c}) = \hat{I}

    Where:
       - :math:`I` is the original image to be coded.
       - :math:`q` is the JPEG quality controlling the compression strength.
       - :math:`QT_{y}` is the luma quantization table.
       - :math:`QT_{c}` is the chroma quantization table.
       - :math:`\hat{I}` is the resulting JPEG encoded-decoded image.

    .. image:: _static/img/jpeg_codec_differentiable.png

    .. note::
        The input (and output) pixel range is :math:`[0, 1]`. In case you want to handle normalized images you are
        required to first perform denormalization followed by normalizing the output images again.

        Note, that this implementation models the encoding-decoding mapping of JPEG in a differentiable setting,
        however, does not allow the excess of the JPEG-coded byte file itself.
        For more details please refer to :cite:`reich2024`.

        This implementation is not meant for data loading. For loading JPEG images please refer to `kornia.io`.
        There we provide an optimized Rust implementation for fast JPEG loading.

    Args:
        quantization_table_y: quantization table for Y channel. Default: `None`, which will load the standard
          quantization table.
        quantization_table_c: quantization table for C channels. Default: `None`, which will load the standard
          quantization table.

    Shape:
        - quantization_table_y: :math:`(8, 8)` or :math:`(B, 8, 8)` (if used batch dim. needs to match w/ image_rgb).
        - quantization_table_c: :math:`(8, 8)` or :math:`(B, 8, 8)` (if used batch dim. needs to match w/ image_rgb).
        - image_rgb: :math:`(*, 3, H, W)`.
        - jpeg_quality: :math:`(1)` or :math:`(B)` (if used batch dim. needs to match w/ image_rgb).

    Example:
        You can use the differentiable JPEG module with standard quantization tables by

        >>> diff_jpeg_module = JPEGCodecDifferentiable()
        >>> img = torch.rand(2, 3, 32, 32, requires_grad=True, dtype=torch.float)
        >>> jpeg_quality = torch.tensor((99.0, 1.0), requires_grad=True)
        >>> img_jpeg = diff_jpeg_module(img, jpeg_quality)
        >>> img_jpeg.sum().backward()

        You can also specify custom quantization tables to be used by

        >>> quantization_table_y = torch.randint(1, 256, size=(2, 8, 8), dtype=torch.float)
        >>> quantization_table_c = torch.randint(1, 256, size=(2, 8, 8), dtype=torch.float)
        >>> diff_jpeg_module = JPEGCodecDifferentiable(quantization_table_y, quantization_table_c)
        >>> img = torch.rand(2, 3, 32, 32, requires_grad=True, dtype=torch.float)
        >>> jpeg_quality = torch.tensor((99.0, 1.0), requires_grad=True)
        >>> img_jpeg = diff_jpeg_module(img, jpeg_quality)
        >>> img_jpeg.sum().backward()

        In case you want to learn the quantization tables just pass parameters `nn.Parameter`

        >>> quantization_table_y = torch.nn.Parameter(torch.randint(1, 256, size=(2, 8, 8), dtype=torch.float))
        >>> quantization_table_c = torch.nn.Parameter(torch.randint(1, 256, size=(2, 8, 8), dtype=torch.float))
        >>> diff_jpeg_module = JPEGCodecDifferentiable(quantization_table_y, quantization_table_c)
        >>> img = torch.rand(2, 3, 32, 32, requires_grad=True, dtype=torch.float)
        >>> jpeg_quality = torch.tensor((99.0, 1.0), requires_grad=True)
        >>> img_jpeg = diff_jpeg_module(img, jpeg_quality)
        >>> img_jpeg.sum().backward()

    """

    def __init__(
        self,
        quantization_table_y: torch.Tensor | nn.Parameter | None = None,
        quantization_table_c: torch.Tensor | nn.Parameter | None = None,
    ) -> None:
        super().__init__()
        # Get default quantization tables if needed
        quantization_table_y = _get_default_qt_y(None, None) if quantization_table_y is None else quantization_table_y
        quantization_table_c = _get_default_qt_c(None, None) if quantization_table_c is None else quantization_table_c
        if isinstance(quantization_table_y, nn.Parameter):
            self.register_parameter("quantization_table_y", quantization_table_y)
        else:
            self.register_buffer("quantization_table_y", quantization_table_y)
        if isinstance(quantization_table_c, nn.Parameter):
            self.register_parameter("quantization_table_c", quantization_table_c)
        else:
            self.register_buffer("quantization_table_c", quantization_table_c)

    def forward(
        self,
        image_rgb: torch.Tensor,
        jpeg_quality: torch.Tensor,
    ) -> torch.Tensor:
        device = image_rgb.device
        dtype = image_rgb.dtype
        # Move quantization tables to the same device and dtype as input
        # and store it in the local variables created in init
        quantization_table_y = self.quantization_table_y.to(device, dtype)
        quantization_table_c = self.quantization_table_c.to(device, dtype)
        # Perform encoding-decoding
        image_rgb_jpeg: torch.Tensor = jpeg_codec_differentiable(
            image_rgb,
            jpeg_quality=jpeg_quality,
            quantization_table_c=quantization_table_c,
            quantization_table_y=quantization_table_y,
        )
        return image_rgb_jpeg
