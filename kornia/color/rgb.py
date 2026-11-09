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

from typing import ClassVar, Optional, Union, cast

import torch
import torch.nn.functional as F
from torch import nn

from kornia.core.check import KORNIA_CHECK_IS_COLOR


def rgb_to_bgr(image: torch.Tensor) -> torch.Tensor:
    r"""Convert an RGB image to BGR.

    ``bgr = rgb_to_bgr(image)`` reorders the channels of a ``(*, 3, H, W)`` RGB image to BGR, the order OpenCV uses.

    .. image:: _static/img/rgb_to_bgr.png

    Convention:
        Reverses the three channels at axis -3; it is its own inverse.

    Args:
        image: RGB image with shape ``(*, 3, H, W)``.

    Returns:
        BGR image with shape ``(*, 3, H, W)``.

    See Also:
        - :func:`bgr_to_rgb`: the same operation under the other name.
        - :class:`RgbToBgr`: the module form.

    Example:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> output = rgb_to_bgr(input) # 2x3x4x5

    """
    if not isinstance(image, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(image)}")

    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError(f"Input size must have a shape of (*, 3, H, W).Got {image.shape}")

    return bgr_to_rgb(image)


def bgr_to_rgb(image: torch.Tensor) -> torch.Tensor:
    r"""Convert a BGR image to RGB.

    ``rgb = bgr_to_rgb(image)`` reorders the channels of a ``(*, 3, H, W)`` BGR image, as OpenCV loads it, to RGB.

    Convention:
        Reverses the three channels at axis -3; it is its own inverse.

    Args:
        image: BGR image with shape ``(*, 3, H, W)``.

    Returns:
        RGB image with shape ``(*, 3, H, W)``.

    See Also:
        - :func:`rgb_to_bgr`: the same operation under the other name.
        - :class:`BgrToRgb`: the module form.

    Example:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> output = bgr_to_rgb(input) # 2x3x4x5

    """
    if not isinstance(image, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(image)}")

    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError(f"Input size must have a shape of (*, 3, H, W).Got {image.shape}")

    # flip image channels
    out: torch.Tensor = image.flip(-3)
    return out


def rgb_to_rgba(image: torch.Tensor, alpha_val: Union[float, torch.Tensor]) -> torch.Tensor:
    r"""Convert an RGB image to RGBA by appending an alpha channel.

    ``rgba = rgb_to_rgba(image, alpha_val)`` returns a ``(*, 4, H, W)`` image whose first three channels are the RGB
    image and whose fourth is the alpha value.

    Convention:
        - A float ``alpha_val`` fills the alpha channel; an ``int`` raises ``TypeError``.
        - A tensor ``alpha_val`` must have shape ``(*, 1, H, W)`` equal to the image's leading and spatial axes; it is
          not broadcast.

    Args:
        image: RGB image with shape ``(*, 3, H, W)``.
        alpha_val: alpha, one float or a tensor.

    Returns:
        RGBA image with shape ``(*, 4, H, W)``.

    .. note:: The current functionality is NOT supported by Torchscript.

    See Also:
        - :func:`rgba_to_rgb`: the other direction, which composites the alpha over a background.
        - :func:`bgr_to_rgba`: takes a BGR image; the output is RGBA either way.
        - :class:`RgbToRgba`: the module form, with the alpha stored.

    Example:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> output = rgb_to_rgba(input, 1.) # 2x4x4x5

    """
    if not isinstance(image, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(image)}")

    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError(f"Input size must have a shape of (*, 3, H, W).Got {image.shape}")

    if not isinstance(alpha_val, (float, torch.Tensor)):
        raise TypeError(f"alpha_val type is not a float or torch.Tensor. Got {type(alpha_val)}")

    # add one channel
    r, g, b = torch.chunk(image, image.shape[-3], dim=-3)

    a: torch.Tensor = cast(torch.Tensor, alpha_val)

    if isinstance(alpha_val, float):
        a = torch.full_like(r, fill_value=float(alpha_val))
    elif alpha_val.shape != r.shape:
        raise ValueError(f"alpha_val must have shape {r.shape}. Got {alpha_val.shape}")

    return torch.cat([r, g, b, a], dim=-3)


def bgr_to_rgba(image: torch.Tensor, alpha_val: Union[float, torch.Tensor]) -> torch.Tensor:
    r"""Convert a BGR image to RGBA, with the color channels in RGB order.

    ``rgba = bgr_to_rgba(image, alpha_val)`` reverses the channels of a ``(*, 3, H, W)`` BGR image and appends alpha.

    Convention:
        - ``alpha_val`` is as in :func:`rgb_to_rgba`.

    Args:
        image: BGR image with shape ``(*, 3, H, W)``.
        alpha_val: alpha, one float or a tensor.

    Returns:
        RGBA image with shape ``(*, 4, H, W)``.

    .. note:: The current functionality is NOT supported by Torchscript.

    See Also:
        - :func:`rgb_to_rgba`: takes an RGB image.
        - :func:`rgba_to_bgr`: composites the alpha away and returns BGR.
        - :class:`BgrToRgba`: the module form, with the alpha stored.

    Example:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> output = bgr_to_rgba(input, 1.) # 2x4x4x5

    """
    if not isinstance(image, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(image)}")

    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError(f"Input size must have a shape of (*, 3, H, W).Got {image.shape}")

    if not isinstance(alpha_val, (float, torch.Tensor)):
        raise TypeError(f"alpha_val type is not a float or torch.Tensor. Got {type(alpha_val)}")

    # convert first to RGB, then add alpha channel
    x_rgb: torch.Tensor = bgr_to_rgb(image)
    return rgb_to_rgba(x_rgb, alpha_val)


def rgba_to_rgb(image: torch.Tensor, background_color: Optional[torch.Tensor] = None) -> torch.Tensor:
    r"""Convert an RGBA image to RGB by compositing it over a background.

    ``rgb = rgba_to_rgb(image, background_color)`` returns ``image_rgb * alpha + background * (1 - alpha)`` for a
    ``(*, 4, H, W)`` image, over white by default.

    Convention:
        - Alpha is straight, not premultiplied, and in :math:`[0, 1]`; color channels and ``background_color`` are
          unit range.
        - ``background_color`` is a tuple or list of three floats, or a tensor with three channels at axis -3 that
          broadcasts against the image, such as ``(3, 1, 1)``, ``(*, 3, 1, 1)`` or ``(*, 3, H, W)``. A 1D or 2D tensor
          raises (`#6621 <https://github.com/kornia/kornia/issues/6621>`_); reshape it to ``(3, 1, 1)``.

    Args:
        image: RGBA image with shape ``(*, 4, H, W)``.
        background_color: background color; ``None`` is white.

    Returns:
        RGB image with shape ``(*, 3, H, W)``.

    See Also:
        - :func:`rgb_to_rgba`: the other direction, which appends an alpha.
        - :func:`rgba_to_bgr`: composites over white only and returns BGR.
        - :class:`RgbaToRgb`: the module form, which composites over white.

    Example:
        >>> rgba_image = torch.rand(2, 4, 32, 32)
        >>> # Test with default white background
        >>> rgb_image_default = rgba_to_rgb(rgba_image) # 2x3x32x32
        >>> # Test with a custom background color
        >>> rgb_image_custom = rgba_to_rgb(rgba_image, background_color=(0.0, 0.0, 1.0)) # 2x3x32x32
    """
    if not isinstance(image, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(image)}")

    if len(image.shape) < 3 or image.shape[-3] != 4:
        raise ValueError(f"Input size must have a shape of (*, 4, H, W).Got {image.shape}")

    # unpack channels
    image_rgb = image[..., :3, :, :]
    alpha = image[..., 3:4, :, :]

    if background_color is None:
        background_rgb = torch.ones_like(image_rgb)
    elif isinstance(background_color, (tuple, list)):
        if len(background_color) != 3:
            raise ValueError("background_color as a list/tuple must have 3 elements (R, G, B).")
        # A channel-first color broadcasts over any leading dimensions without adding a batch axis.
        background_rgb = torch.as_tensor(background_color, device=image.device, dtype=image.dtype).view(3, 1, 1)

    elif isinstance(background_color, torch.Tensor):
        if background_color.shape[-3] != 3:
            raise ValueError(
                f"background_color torch.Tensor must have 3 channels in dimension -3. "
                f"Got shape {background_color.shape}"
            )
        background_rgb = background_color

    else:
        raise TypeError(f"Unsupported type for background_color: {type(background_color)}")

    return image_rgb * alpha + background_rgb * (1.0 - alpha)


def rgba_to_bgr(image: torch.Tensor) -> torch.Tensor:
    r"""Convert an RGBA image to BGR by compositing it over white.

    ``bgr = rgba_to_bgr(image)`` composites a ``(*, 4, H, W)`` image over white and returns it in BGR order. It takes no
    background: call :func:`rgba_to_rgb` with ``background_color`` and then :func:`rgb_to_bgr` for another one.

    Convention:
        - Composites as :func:`rgba_to_rgb` does, so color and alpha are unit range and alpha is straight.

    Args:
        image: RGBA image with shape ``(*, 4, H, W)``.

    Returns:
        BGR image with shape ``(*, 3, H, W)``.

    See Also:
        - :func:`bgr_to_rgba`: the other direction, which appends an alpha and returns RGBA.
        - :class:`RgbaToBgr`: the module form.

    Example:
        >>> input = torch.rand(2, 4, 4, 5)
        >>> output = rgba_to_bgr(input) # 2x3x4x5

    """
    if not isinstance(image, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(image)}")

    if len(image.shape) < 3 or image.shape[-3] != 4:
        raise ValueError(f"Input size must have a shape of (*, 4, H, W).Got {image.shape}")

    # convert to RGB first, then to BGR
    x_rgb: torch.Tensor = rgba_to_rgb(image)
    return rgb_to_bgr(x_rgb)


def rgb_to_linear_rgb(image: torch.Tensor) -> torch.Tensor:
    r"""Convert an sRGB image to linear RGB.

    ``lin = rgb_to_linear_rgb(image)`` removes the sRGB transfer curve from a unit-range ``(*, 3, H, W)`` image. Convert
    before :func:`rgb_to_xyz`, which expects linear RGB.

    .. image:: _static/img/rgb_to_linear_rgb.png

    Convention:
        Applies the curve channelwise: ``x / 12.92`` up to 0.04045 and ``((x + 0.055) / 1.055) ** 2.4`` above it.
        Values outside :math:`[0, 1]` follow the same formulas without clipping.

    Args:
        image: sRGB image with shape ``(*, 3, H, W)``.

    Returns:
        Linear RGB image with shape ``(*, 3, H, W)``.

    See Also:
        - :func:`linear_rgb_to_rgb`: the inverse.
        - :class:`RgbToLinearRgb`: the module form.

    Example:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> output = rgb_to_linear_rgb(input) # 2x3x4x5

    """
    if not isinstance(image, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(image)}")

    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError(f"Input size must have a shape of (*, 3, H, W).Got {image.shape}")

    threshold = 0.04045
    lin_rgb: torch.Tensor = torch.where(
        image > threshold,
        torch.pow(((image + 0.055).clamp(min=0.0) / 1.055), 2.4),
        image / 12.92,
    )

    return lin_rgb


def linear_rgb_to_rgb(image: torch.Tensor) -> torch.Tensor:
    r"""Convert a linear RGB image to sRGB.

    ``rgb = linear_rgb_to_rgb(image)`` applies the sRGB transfer curve to a unit-range ``(*, 3, H, W)`` linear-light
    image, for example the result of :func:`xyz_to_rgb`.

    Convention:
        Applies the curve channelwise: ``12.92 * x`` up to 0.0031308 and ``1.055 * x ** (1 / 2.4) - 0.055`` above it.
        Values outside :math:`[0, 1]` follow the same formulas without clipping.

    Args:
        image: linear RGB image with shape ``(*, 3, H, W)``.

    Returns:
        sRGB image with shape ``(*, 3, H, W)``.

    See Also:
        - :func:`rgb_to_linear_rgb`: removes the curve.
        - :class:`LinearRgbToRgb`: the module form.

    Example:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> output = linear_rgb_to_rgb(input) # 2x3x4x5

    """
    if not isinstance(image, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(image)}")

    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError(f"Input size must have a shape of (*, 3, H, W).Got {image.shape}")

    threshold = 0.0031308
    rgb: torch.Tensor = torch.where(
        image > threshold, 1.055 * torch.pow(image.clamp(min=threshold), 1 / 2.4) - 0.055, 12.92 * image
    )

    return rgb


def normals_to_rgb255(image: torch.Tensor) -> torch.Tensor:
    r"""Map surface normals in [-1, 1] to RGB in [0, 255].

    ``rgb = normals_to_rgb255(image)`` maps each component ``n`` of a ``(*, 3, H, W)`` normal map to
    ``255 * (n + 1) / 2``, so -1, 0 and 1 become 0, 127.5 and 255.

    Convention:
        Components outside :math:`[-1, 1]` are clamped; the vectors are not normalized.

    Args:
        image: surface normals with shape ``(*, 3, H, W)``.

    Returns:
        RGB image in :math:`[0, 255]` with shape ``(*, 3, H, W)``.

    See Also:
        - :func:`rgb255_to_normals`: the inverse for unit-length normals.
        - :class:`NormalsToRgb255`: the module form.

    Example:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> output = normals_to_rgb255(input) # 2x3x4x5

    """
    KORNIA_CHECK_IS_COLOR(image)

    return (0.5 * (image + 1.0)).clip(0.0, 1.0) * 255


def rgb_to_rgb255(image: torch.Tensor) -> torch.Tensor:
    r"""Scale an RGB image from [0, 1] to [0, 255].

    ``out = rgb_to_rgb255(image)`` multiplies a ``(*, 3, H, W)`` unit-range RGB image by 255. The result is a float
    image, not uint8.

    Convention:
        Values are clamped to :math:`[0, 255]` and not rounded.

    Args:
        image: RGB image with shape ``(*, 3, H, W)``.

    Returns:
        RGB image in :math:`[0, 255]` with shape ``(*, 3, H, W)``.

    See Also:
        - :func:`rgb255_to_rgb`: divides by 255 and does not clip.
        - :class:`RgbToRgb255`: the module form.

    Example:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> output = rgb_to_rgb255(input) # 2x3x4x5

    """
    KORNIA_CHECK_IS_COLOR(image)
    return (image * 255).clip(0.0, 255.0)


def rgb255_to_rgb(image: torch.Tensor) -> torch.Tensor:
    r"""Scale an RGB image from [0, 255] to [0, 1].

    ``out = rgb255_to_rgb(image)`` divides a ``(*, 3, H, W)`` image in :math:`[0, 255]` by 255.

    Convention:
        Values outside :math:`[0, 255]` are not clipped.

    Args:
        image: RGB image in :math:`[0, 255]` with shape ``(*, 3, H, W)``.

    Returns:
        RGB image in :math:`[0, 1]` with shape ``(*, 3, H, W)``.

    See Also:
        - :func:`rgb_to_rgb255`: multiplies by 255 and clamps.
        - :class:`Rgb255ToRgb`: the module form.

    Example:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> output = rgb255_to_rgb(input) # 2x3x4x5

    """
    KORNIA_CHECK_IS_COLOR(image)

    return image / 255.0


def rgb255_to_normals(image: torch.Tensor) -> torch.Tensor:
    r"""Map an RGB image in [0, 255] to unit surface normals.

    ``normals = rgb255_to_normals(image)`` maps each channel ``c`` of a ``(*, 3, H, W)`` image to ``2 * c / 255 - 1``
    and scales every pixel's three components to unit length.

    Convention:
        - A pixel that maps to the zero vector, such as mid grey (127.5, 127.5, 127.5), stays zero.
        - Half inputs are computed in float32 and rounded to the input dtype once at return.

    Args:
        image: RGB image in :math:`[0, 255]` with shape ``(*, 3, H, W)``.

    Returns:
        Surface normals with shape ``(*, 3, H, W)``.

    See Also:
        - :func:`normals_to_rgb255`: the inverse for unit-length normals; it does not normalize.
        - :class:`Rgb255ToNormals`: the module form.

    Example:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> output = rgb255_to_normals(input) # 2x3x4x5

    """
    KORNIA_CHECK_IS_COLOR(image)
    dtype = image.dtype
    if dtype in (torch.float16, torch.bfloat16):
        image = image.float()
    output = F.normalize((image / 255.0) * 2.0 - 1.0, dim=-3, p=2.0)
    return output.to(dtype) if dtype in (torch.float16, torch.bfloat16) else output


class BgrToRgb(nn.Module):
    r"""Convert an image from BGR to RGB (module form).

    ``out = BgrToRgb()(image)`` is :func:`bgr_to_rgb`.

    Shape:
        - image: ``(*, 3, H, W)``
        - output: ``(*, 3, H, W)``

    See Also:
        - :func:`bgr_to_rgb`: the one-call form.
        - :class:`RgbToBgr`: the same channel reversal.

    Example:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> rgb = BgrToRgb()
        >>> output = rgb(input)  # 2x3x4x5

    """

    ONNX_DEFAULT_INPUTSHAPE: ClassVar[list[int]] = [-1, 3, -1, -1]
    ONNX_DEFAULT_OUTPUTSHAPE: ClassVar[list[int]] = [-1, 3, -1, -1]

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Convert a BGR tensor to RGB.

        Args:
            image: Input tensor with shape ``(*, 3, H, W)``.
                Here, ``*`` means any number of leading dimensions (for example, batch size),
                ``3`` corresponds to BGR channels, and ``H``/``W`` are height and width.

        Returns:
            RGB tensor with shape ``(*, 3, H, W)``.
        """
        return bgr_to_rgb(image)


class RgbToBgr(nn.Module):
    r"""Convert an image from RGB to BGR (module form).

    ``out = RgbToBgr()(image)`` is :func:`rgb_to_bgr`.

    Shape:
        - image: ``(*, 3, H, W)``
        - output: ``(*, 3, H, W)``

    See Also:
        - :func:`rgb_to_bgr`: the one-call form.
        - :class:`BgrToRgb`: the same channel reversal.

    Example:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> bgr = RgbToBgr()
        >>> output = bgr(input)  # 2x3x4x5

    """

    ONNX_DEFAULT_INPUTSHAPE: ClassVar[list[int]] = [-1, 3, -1, -1]
    ONNX_DEFAULT_OUTPUTSHAPE: ClassVar[list[int]] = [-1, 3, -1, -1]

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Convert an RGB tensor to BGR.

        Args:
            image: Input tensor with shape ``(*, 3, H, W)``.
                Here, ``*`` means any number of leading dimensions (for example, batch size),
                ``3`` corresponds to RGB channels, and ``H``/``W`` are height and width.

        Returns:
            BGR tensor with shape ``(*, 3, H, W)``.
        """
        return rgb_to_bgr(image)


class RgbToRgba(nn.Module):
    r"""Convert an image from RGB to RGBA (module form).

    ``conv = RgbToRgba(alpha_val)`` stores the alpha, and ``out = conv(image)`` is :func:`rgb_to_rgba` with it.

    Convention:
        - See the convention block of :func:`rgb_to_rgba`.

    Args:
        alpha_val: alpha, as in :func:`rgb_to_rgba`. A plain tensor is kept as a non-persistent buffer, so
          ``.to(device)`` moves it and ``state_dict`` omits it.

    Shape:
        - image: ``(*, 3, H, W)``
        - output: ``(*, 4, H, W)``

    .. note:: The current functionality is NOT supported by Torchscript.

    See Also:
        - :func:`rgb_to_rgba`: the one-call form.
        - :class:`BgrToRgba`: takes a BGR image.

    Example:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> rgba = RgbToRgba(1.)
        >>> output = rgba(input)  # 2x4x4x5

    """

    ONNX_DEFAULT_INPUTSHAPE: ClassVar[list[int]] = [-1, 3, -1, -1]
    ONNX_DEFAULT_OUTPUTSHAPE: ClassVar[list[int]] = [-1, 4, -1, -1]

    def __init__(self, alpha_val: Union[float, torch.Tensor]) -> None:
        super().__init__()
        if isinstance(alpha_val, torch.Tensor) and not isinstance(alpha_val, nn.Parameter):
            self.register_buffer("alpha_val", alpha_val, persistent=False)
        else:
            self.alpha_val = alpha_val

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Convert an RGB tensor to RGBA.

        Args:
            image: Input tensor with shape ``(*, 3, H, W)``.
                Here, ``*`` means any number of leading dimensions (for example, batch size),
                ``3`` corresponds to RGB channels, and ``H``/``W`` are height and width.

        Returns:
            RGBA tensor with shape ``(*, 4, H, W)`` using this module's alpha value.
        """
        return rgb_to_rgba(image, self.alpha_val)


class BgrToRgba(nn.Module):
    r"""Convert an image from BGR to RGBA (module form).

    ``conv = BgrToRgba(alpha_val)`` stores the alpha, and ``out = conv(image)`` is :func:`bgr_to_rgba` with it.

    Convention:
        - See the convention block of :func:`bgr_to_rgba`.

    Args:
        alpha_val: alpha, as in :func:`bgr_to_rgba`. A plain tensor is kept as a non-persistent buffer, so
          ``.to(device)`` moves it and ``state_dict`` omits it.

    Shape:
        - image: ``(*, 3, H, W)``
        - output: ``(*, 4, H, W)``

    .. note:: The current functionality is NOT supported by Torchscript.

    See Also:
        - :func:`bgr_to_rgba`: the one-call form.
        - :class:`RgbToRgba`: takes an RGB image.

    Example:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> rgba = BgrToRgba(1.)
        >>> output = rgba(input)  # 2x4x4x5

    """

    ONNX_DEFAULT_INPUTSHAPE: ClassVar[list[int]] = [-1, 3, -1, -1]
    ONNX_DEFAULT_OUTPUTSHAPE: ClassVar[list[int]] = [-1, 4, -1, -1]

    def __init__(self, alpha_val: Union[float, torch.Tensor]) -> None:
        super().__init__()
        if isinstance(alpha_val, torch.Tensor) and not isinstance(alpha_val, nn.Parameter):
            self.register_buffer("alpha_val", alpha_val, persistent=False)
        else:
            self.alpha_val = alpha_val

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Convert a BGR tensor to RGBA with this module's alpha value.

        Args:
            image: BGR image with shape ``(*, 3, H, W)``.

        Returns:
            RGBA image with shape ``(*, 4, H, W)``.
        """
        return bgr_to_rgba(image, self.alpha_val)


class RgbaToRgb(nn.Module):
    r"""Convert an image from RGBA to RGB (module form).

    ``out = RgbaToRgb()(image)`` is :func:`rgba_to_rgb` over its default white background.

    Convention:
        - See the convention block of :func:`rgba_to_rgb`.

    Shape:
        - image: ``(*, 4, H, W)``
        - output: ``(*, 3, H, W)``

    See Also:
        - :func:`rgba_to_rgb`: the one-call form, which also takes a ``background_color``.
        - :class:`RgbaToBgr`: returns BGR.

    Example:
        >>> input = torch.rand(2, 4, 4, 5)
        >>> rgba = RgbaToRgb()
        >>> output = rgba(input)  # 2x3x4x5

    """

    ONNX_DEFAULT_INPUTSHAPE: ClassVar[list[int]] = [-1, 4, -1, -1]
    ONNX_DEFAULT_OUTPUTSHAPE: ClassVar[list[int]] = [-1, 3, -1, -1]

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Convert an RGBA tensor to RGB.

        Args:
            image: Input tensor with shape ``(*, 4, H, W)``.
                Here, ``*`` means any number of leading dimensions (for example, batch size),
                ``4`` corresponds to RGBA channels, and ``H``/``W`` are height and width.

        Returns:
            RGB tensor with shape ``(*, 3, H, W)``.
        """
        return rgba_to_rgb(image)


class RgbaToBgr(nn.Module):
    r"""Convert an image from RGBA to BGR (module form).

    ``out = RgbaToBgr()(image)`` is :func:`rgba_to_bgr`.

    Convention:
        - See the convention block of :func:`rgba_to_bgr`.

    Shape:
        - image: ``(*, 4, H, W)``
        - output: ``(*, 3, H, W)``

    See Also:
        - :func:`rgba_to_bgr`: the one-call form.
        - :class:`RgbaToRgb`: returns RGB.

    Example:
        >>> input = torch.rand(2, 4, 4, 5)
        >>> rgba = RgbaToBgr()
        >>> output = rgba(input)  # 2x3x4x5

    """

    ONNX_DEFAULT_INPUTSHAPE: ClassVar[list[int]] = [-1, 4, -1, -1]
    ONNX_DEFAULT_OUTPUTSHAPE: ClassVar[list[int]] = [-1, 3, -1, -1]

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Convert an RGBA tensor to BGR.

        Args:
            image: Input tensor with shape ``(*, 4, H, W)``.
                Here, ``*`` means any number of leading dimensions (for example, batch size),
                ``4`` corresponds to RGBA channels, and ``H``/``W`` are height and width.

        Returns:
            BGR tensor with shape ``(*, 3, H, W)``.
        """
        return rgba_to_bgr(image)


class RgbToLinearRgb(nn.Module):
    r"""Convert an image from sRGB to linear RGB (module form).

    ``out = RgbToLinearRgb()(image)`` is :func:`rgb_to_linear_rgb`.

    Convention:
        - See the convention block of :func:`rgb_to_linear_rgb`.

    Shape:
        - image: ``(*, 3, H, W)``
        - output: ``(*, 3, H, W)``

    See Also:
        - :func:`rgb_to_linear_rgb`: the one-call form.
        - :class:`LinearRgbToRgb`: the inverse.

    Example:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> rgb_lin = RgbToLinearRgb()
        >>> output = rgb_lin(input)  # 2x3x4x5

    References:
        [1] https://stackoverflow.com/questions/35952564/convert-rgb-to-srgb

        [2] https://www.cambridgeincolour.com/tutorials/gamma-correction.htm

        [3] https://en.wikipedia.org/wiki/SRGB

    """

    ONNX_DEFAULT_INPUTSHAPE: ClassVar[list[int]] = [-1, 3, -1, -1]
    ONNX_DEFAULT_OUTPUTSHAPE: ClassVar[list[int]] = [-1, 3, -1, -1]

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Convert an sRGB tensor to linear RGB.

        Args:
            image: Input tensor with shape ``(*, 3, H, W)``.
                Here, ``*`` means any number of leading dimensions (for example, batch size),
                ``3`` corresponds to RGB channels, and ``H``/``W`` are height and width.

        Returns:
            Linear RGB tensor with shape ``(*, 3, H, W)``.
        """
        return rgb_to_linear_rgb(image)


class LinearRgbToRgb(nn.Module):
    r"""Convert an image from linear RGB to sRGB (module form).

    ``out = LinearRgbToRgb()(image)`` is :func:`linear_rgb_to_rgb`.

    Convention:
        - See the convention block of :func:`linear_rgb_to_rgb`.

    Shape:
        - image: ``(*, 3, H, W)``
        - output: ``(*, 3, H, W)``

    See Also:
        - :func:`linear_rgb_to_rgb`: the one-call form.
        - :class:`RgbToLinearRgb`: removes the curve.

    Example:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> srgb = LinearRgbToRgb()
        >>> output = srgb(input)  # 2x3x4x5

    References:
        [1] https://stackoverflow.com/questions/35952564/convert-rgb-to-srgb

        [2] https://www.cambridgeincolour.com/tutorials/gamma-correction.htm

        [3] https://en.wikipedia.org/wiki/SRGB

    """

    ONNX_DEFAULT_INPUTSHAPE: ClassVar[list[int]] = [-1, 3, -1, -1]
    ONNX_DEFAULT_OUTPUTSHAPE: ClassVar[list[int]] = [-1, 3, -1, -1]

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Convert a linear RGB tensor to sRGB.

        Args:
            image: Input tensor with shape ``(*, 3, H, W)``.
                Here, ``*`` means any number of leading dimensions (for example, batch size),
                ``3`` corresponds to RGB channels, and ``H``/``W`` are height and width.

        Returns:
            sRGB tensor with shape ``(*, 3, H, W)``.
        """
        return linear_rgb_to_rgb(image)


class NormalsToRgb255(nn.Module):
    r"""Map surface normals in [-1, 1] to RGB in [0, 255] (module form).

    ``out = NormalsToRgb255()(image)`` is :func:`normals_to_rgb255`.

    Convention:
        - See the convention block of :func:`normals_to_rgb255`.

    Shape:
        - image: ``(*, 3, H, W)``
        - output: ``(*, 3, H, W)``

    See Also:
        - :func:`normals_to_rgb255`: the one-call form.
        - :class:`Rgb255ToNormals`: the inverse for unit-length normals.

    Example:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> rgb = NormalsToRgb255()
        >>> output = rgb(input)  # 2x3x4x5

    """

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Convert surface normals to RGB values in :math:`[0, 255]`.

        Args:
            image: Input normal map tensor with shape ``(*, 3, H, W)``.
                Here, ``*`` means any number of leading dimensions (for example, batch size),
                ``3`` corresponds to XYZ normal components, and ``H``/``W`` are height and width.

        Returns:
            RGB tensor in :math:`[0, 255]` with shape ``(*, 3, H, W)``.
        """
        return normals_to_rgb255(image)


class RgbToRgb255(nn.Module):
    r"""Scale an RGB image from [0, 1] to [0, 255] (module form).

    ``out = RgbToRgb255()(image)`` is :func:`rgb_to_rgb255`.

    Convention:
        - See the convention block of :func:`rgb_to_rgb255`.

    Shape:
        - image: ``(*, 3, H, W)``
        - output: ``(*, 3, H, W)``

    See Also:
        - :func:`rgb_to_rgb255`: the one-call form.
        - :class:`Rgb255ToRgb`: divides by 255.

    Example:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> rgb = RgbToRgb255()
        >>> output = rgb(input)  # 2x3x4x5

    """

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Convert RGB values from :math:`[0, 1]` to :math:`[0, 255]`.

        Args:
            image: Input RGB tensor with shape ``(*, 3, H, W)``.
                Here, ``*`` means any number of leading dimensions (for example, batch size),
                ``3`` corresponds to RGB channels, and ``H``/``W`` are height and width.

        Returns:
            RGB tensor in :math:`[0, 255]` with shape ``(*, 3, H, W)``.
        """
        return rgb_to_rgb255(image)


class Rgb255ToRgb(nn.Module):
    r"""Scale an RGB image from [0, 255] to [0, 1] (module form).

    ``out = Rgb255ToRgb()(image)`` is :func:`rgb255_to_rgb`.

    Convention:
        - See the convention block of :func:`rgb255_to_rgb`.

    Shape:
        - image: ``(*, 3, H, W)``
        - output: ``(*, 3, H, W)``

    See Also:
        - :func:`rgb255_to_rgb`: the one-call form.
        - :class:`RgbToRgb255`: multiplies by 255.

    Example:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> rgb = Rgb255ToRgb()
        >>> output = rgb(input)  # 2x3x4x5

    """

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Convert RGB values from :math:`[0, 255]` to :math:`[0, 1]`.

        Args:
            image: Input RGB tensor in :math:`[0, 255]` with shape ``(*, 3, H, W)``.
                Here, ``*`` means any number of leading dimensions (for example, batch size),
                ``3`` corresponds to RGB channels, and ``H``/``W`` are height and width.

        Returns:
            RGB tensor in :math:`[0, 1]` with shape ``(*, 3, H, W)``.
        """
        return rgb255_to_rgb(image)


class Rgb255ToNormals(nn.Module):
    r"""Map an RGB image in [0, 255] to unit surface normals (module form).

    ``out = Rgb255ToNormals()(image)`` is :func:`rgb255_to_normals`.

    Convention:
        - See the convention block of :func:`rgb255_to_normals`.

    Shape:
        - image: ``(*, 3, H, W)``
        - output: ``(*, 3, H, W)``

    See Also:
        - :func:`rgb255_to_normals`: the one-call form.
        - :class:`NormalsToRgb255`: maps normals to colors.

    Example:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> normals = Rgb255ToNormals()
        >>> output = normals(input)  # 2x3x4x5

    """

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Convert RGB values in :math:`[0, 255]` to surface normals.

        Args:
            image: Input RGB tensor in :math:`[0, 255]` with shape ``(*, 3, H, W)``.
                Here, ``*`` means any number of leading dimensions (for example, batch size),
                ``3`` corresponds to RGB channels, and ``H``/``W`` are height and width.

        Returns:
            Normal map tensor with shape ``(*, 3, H, W)``.
        """
        return rgb255_to_normals(image)
