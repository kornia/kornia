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

import errno
import numbers
import os
import zlib
from enum import Enum
from pathlib import Path
from typing import Any, Union

import numpy as np
import torch
from kornia_rs import io as _rs_io

import kornia
from kornia.core.check import KORNIA_CHECK, KORNIA_CHECK_IS_TENSOR
from kornia.core.exceptions import ShapeError, TypeCheckError, ValueCheckError
from kornia.image.utils import image_to_tensor, tensor_to_image

_WRITE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".tif", ".tiff")


def _color_mode(img_np: Any) -> str:
    """Return the kornia_rs colour mode of an HxWxC array with 1, 3 or 4 channels."""
    return {1: "mono", 3: "rgb", 4: "rgba"}[img_np.shape[-1]]


class ImageLoadType(Enum):
    r"""Enum to specify the desired image type.

    Convention:
        See the Convention block of :func:`~kornia.io.load_image` for the channels, dtype and range of each member.
    """

    UNCHANGED = 0
    GRAY8 = 1
    RGB8 = 2
    RGBA8 = 3
    GRAY32 = 4
    RGB32 = 5


_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def _read_png_header(path_file: Path) -> tuple[int, int] | None:
    """Read the bit depth and color type bytes from a PNG file header.

    Returns ``(bit_depth, color_type)``, in that order, or None if the file is truncated or has an invalid PNG
    signature.
    PNG color types: 0=Grayscale, 2=RGB, 3=Indexed, 4=Grayscale+Alpha, 6=RGBA.
    """
    with open(path_file, "rb") as f:
        # Need 26 bytes: 8 (signature) + 8 (IHDR chunk header) + 8 (width/height) + 1 (bit depth) + 1 (color type)
        header = f.read(26)
        if len(header) < 26 or header[:8] != _PNG_SIGNATURE:
            return None
        return header[24], header[25]


# Map PNG color type byte to kornia_rs read mode.
# Types not listed here fall through to ``kornia_rs.io.read_image`` (which handles RGB).
# Note: color type 4 (Grayscale+Alpha) is intentionally omitted: no kornia_rs
# release decodes it, so ``read_image`` raises for such a file.
_PNG_COLOR_TYPE_TO_MODE: dict[int, str] = {
    0: "mono",  # Grayscale
    3: "mono",  # Indexed (palette): kornia_rs returns the indices; load_image expands them unless UNCHANGED
    6: "rgba",  # RGBA
}

# Start-of-frame markers (SOF0-SOF15 without DHT 0xC4, JPG 0xC8 and DAC 0xCC); byte 5 of their payload is the number
# of colour components.
_JPEG_SOF_MARKERS = frozenset({0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF})


def _read_jpeg_components(path_file: Path) -> int | None:
    """Return the number of colour components in a JPEG frame header, or None if the file has no readable one."""
    with open(path_file, "rb") as f:
        if f.read(2) != b"\xff\xd8":
            return None
        while True:
            if f.read(1) != b"\xff":
                return None
            code = f.read(1)
            while code == b"\xff":  # fill bytes may precede a marker
                code = f.read(1)
            if not code:
                return None
            marker = code[0]
            if marker == 0x01 or 0xD0 <= marker <= 0xD7:  # markers without a length field
                continue
            length = f.read(2)
            if len(length) < 2:
                return None
            if marker in _JPEG_SOF_MARKERS:
                frame = f.read(6)  # precision, height, width, number of components
                return frame[5] if len(frame) == 6 else None
            if marker == 0xDA:  # start of scan before any frame header
                return None
            f.seek(int.from_bytes(length, "big") - 2, os.SEEK_CUR)


def _read_png_chunks(path_file: Path) -> tuple[bytes | None, bytes | None]:
    """Return the raw bodies of a PNG's PLTE and tRNS chunks, None for a chunk the file does not have.

    Both chunks are checked as a PNG decoder checks them: a truncated body or a stored CRC that does not match the
    chunk raises ``ValueError`` naming the file. Every other chunk is skipped unread.
    """
    plte: bytes | None = None
    trns: bytes | None = None
    with open(path_file, "rb") as f:
        if f.read(8) != _PNG_SIGNATURE:
            return None, None
        while True:
            head = f.read(8)
            kind = head[4:8]
            if len(head) < 8 or kind in (b"IDAT", b"IEND"):  # PLTE and tRNS precede the image data
                break
            size = int.from_bytes(head[:4], "big")
            if kind not in (b"PLTE", b"tRNS"):
                f.seek(size + 4, os.SEEK_CUR)  # body and CRC
                continue
            body = f.read(size)
            crc = f.read(4)
            name = kind.decode("ascii")
            if len(body) < size or len(crc) < 4:
                raise ValueError(f"Failed to decode image '{path_file}': its {name} chunk is truncated.")
            if int.from_bytes(crc, "big") != zlib.crc32(kind + body):
                raise ValueError(f"Failed to decode image '{path_file}': its {name} chunk fails its CRC check.")
            if kind == b"PLTE":
                plte = body
            else:
                trns = body
    return plte, trns


def _apply_png_palette_and_transparency(path_file: Path, image: torch.Tensor) -> torch.Tensor:
    """Apply a PNG's palette and tRNS chunk to the decoded pixels; any other file is returned unchanged.

    An 8-bit palette PNG (colour type 3) becomes ``(3, H, W)`` colours, or ``(4, H, W)`` with a tRNS chunk, whose
    per-entry alphas leave unlisted entries opaque. An 8-bit gray or RGB PNG (colour types 0 and 2) with a tRNS colour
    key becomes ``(4, H, W)`` RGBA, with alpha 0 where every sample equals the key and 255 elsewhere.
    """
    if path_file.suffix.lower() != ".png":
        return image
    header = _read_png_header(path_file)
    if header is None or header[1] not in (0, 2, 3):
        return image
    bit_depth, color_type = header
    plte, trns = _read_png_chunks(path_file)
    if color_type == 3:
        if plte is None:
            return image
        if len(plte) % 3:
            raise ValueError(
                f"Failed to decode image '{path_file}': its PLTE chunk has {len(plte)} bytes, not a whole number of "
                "RGB entries."
            )
        colors = torch.frombuffer(bytearray(plte), dtype=torch.uint8).view(-1, 3)
        table = colors
        if trns is not None:
            alpha = torch.full((colors.shape[0],), 255, dtype=torch.uint8)
            listed = torch.frombuffer(bytearray(trns), dtype=torch.uint8)[: colors.shape[0]]
            alpha[: listed.numel()] = listed
            table = torch.cat([colors, alpha[:, None]], dim=1)
        indices = image[0].long()
        if indices.numel() and int(indices.max()) >= table.shape[0]:
            raise ValueError(
                f"Failed to decode image '{path_file}': a pixel uses palette entry {int(indices.max())}, "
                f"but the palette has {table.shape[0]} entries."
            )
        return table.to(image.device)[indices].permute(2, 0, 1).contiguous()

    # colour key: one (gray) or three (RGB) big-endian 16-bit samples; an 8-bit image uses their low bytes
    n_samples = 1 if color_type == 0 else 3
    if bit_depth != 8 or trns is None or len(trns) < 2 * n_samples:
        return image
    key = torch.tensor([trns[2 * i + 1] for i in range(n_samples)], dtype=torch.uint8, device=image.device)
    transparent = (image == key[:, None, None]).all(dim=0, keepdim=True)
    alpha = torch.where(transparent, 0, 255).to(torch.uint8)
    return torch.cat([image.expand(3, -1, -1), alpha], dim=0)


def _load_image_to_tensor(path_file: Path, device: Union[str, torch.device, None]) -> torch.Tensor:
    """Read an image file and decode using the Kornia Rust backend.

    The decoded image is returned as numpy array with shape HxWxC.

    Args:
        path_file: Path to an image file in a format kornia-rs decodes; the extension may be in any letter case.
        device: the device where you want to get your image placed.

    Return:
        Image torch.Tensor with shape :math:`(C, H, W)`, as decoded.

    """
    # read image and return as `np.ndarray` with shape HxWxC
    try:
        # open the file first so Python raises the specific OSError subclass
        # (FileNotFoundError, IsADirectoryError, PermissionError) naming the path
        # for every format, instead of kornia_rs's bare OSError without the name
        with open(path_file, "rb"):
            pass
        if path_file.suffix.lower() in [".jpg", ".jpeg"]:
            img = _rs_io.read_image_jpegturbo(str(path_file))
            if _read_jpeg_components(path_file) == 1:
                # libjpeg-turbo decodes a grayscale JPEG into three equal channels; keep the file's one channel
                # in its own buffer, so the three-channel decode is released
                img = np.ascontiguousarray(img[..., :1])
        elif path_file.suffix.lower() == ".png":
            # None (truncated/invalid header) intentionally falls through to read_image
            bit_depth, color_type = _read_png_header(path_file) or (None, None)
            mode = _PNG_COLOR_TYPE_TO_MODE.get(color_type)
            if mode is None or mode == "rgb":
                # RGB is the default of read_image; use it for unknown types too
                img = _rs_io.read_image(str(path_file))
            elif bit_depth == 16:
                # the u8 reader rejects a 16-bit buffer; UNCHANGED loads keep it as uint16
                img = _rs_io.read_image_png_u16(str(path_file), mode)
            else:
                img = _rs_io.read_image_png_u8(str(path_file), mode)
        else:
            img = _rs_io.read_image(str(path_file))
    except OSError:
        raise  # a missing or unreadable file keeps its specific type (FileNotFoundError, PermissionError, ...)
    except Exception as e:
        raise ValueError(f"Failed to decode image '{path_file}': {e}") from e

    # convert the image to torch.Tensor with shape CxHxW
    img_t = image_to_tensor(img, keepdim=True)

    # move the torch.Tensor to the desired device,
    dev = device if isinstance(device, torch.device) or device is None else torch.device(device)

    return img_t.to(device=dev)


def _to_float32(image: torch.Tensor) -> torch.Tensor:
    """Convert an image torch.Tensor to float32."""
    KORNIA_CHECK(image.dtype == torch.uint8)
    return image.float() / 255.0


def _to_uint8(image: torch.Tensor) -> torch.Tensor:
    """Convert a float image in [0, 1] to uint8, rounding to the nearest level."""
    KORNIA_CHECK(image.dtype == torch.float32)
    return image.mul(255.0).round_().clamp_(0, 255).to(torch.uint8)


def _convert_image_type(image: torch.Tensor, desired_type: ImageLoadType) -> torch.Tensor:
    """Convert a raw CxHxW uint8 image tensor to the desired type.

    ``UNCHANGED`` returns whatever the decoder produced, so a 16-bit PNG or TIFF comes back as ``uint16`` and a
    float TIFF as ``float32``. Every other load type is defined for 8-bit input; asking for one on a wider
    decode raises rather than mislabelling it.
    """
    channels = image.shape[0]

    if desired_type == ImageLoadType.UNCHANGED:
        return image
    if image.dtype != torch.uint8:
        raise NotImplementedError(
            f"The file decoded to {image.dtype}, and ImageLoadType.{desired_type.name} is defined for 8-bit images "
            "only; load it with ImageLoadType.UNCHANGED and convert the tensor yourself."
        )

    # Normalize RGBA to RGB for types that don't need alpha
    if channels == 4 and desired_type != ImageLoadType.RGBA8:
        image = _to_uint8(kornia.color.rgba_to_rgb(_to_float32(image)))
        channels = 3

    match (desired_type, channels):
        case (ImageLoadType.GRAY8, 1):
            return image
        case (ImageLoadType.GRAY8, 3):
            return kornia.color.rgb_to_grayscale(image)
        case (ImageLoadType.RGB8, 3):
            return image
        case (ImageLoadType.RGB8, 1):
            return kornia.color.grayscale_to_rgb(image)
        case (ImageLoadType.RGBA8, 4):
            return image
        case (ImageLoadType.RGBA8, 3):
            return _to_uint8(kornia.color.rgb_to_rgba(_to_float32(image), 1.0))
        case (ImageLoadType.RGBA8, 1):
            return _to_uint8(kornia.color.rgb_to_rgba(kornia.color.grayscale_to_rgb(_to_float32(image)), 1.0))
        case (ImageLoadType.GRAY32, 1):
            return _to_float32(image)
        case (ImageLoadType.GRAY32, 3):
            return kornia.color.rgb_to_grayscale(_to_float32(image))
        case (ImageLoadType.RGB32, 3):
            return _to_float32(image)
        case (ImageLoadType.RGB32, 1):
            return kornia.color.grayscale_to_rgb(_to_float32(image))
        case _:
            raise NotImplementedError(f"Unsupported conversion: {channels} channels to {desired_type}")


def load_image(
    path_file: str | Path,
    desired_type: ImageLoadType = ImageLoadType.RGB32,
    device: Union[str, torch.device, None] = "cpu",
) -> torch.Tensor:
    """Read an image file and decode using the Kornia Rust backend.

    Convention:
        - ``desired_type`` fixes the channels, the dtype and the range, whatever the file holds: ``GRAY8``, ``RGB8``
          and ``RGBA8`` give 1, 3 and 4 ``uint8`` channels in [0, 255], and ``GRAY32`` and ``RGB32`` (the default)
          1 and 3 ``float32`` channels in [0, 1]. ``UNCHANGED`` keeps the file's channels and dtype, so a 16-bit PNG
          or TIFF loads as ``uint16`` and a float TIFF as ``float32``; the other types raise for such a file.
        - Channels are in RGB order, with alpha last. Under every type except ``UNCHANGED``, an 8-bit palette PNG
          loads as its colours and a PNG ``tRNS`` chunk counts as alpha; ``RGBA8`` keeps the alpha (255 for an
          opaque file), and the other types composite it over white, rounded to the nearest level. ``UNCHANGED``
          returns a palette PNG's indices and keeps an alpha channel as stored.
        - ``GRAY8`` and ``GRAY32`` convert a colour file with :func:`~kornia.color.rgb_to_grayscale`.
        - The pixels come back as stored: the EXIF orientation is ignored.
        - The extension, in any letter case, chooses the decoder; the file content does not. The file is decoded on
          the CPU and the tensor is then moved to ``device``.
        - kornia-rs cannot decode a PNG with 1, 2 or 4 bits per sample, a gray+alpha PNG, a 4-channel TIFF or a CMYK
          JPEG; these, a truncated file and, under every type except ``UNCHANGED``, a malformed palette or
          ``PLTE``/``tRNS`` chunk raise ``ValueError``.
        - :ref:`Files and arrays <image-io-conventions>` maps OpenCV's and Pillow's loaders onto the load types.

    Args:
        path_file: Path to an image file in a format kornia-rs decodes, such as JPEG, PNG, TIFF or WebP; the
            extension may be in any letter case.
        desired_type: the desired image type, defined by color space and dtype.
        device: the device where you want to get your image placed.

    Return:
        Image torch.Tensor with shape :math:`(C, H, W)`.

    Raises:
        TypeCheckError: if ``desired_type`` is not an :class:`ImageLoadType` (also a ``TypeError``).
        OSError: if the file cannot be read, for example ``FileNotFoundError`` when it does not exist.
        ValueError: if the file cannot be decoded; the message names the file.
        NotImplementedError: if a type other than ``UNCHANGED`` is requested for a file wider than 8 bits.

    """
    if not isinstance(desired_type, ImageLoadType):
        raise TypeCheckError(
            f"desired_type must be an ImageLoadType, got {desired_type!r}.",
            actual_type=type(desired_type),
            expected_type=ImageLoadType,
        )
    if not isinstance(path_file, Path):
        path_file = Path(path_file)

    # read the image using the kornia_rs package
    image: torch.Tensor = _load_image_to_tensor(path_file, device)  # CxHxW

    if desired_type != ImageLoadType.UNCHANGED:
        image = _apply_png_palette_and_transparency(path_file, image)

    return _convert_image_type(image, desired_type)


def _write_uint8_image(path_file: Path, img_np: Any, quality: int) -> None:
    """Write uint8 image to file."""
    mode = _color_mode(img_np)
    if path_file.suffix.lower() in [".jpg", ".jpeg"]:
        _rs_io.write_image_jpeg(str(path_file), img_np, mode=mode, quality=quality)
    elif path_file.suffix.lower() == ".png":
        _rs_io.write_image_png_u8(str(path_file), img_np, mode=mode)
    elif path_file.suffix.lower() in (".tif", ".tiff"):
        _rs_io.write_image_tiff_u8(str(path_file), img_np, mode=mode)
    else:
        raise NotImplementedError(f"Unsupported file extension: {path_file.suffix} for uint8 image")


def _write_uint16_image(path_file: Path, img_np: Any) -> None:
    """Write uint16 image to file."""
    mode = _color_mode(img_np)
    if path_file.suffix.lower() == ".png":
        _rs_io.write_image_png_u16(str(path_file), img_np, mode=mode)
    elif path_file.suffix.lower() in (".tif", ".tiff"):
        _rs_io.write_image_tiff_u16(str(path_file), img_np, mode=mode)
    else:
        raise NotImplementedError(f"Unsupported file extension: {path_file.suffix} for uint16 image")


def _write_float32_image(path_file: Path, img_np: Any) -> None:
    """Write float32 image to file."""
    mode = _color_mode(img_np)
    if path_file.suffix.lower() in (".tif", ".tiff"):
        _rs_io.write_image_tiff_f32(str(path_file), img_np, mode=mode)
    else:
        raise NotImplementedError(f"Unsupported file extension: {path_file.suffix} for float32 image")


def write_image(path_file: str | Path, image: torch.Tensor, quality: int = 80) -> None:
    """Save an image file using the Kornia Rust backend.

    Convention:
        - The extension chooses the format, and the format limits the dtype and the channels: JPEG stores ``uint8``,
          PNG ``uint8`` and ``uint16``, and TIFF ``uint8``, ``uint16`` and ``float32``; each stores 1 or 3 channels,
          and PNG also 4. Any other dtype or channel count raises.
        - The tensor is written as it is, in RGB order with alpha last, without conversion, scaling or clamping: a
          ``float32`` TIFF keeps values outside [0, 1].
        - ``image`` is channels first. A channels-last :math:`(H, W, C)` tensor is not detected: its first axis is
          read as the channels.
        - :ref:`Files and arrays <image-io-conventions>` maps ``cv2.imwrite`` onto these arguments.
        - Known defect: an image with zero height or width reaches kornia-rs, which raises a bare ``Exception`` and
          leaves an empty or truncated file behind (`#5770 <https://github.com/kornia/kornia/issues/5770>`_).

    Args:
        path_file: Path with a .jpg, .jpeg, .png, .tif or .tiff extension, in any letter case.
        image: Image torch.Tensor with shape :math:`(C, H, W)` or :math:`(H, W)`.
        quality: The quality of the JPEG encoding, an integer in [0, 100] and not a ``bool``. It is validated for
            every format and used only for JPEG.

    Return:
        None.

    Raises:
        TypeCheckError: if ``image`` is not a tensor or ``quality`` is not an integer (also a ``TypeError``).
        ValueCheckError: if ``quality`` is outside [0, 100] (also a ``ValueError``).
        BaseError: if the extension is not one of the five above.
        ShapeError: if ``image`` is not 2-D or 3-D, or the format cannot store its channels (also a ``ValueError``).
        FileNotFoundError: if the directory to write into does not exist (``NotADirectoryError`` if it is a file).
        NotImplementedError: if the format cannot store the dtype.
        Exception: kornia-rs's bare ``Exception`` when the file cannot be created or the image cannot be encoded.
    """
    if not isinstance(path_file, Path):
        path_file = Path(path_file)

    KORNIA_CHECK_IS_TENSOR(image)
    if not isinstance(quality, numbers.Integral) or isinstance(quality, bool):
        raise TypeCheckError(
            f"quality must be an integer in [0, 100], got {quality!r}.",
            actual_type=type(quality),
            expected_type=numbers.Integral,
        )
    if not 0 <= quality <= 100:
        raise ValueCheckError(
            f"quality must be an integer in [0, 100], got {quality!r}.", actual_value=quality, expected_range=(0, 100)
        )
    suffix = path_file.suffix.lower()
    KORNIA_CHECK(
        suffix in _WRITE_EXTENSIONS,
        f"Invalid file extension: {path_file}, only .jpg, .jpeg, .png, .tif and .tiff are supported.",
    )
    if image.dim() not in (2, 3):
        raise ShapeError(
            f"Invalid image shape: {tuple(image.shape)}. Expected (C, H, W) or (H, W).",
            actual_shape=tuple(image.shape),
            expected_shape=["C", "H", "W"],
        )

    parent = path_file.parent
    if not parent.is_dir():
        if parent.exists():
            raise NotADirectoryError(errno.ENOTDIR, f"Cannot write {path_file.name} into a file", str(parent))
        raise FileNotFoundError(errno.ENOENT, f"No such directory to write {path_file.name} into", str(parent))

    img_np = tensor_to_image(image, keepdim=True, force_contiguous=True)  # HxWxC
    if img_np.ndim == 2:
        img_np = img_np[..., None]  # ensures channel dimension
    channels = img_np.shape[-1]
    supported = (1, 3, 4) if suffix == ".png" else (1, 3)
    if channels not in supported:
        raise ShapeError(
            f"Cannot write a {channels}-channel image to {suffix}; the supported channel counts are {supported}.",
            actual_shape=tuple(image.shape),
        )
    if image.dtype == torch.uint8:
        _write_uint8_image(path_file, img_np, quality)
    elif image.dtype == torch.uint16:
        _write_uint16_image(path_file, img_np)
    elif image.dtype == torch.float32:
        _write_float32_image(path_file, img_np)
    else:
        raise NotImplementedError(f"Unsupported image dtype: {image.dtype}")
