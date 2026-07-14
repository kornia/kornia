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

from enum import Enum
from pathlib import Path
from typing import Any, Union

import kornia_rs
import torch

import kornia
from kornia.core.check import KORNIA_CHECK
from kornia.image.utils import image_to_tensor, tensor_to_image


class ImageLoadType(Enum):
    r"""Enum to specify the desired image type."""

    UNCHANGED = 0
    GRAY8 = 1
    RGB8 = 2
    RGBA8 = 3
    GRAY32 = 4
    RGB32 = 5


_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def _read_png_color_type(path_file: Path) -> int | None:
    """Read the color type byte from a PNG file header.

    Returns None if the file is truncated or has an invalid PNG signature.
    PNG color types: 0=Grayscale, 2=RGB, 3=Indexed, 4=Grayscale+Alpha, 6=RGBA.
    """
    with open(path_file, "rb") as f:
        # Need 26 bytes: 8 (signature) + 8 (IHDR chunk header) + 8 (width/height) + 1 (bit depth) + 1 (color type)
        header = f.read(26)
        if len(header) < 26 or header[:8] != _PNG_SIGNATURE:
            return None
        return header[25]


# Map PNG color type byte to kornia_rs read mode.
# Types not listed here fall through to read_image_any (which handles RGB).
# Note: color type 4 (Grayscale+Alpha) is intentionally omitted — kornia_rs
# does not support it, so it falls through to read_image_any as RGB.
_PNG_COLOR_TYPE_TO_MODE: dict[int, str] = {
    0: "mono",  # Grayscale
    3: "mono",  # Indexed (palette) — decoded as single channel by kornia_rs
    6: "rgba",  # RGBA
}


def _load_image_to_tensor(path_file: Path, device: Union[str, torch.device, None]) -> torch.Tensor:
    """Read an image file and decode using the Kornia Rust backend.

    The decoded image is returned as numpy array with shape HxWxC.

    Args:
        path_file: Path to a valid image file.
        device: the device where you want to get your image placed.

    Return:
        Image torch.Tensor with shape :math:`(3,H,W)`.

    """
    # read image and return as `np.ndarray` with shape HxWxC
    if path_file.suffix.lower() in [".jpg", ".jpeg"]:
        img = kornia_rs.read_image_jpegturbo(str(path_file))
    elif path_file.suffix.lower() == ".png":
        color_type = _read_png_color_type(path_file)
        # None (truncated/invalid header) intentionally falls through to read_image_any
        mode = _PNG_COLOR_TYPE_TO_MODE.get(color_type)
        if mode is None or mode == "rgb":
            # RGB is the default for read_image_any; use it for unknown types too
            img = kornia_rs.read_image_any(str(path_file))
        else:
            img = kornia_rs.read_image_png_u8(str(path_file), mode)
    else:
        img = kornia_rs.read_image_any(str(path_file))

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
    """Convert an image torch.Tensor to uint8."""
    KORNIA_CHECK(image.dtype == torch.float32)
    return image.mul(255.0).byte()


def _convert_image_type(image: torch.Tensor, desired_type: ImageLoadType) -> torch.Tensor:
    """Convert a raw CxHxW uint8 image tensor to the desired type."""
    channels = image.shape[0]

    if desired_type == ImageLoadType.UNCHANGED:
        return image

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
            return _to_uint8(kornia.color.rgb_to_rgba(_to_float32(image), 0.0))
        case (ImageLoadType.RGBA8, 1):
            return _to_uint8(kornia.color.rgb_to_rgba(kornia.color.grayscale_to_rgb(_to_float32(image)), 0.0))
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

    Args:
        path_file: Path to a valid image file.
        desired_type: the desired image type, defined by color space and dtype.
        device: the device where you want to get your image placed.

    Return:
        Image torch.Tensor with shape :math:`(3,H,W)`.

    """
    if not isinstance(path_file, Path):
        path_file = Path(path_file)

    # read the image using the kornia_rs package
    image: torch.Tensor = _load_image_to_tensor(path_file, device)  # CxHxW

    return _convert_image_type(image, desired_type)


def _write_uint8_image(path_file: Path, img_np: Any, quality: int) -> None:
    """Write uint8 image to file."""
    if path_file.suffix.lower() in [".jpg", ".jpeg"]:
        mode = "mono" if img_np.ndim == 2 or (img_np.ndim == 3 and img_np.shape[-1] == 1) else "rgb"
        kornia_rs.write_image_jpeg(str(path_file), img_np, mode=mode, quality=quality)
    elif path_file.suffix.lower() == ".png":
        mode = "mono" if img_np.ndim == 2 or (img_np.ndim == 3 and img_np.shape[-1] == 1) else "rgb"
        kornia_rs.write_image_png_u8(str(path_file), img_np, mode=mode)
    elif path_file.suffix.lower() == ".tiff":
        mode = "mono" if img_np.ndim == 2 or (img_np.ndim == 3 and img_np.shape[-1] == 1) else "rgb"
        kornia_rs.write_image_tiff_u8(str(path_file), img_np, mode=mode)
    else:
        raise NotImplementedError(f"Unsupported file extension: {path_file.suffix} for uint8 image")


def _write_uint16_image(path_file: Path, img_np: Any) -> None:
    """Write uint16 image to file."""
    mode = "mono" if img_np.ndim == 2 or (img_np.ndim == 3 and img_np.shape[-1] == 1) else "rgb"
    if path_file.suffix.lower() == ".png":
        kornia_rs.write_image_png_u16(str(path_file), img_np, mode=mode)
    elif path_file.suffix.lower() == ".tiff":
        kornia_rs.write_image_tiff_u16(str(path_file), img_np, mode=mode)
    else:
        raise NotImplementedError(f"Unsupported file extension: {path_file.suffix} for uint16 image")


def _write_float32_image(path_file: Path, img_np: Any) -> None:
    """Write float32 image to file."""
    mode = "mono" if img_np.ndim == 2 or (img_np.ndim == 3 and img_np.shape[-1] == 1) else "rgb"
    if path_file.suffix.lower() == ".tiff":
        kornia_rs.write_image_tiff_f32(str(path_file), img_np, mode=mode)
    else:
        raise NotImplementedError(f"Unsupported file extension: {path_file.suffix} for float32 image")


def write_image(path_file: str | Path, image: torch.Tensor, quality: int = 80) -> None:
    """Save an image file using the Kornia Rust backend.

    Args:
        path_file: Path to a valid image file.
        image: Image torch.Tensor with shape :math:`(3,H,W)`, `(1,H,W)` and `(H,W)`.
        quality: The quality of the JPEG encoding. If the file extension is .png or .tiff, the quality is ignored.

    Return:
        None.
    """
    if not isinstance(path_file, Path):
        path_file = Path(path_file)

    KORNIA_CHECK(
        path_file.suffix in [".jpg", ".jpeg", ".png", ".tiff"],
        f"Invalid file extension: {path_file}, only .jpg, .jpeg, .png and .tiff are supported.",
    )

    KORNIA_CHECK(image.dim() >= 2, f"Invalid image shape: {image.shape}. Must be at least 2D.")

    img_np = tensor_to_image(image, keepdim=True, force_contiguous=True)  # HxWxC
    if img_np.ndim == 2:
        img_np = img_np[..., None]  # ensures channel dimension
    if image.dtype == torch.uint8:
        _write_uint8_image(path_file, img_np, quality)
    elif image.dtype == torch.uint16:
        _write_uint16_image(path_file, img_np)
    elif image.dtype == torch.float32:
        _write_float32_image(path_file, img_np)
    else:
        raise NotImplementedError(f"Unsupported image dtype: {image.dtype}")
