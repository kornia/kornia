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
import numpy as np
import torch

import kornia
from kornia.core.check import KORNIA_CHECK
from kornia.utils import image_to_tensor, tensor_to_image


class ImageLoadType(Enum):
    r"""Enum to specify the desired image type."""

    UNCHANGED = 0
    GRAY8 = 1
    RGB8 = 2
    RGBA8 = 3
    GRAY32 = 4
    RGB32 = 5


def _load_image_to_tensor(path_file: Path, device: Union[str, torch.device, None]) -> torch.Tensor:
    """Read an image file and decode using the Kornia Rust backend.

    The decoded image is returned as numpy array with shape HxWxC.

    Args:
        path_file: Path to a valid image file.
        device: the device torch.where you want to get your image placed.

    Return:
        Image torch.tensor with shape :math:`(3,H,W)`.

    """
    # read image and return as `np.ndarray` with shape HxWxC
    if path_file.suffix.lower() in [".jpg", ".jpeg"]:
        img = kornia_rs.read_image_jpegturbo(str(path_file))
    else:
        img = kornia_rs.read_image(str(path_file))
    # convert the image to torch.tensor with shape CxHxW
    img_t = image_to_tensor(img, keepdim=True)

    # move the torch.tensor to the desired device,
    dev = device if isinstance(device, torch.device) or device is None else torch.device(device)

    return img_t.to(device=dev)


def _to_float32(image: torch.Tensor) -> torch.Tensor:
    """Convert an image tensor to float32 and normalize based on its dtype.

    Args:
        image: The input image tensor.

    Returns:
        The normalized float32 image tensor.

    Raises:
        NotImplementedError: if the image dtype is not supported.
    """
    if image.dtype == torch.uint8:
        return image.float() / 255.0
    elif image.dtype == torch.uint16:
        return image.float() / 65535.0
    elif image.dtype == torch.float32:
        return image
    else:
        raise NotImplementedError(f"Unsupported dtype: {image.dtype}")


def _to_float16(image: torch.Tensor) -> torch.Tensor:
    if image.dtype == torch.uint8:
        return (image.float() / 255.0).half()
    elif image.dtype == torch.uint16:
        return (image.float() / 65535.0).half()
    elif image.dtype == torch.float32:
        return image.half()
    elif image.dtype == torch.float16:
        return image
    else:
        raise NotImplementedError(f"Unsupported dtype: {image.dtype}")


def _to_uint8(image: torch.Tensor) -> torch.Tensor:
    """Convert an image tensor to uint8.

    Args:
        image: The input image tensor.

    Returns:
        The uint8 image tensor.

    Raises:
        NotImplementedError: if the image dtype is not supported.
    """
    if image.dtype == torch.float32:
        return torch.round(image * 255.0).clamp(0, 255).to(torch.uint8)
    elif image.dtype == torch.uint16:
        return (image.to(torch.float32) / 256.0).to(torch.uint8)
    elif image.dtype == torch.uint8:
        return image
    else:
        raise NotImplementedError(f"Unsupported dtype: {image.dtype}")


def load_image(
    path_file: str | Path,
    desired_type: ImageLoadType = ImageLoadType.RGB32,
    device: Union[str, torch.device, None] = "cpu",
) -> torch.Tensor:
    """Read an image file and decode using the Kornia Rust backend.

    Args:
        path_file: Path to a valid image file.
        desired_type: the desired image type, defined by color space and dtype.
        device: the device torch.where you want to get your image placed.

    Return:
        Image torch.tensor with shape :math:`(3,H,W)`.

    """
    if not isinstance(path_file, Path):
        path_file = Path(path_file)

    # read the image using the kornia_rs package
    image: torch.Tensor = _load_image_to_tensor(path_file, device)  # CxHxW

    if desired_type == ImageLoadType.UNCHANGED:
        return image
    elif desired_type == ImageLoadType.GRAY8:
        if image.shape[0] == 1:
            return _to_uint8(image)
        elif image.shape[0] == 3:
            gray32 = kornia.color.rgb_to_grayscale(_to_float32(image))
            return _to_uint8(gray32)
        elif image.shape[0] == 4:
            gray32 = kornia.color.rgb_to_grayscale(kornia.color.rgba_to_rgb(_to_float32(image)))
            return _to_uint8(gray32)

    elif desired_type == ImageLoadType.RGB8:
        if image.shape[0] == 3:
            return _to_uint8(image)
        elif image.shape[0] == 1:
            rgb32 = kornia.color.grayscale_to_rgb(_to_float32(image))
            return _to_uint8(rgb32)

    elif desired_type == ImageLoadType.RGBA8:
        if image.shape[0] == 3:
            rgba32 = kornia.color.rgb_to_rgba(_to_float32(image), 0.0)
            return _to_uint8(rgba32)

    elif desired_type == ImageLoadType.GRAY32:
        if image.shape[0] == 1:
            return _to_float32(image)
        elif image.shape[0] == 3:
            gray32 = kornia.color.rgb_to_grayscale(_to_float32(image))
            return gray32
        elif image.shape[0] == 4:
            gray32 = kornia.color.rgb_to_grayscale(kornia.color.rgba_to_rgb(_to_float32(image)))
            return gray32

    elif desired_type == ImageLoadType.RGB32:
        if image.shape[0] == 3:
            return _to_float32(image)
        elif image.shape[0] == 1:
            rgb32 = kornia.color.grayscale_to_rgb(_to_float32(image))
            return rgb32

    raise NotImplementedError(f"Unknown type: {desired_type}")


def _detect_image_mode(img_np: Any) -> str:
    """Detect the image mode from a numpy-like image array.

    Args:
        img_np: The input image array with shape (H, W) or (H, W, C).

    Returns:
        The detected mode: "mono", "rgb", or "rgba".

    Raises:
        ValueError: if the number of channels is not 1, 3, or 4.
    """
    if img_np.ndim == 2:
        return "mono"
    channels = img_np.shape[-1]
    if channels == 1:
        return "mono"
    elif channels == 3:
        return "rgb"
    elif channels == 4:
        return "rgba"
    else:
        raise ValueError(f"Unsupported number of channels: {channels}. Expected 1, 3, or 4.")


def _write_uint8_image(path_file: Path, img_np: Any, quality: int | None) -> None:
    """Write a uint8 image to a file.

    Args:
        path_file: The path to the output file.
        img_np: The image array to write.
        quality: The JPEG quality (only for JPEG files).

    Raises:
        TypeError: if the image is not uint8.
        NotImplementedError: if the file extension is not supported.
    """
    if img_np.dtype != np.uint8:
        raise TypeError(f"Expected uint8 image, got {img_np.dtype}")
    suffix = path_file.suffix.lower()
    mode = _detect_image_mode(img_np)
    if suffix in [".jpg", ".jpeg"]:
        if quality is None:
            quality = 80
        kornia_rs.write_image_jpeg(str(path_file), img_np, mode=mode, quality=quality)
    elif suffix == ".png":
        kornia_rs.write_image_png_u8(str(path_file), img_np, mode=mode)
    elif suffix == ".tiff":
        kornia_rs.write_image_tiff_u8(str(path_file), img_np, mode=mode)
    else:
        raise NotImplementedError(f"Unsupported file extension: {path_file.suffix} for uint8 image")


def _write_uint16_image(path_file: Path, img_np: Any) -> None:
    """Write a uint16 image to a file.

    Args:
        path_file: The path to the output file.
        img_np: The image array to write.

    Raises:
        TypeError: if the image is not uint16.
        NotImplementedError: if the file extension is not supported.
    """
    if img_np.dtype != np.uint16:
        raise TypeError(f"Expected uint16 image, got {img_np.dtype}")
    suffix = path_file.suffix.lower()
    mode = _detect_image_mode(img_np)
    if suffix == ".png":
        kornia_rs.write_image_png_u16(str(path_file), img_np, mode=mode)
    elif suffix == ".tiff":
        kornia_rs.write_image_tiff_u16(str(path_file), img_np, mode=mode)
    else:
        raise NotImplementedError(f"Unsupported file extension: {path_file.suffix} for uint16 image")


def _write_float32_image(path_file: Path, img_np: Any) -> None:
    """Write a float32 image to a file.

    Args:
        path_file: The path to the output file.
        img_np: The image array to write.

    Raises:
        TypeError: if the image is not float32.
        NotImplementedError: if the file extension is not supported.
    """
    if img_np.dtype != np.float32:
        raise TypeError(f"Expected float32 image, got {img_np.dtype}")
    suffix = path_file.suffix.lower()
    mode = _detect_image_mode(img_np)
    if suffix == ".tiff":
        kornia_rs.write_image_tiff_f32(str(path_file), img_np, mode=mode)
    else:
        raise NotImplementedError(f"Unsupported file extension: {path_file.suffix} for float32 image")


def write_image(path_file: str | Path, image: torch.Tensor, quality: int | None = None) -> None:
    """Save an image file using the Kornia Rust backend.

    Args:
        path_file: Path to a valid image file.
        image: Image torch.tensor with shape :math:`(3,H,W)`, `(1,H,W)` and `(H,W)`.
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
