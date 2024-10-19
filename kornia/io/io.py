from __future__ import annotations

from enum import Enum
from pathlib import Path

import kornia_rs
import torch

import kornia
from kornia.core import Device, Tensor
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


def _load_image_to_tensor(path_file: Path, device: Device) -> Tensor:
    """Read an image file and decode using the Kornia Rust backend.

    The decoded image is returned as numpy array with shape HxWxC.

    Args:
        path_file: Path to a valid image file.
        device: the device where you want to get your image placed.

    Return:
        Image tensor with shape :math:`(3,H,W)`.
    """

    # read image and return as `np.ndarray` with shape HxWxC
    if path_file.suffix.lower() in [".jpg", ".jpeg"]:
        img = kornia_rs.read_image_jpeg(str(path_file))
    else:
        img = kornia_rs.read_image_any(str(path_file))

    # convert the image to tensor with shape CxHxW
    img_t = image_to_tensor(img, keepdim=True)

    # move the tensor to the desired device,
    dev = device if isinstance(device, torch.device) or device is None else torch.device(device)

    return img_t.to(device=dev)


def _to_float32(image: Tensor) -> Tensor:
    """Convert an image tensor to float32."""
    KORNIA_CHECK(image.dtype == torch.uint8)
    return image.float() / 255.0


def _to_uint8(image: Tensor) -> Tensor:
    """Convert an image tensor to uint8."""
    KORNIA_CHECK(image.dtype == torch.float32)
    return image.mul(255.0).byte()


def load_image(
    path_file: str | Path, desired_type: ImageLoadType = ImageLoadType.RGB32, device: Device = "cpu"
) -> Tensor:
    """Read an image file and decode using the Kornia Rust backend.

    Args:
        path_file: Path to a valid image file.
        desired_type: the desired image type, defined by color space and dtype.
        device: the device where you want to get your image placed.

    Return:
        Image tensor with shape :math:`(3,H,W)`.
    """
    if not isinstance(path_file, Path):
        path_file = Path(path_file)

    # read the image using the kornia_rs package
    image: Tensor = _load_image_to_tensor(path_file, device)  # CxHxW

    if desired_type == ImageLoadType.UNCHANGED:
        return image
    elif desired_type == ImageLoadType.GRAY8:
        if image.shape[0] == 1 and image.dtype == torch.uint8:
            return image
        elif image.shape[0] == 3 and image.dtype == torch.uint8:
            gray8 = kornia.color.rgb_to_grayscale(image)
            return gray8
        elif image.shape[0] == 4 and image.dtype == torch.uint8:
            gray32 = kornia.color.rgb_to_grayscale(kornia.color.rgba_to_rgb(_to_float32(image)))
            return _to_uint8(gray32)

    elif desired_type == ImageLoadType.RGB8:
        if image.shape[0] == 3 and image.dtype == torch.uint8:
            return image
        elif image.shape[0] == 1 and image.dtype == torch.uint8:
            rgb8 = kornia.color.grayscale_to_rgb(image)
            return rgb8

    elif desired_type == ImageLoadType.RGBA8:
        if image.shape[0] == 3 and image.dtype == torch.uint8:
            rgba32 = kornia.color.rgb_to_rgba(_to_float32(image), 0.0)
            return _to_uint8(rgba32)

    elif desired_type == ImageLoadType.GRAY32:
        if image.shape[0] == 1 and image.dtype == torch.uint8:
            return _to_float32(image)
        elif image.shape[0] == 3 and image.dtype == torch.uint8:
            gray32 = kornia.color.rgb_to_grayscale(_to_float32(image))
            return gray32
        elif image.shape[0] == 4 and image.dtype == torch.uint8:
            gray32 = kornia.color.rgb_to_grayscale(kornia.color.rgba_to_rgb(_to_float32(image)))
            return gray32

    elif desired_type == ImageLoadType.RGB32:
        if image.shape[0] == 3 and image.dtype == torch.uint8:
            return _to_float32(image)
        elif image.shape[0] == 1 and image.dtype == torch.uint8:
            rgb32 = kornia.color.grayscale_to_rgb(_to_float32(image))
            return rgb32

    raise NotImplementedError(f"Unknown type: {desired_type}")


def write_image(path_file: str | Path, image: Tensor) -> None:
    """Save an image file using the Kornia Rust backend.

    For now, we only support the writing of JPEG of the following types: RGB8.

    Args:
        path_file: Path to a valid image file.
        image: Image tensor with shape :math:`(3,H,W)`, `(1,H,W)` and `(H,W)`.

    Return:
        None.
    """
    if not isinstance(path_file, Path):
        path_file = Path(path_file)

    KORNIA_CHECK(
        path_file.suffix in [".jpg", ".jpeg"],
        f"Invalid file extension: {path_file}, only .jpg and .jpeg are supported.",
    )

    if (image.dim() == 3 and image.shape[0] == 1) or image.dim() == 2:  # Grayscale image
        image = image.squeeze(0)  # HxW
        image = image[None, :, :].repeat(3, 1, 1)  # Convert to HxWx3

    KORNIA_CHECK(image.dim() == 3 and image.shape[0] == 3, f"Invalid image shape: {image.shape}")
    KORNIA_CHECK(image.dtype == torch.uint8, f"Invalid image dtype: {image.dtype}")

    # convert the tensor to numpy
    img_np = tensor_to_image(image, keepdim=True, force_contiguous=True)  # HxWx3

    # save the image using the
    kornia_rs.write_image_jpeg(str(path_file), img_np)
